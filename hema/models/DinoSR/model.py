import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II

from EMA import EMA
from utils import compute_mask_indices, index_put, GradMultiply 
from nn.conv_feature_extractor import FeatureExtractor
from conformer.model import Conformer

MASKING_DISTRIBUTION_CHOICES = (["static", "uniform", "normal", "poisson"])

@dataclass
class config():
    # Codebook related settings
    codebook_size: int = field(default=256)
    normal_init_codebook: bool = field(default=False)
    codebook_init_decay: float = field(default=0.9)
    codebook_end_decay: float = field(default=0.9)
    codebook_end_decay_step: float = field(default=0)
    
    # Training related settings
    freeze_teacher_step: int = field(default=2e5)
    freeze_pre_enc_modules: bool = field(default=True)
    
    # Loss related settings
    loss_beta: float = field(default=0, metadata={"help": "beta for smooth L1 loss, 0 means use L2 loss."})
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8,
        metadata={"help": "How many layers to average"}
    )
    
    # EMA related settings
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    # Update related settings
    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )
    
    # Model topology related settings
    encoder_embed_dim: float = field(default=768)
    
    # Masking related settings
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: str = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )
    
    # Conformer Encoder related settings
    num_encoder_layers: int = field(default=6)
    num_attention_heads: int = field(default=8)
    
    # Normalization related settings
    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


CONV_LAYERS = [(768, 9, 2), (768, 5, 2), (768, 5, 2)]
class DinoSR(torch.nn.Module):
    def __init__(self, cfg: config):
        super().__init__()
        
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(CONV_LAYERS)
        self.extractor_dim = CONV_LAYERS[-1][0]
        
        self.ema = None
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        # From conv blocks to encoder embedding dimensions
        self.post_extract_proj = torch.nn.Linear(self.extractor_dim, cfg.encoder_embed_dim)
        
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        
        # MASK vector
        self.mask_emb = torch.nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        
        # Conformer Encoder
        self.conformer_encoder = Conformer(self.embed, self.embed, cfg.num_encoder_layers, cfg.num_attention_heads)
        
        # Codebooks
        self.pre_encoder_copied = False
        if self.discrete:
            assert cfg.instance_norm_target_layer
            assert not (cfg.layer_norm_target or cfg.instance_norm_targets)
            
            self.codebook_size = cfg.codebook_size
            self.n_codebooks = cfg.average_top_k_layers
            self.codebook_decay = cfg.codebook_init_decay
            # Prediction Heads
            self.heads = torch.nn.ModuleList([
                torch.nn.Linear(
                    cfg.encoder_embed_dim,
                    cfg.codebook_size,
                )
                for i in range(self.n_codebooks)
            ])
            
            # Codebook: use dictionary to store so codebooks are always in fp32
            if cfg.normal_init_codebook:
                codebooks = torch.normal(0.0, (1 / self.codebook_size**0.5),
                            size=(self.n_codebooks, self.codebook_size, cfg.encoder_embed_dim))
            else:
                codebooks = torch.randn(self.n_codebooks, cfg.encoder_embed_dim, self.codebook_size)
                codebooks = F.instance_norm(codebooks).transpose(1,2)
            self.codebooks = {
                i:codebooks[i] for i in range(self.n_codebooks)
            }
            self.codebook_cnts = {
                i:torch.ones([self.codebook_size]) for i in range(self.n_codebooks)
            }
            self.shared_module_state_dict = None
        
        self.num_updates = 0

    def make_ema_teacher(self):
        self.ema = EMA(self, decay=1)

    def move_codebook_to_gpu(self):
        # Move codebook to GPU
        device = next(self.encoder.parameters()).device
        self.codebooks = {
            i:self.codebooks[i].to(device) for i in range(self.n_codebooks)
        }
        self.codebook_cnts = {
            i:self.codebook_cnts[i].to(device) for i in range(self.n_codebooks)
        }