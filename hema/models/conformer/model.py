import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from typing import Callable, Dict, List, Optional, Tuple
import math

from utils import index_put

@dataclass
class TransformerEncoderConfig():
    encoder_layers: int = field(default=12)
    encoder_embed_dim: int = field(default=768)
    encoder_ffn_embed_dim: int = field(default=3072)
    encoder_attn_heads: int = field(default=8)
    
    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    
    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="espnet",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="rope",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=True, metadata={"help": "If fp16 is being used"})

    # dropouts
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )
    
    # Position embedding
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})

def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    def gelu_accurate(x):
        if not hasattr(gelu_accurate, "_a"):
            gelu_accurate._a = math.sqrt(2 / math.pi)
        return (
            0.5
            * x
            * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
        )

    def gelu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x.float()).type_as(x)

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        def relu_squared(x: torch.Tensor):
            return F.relu(x).pow(2)
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation)) 

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        """Rotary positional embedding
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            precision: precision to use for numerical values
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_len=None):
        """
        Args:
            x: Input x with T X B X C
            seq_len: Sequence length of input x
        """
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached

class RelPositionalEncoding(nn.Module):
    """Relative positional encoding module (new implementation).

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(self, max_len, d_model):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x : Input tensor T X B X C.
        Returns:
            torch.Tensor: Encoded tensor T X B X C.

        """
        x = x.transpose(0, 1)  # Change TBC to BTC
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        pos_emb = pos_emb.transpose(0, 1)  # change to TBC
        return pos_emb
 
class ESPNETMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        dropout: Dropout rate.
    """

    def __init__(self, n_feat, n_head, dropout):
        """Construct an MultiHeadedAttention object."""
        super(ESPNETMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward_qkv(self, query, key, value, **kwargs):
        """Transform query, key and value.
        Args:
            query: Query tensor  B X T1 X C
            key: Key tensor B X T2 X C
            value: Value tensor  B X T2 X C
        Returns:
            torch.Tensor: Transformed query tensor  B X n_head X T1 X d_k
            torch.Tensor: Transformed key tensor B X n_head X T2 X d_k
            torch.Tensor: Transformed value tensor  B X n_head X T2 X d_k
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value: Transformed value B X n_head X T2 X d_k.
            scores: Attention score  B X n_head X T1 X T2
            mask: Mask  T2 X B
        Returns:
            torch.Tensor: Transformed value  B X T1 X d_model
                weighted by the attention score  B X T1 X T2
        """
        n_batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2).to(bool),
                float("-inf"),  # (batch, head, time1, time2)
            )
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor T X B X C
            key (torch.Tensor): Key tensor T X B X C
            value (torch.Tensor): Value tensor T X B X C
            mask (torch.Tensor): Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None

class RelPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        dropout: Dropout rate.
        zero_triu: Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_feat, n_head, dropout, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head, dropout)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x: Input tensor B X n_head X T X 2T-1
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            pos_emb: Positional embedding tensor B X 2T-1 X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X C.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        pos_emb = pos_emb.transpose(0, 1)
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None
    
class RotaryPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    def __init__(
        self,
        n_feat,
        n_head,
        dropout,
        precision,
        rotary_emd_base=10000,
    ):
        """Construct an RotaryPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head, dropout)
        precision = torch.float
        self.rotary_ndims = self.d_k  # also try self.d_k//2
        if precision == "fp16":
            precision = torch.half

        self.rotary_emb = RotaryPositionalEmbedding(
            self.rotary_ndims, base=rotary_emd_base, precision=precision
        )

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute rotary position attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        Notes:
            Assumes self attn
        """

        T, B, C = value.size()
        query = query.view(T, B, self.h, self.d_k)
        key = key.view(T, B, self.h, self.d_k)
        value = value.view(T, B, self.h, self.d_k)
        cos, sin = self.rotary_emb(value, seq_len=T)
        query, key = apply_rotary_pos_emb(
            query, key, cos, sin, offset=0
        )  # offset is based on layer_past

        query = query.view(T, B, self.h * self.d_k)
        key = key.view(T, B, self.h * self.d_k)
        value = value.view(T, B, self.h * self.d_k)

        # TBD to BTD
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None

class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
        self,
        embed_dim,
        channels,
        depthwise_kernel_size,
        dropout,
        activation_fn="swish",
        bias=False,
        export=False,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
            export: If layernorm should be exported to jit
        """
        super(ConvolutionModule, self).__init__()
        assert (
            depthwise_kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = nn.LayerNorm(embed_dim, export=export)
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        x = self.layer_norm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)

class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(
        self,
        input_feat,
        hidden_units,
        dropout1,
        dropout2,
        activation_fn="swish",
        bias=True,
    ):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout1: dropout value for layer1
            dropout2: dropout value for layer2
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """

        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_feat)
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)

class ConformerEncoderLayer(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        use_fp16,
        depthwise_conv_kernel_size=31,
        activation_fn="swish",
        attn_type=None,
        pos_enc_type="abs",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation from ESPNET vs fairseq
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        self.pos_enc_type = pos_enc_type
        super(ConformerEncoderLayer, self).__init__()

        self.ffn1 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )

        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos":
                self.self_attn = RelPositionMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            elif self.pos_enc_type == "rope":
                self.self_attn = RotaryPositionMultiHeadedAttention(
                    embed_dim, attention_heads, dropout=dropout, precision=use_fp16
                )
            elif self.pos_enc_type == "abs":
                self.self_attn = ESPNETMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            else:
                raise Exception(f"Unsupported attention type {self.pos_enc_type}")

        self.conv_module = ConvolutionModule(
            embed_dim=embed_dim,
            channels=embed_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
        )

        self.ffn2 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation_fn=activation_fn,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        position_emb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.pos_enc_type == "rel_pos":
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)

        layer_result = x

        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, (attn, layer_result)
    
class ConformerEncoder(torch.nn.Module):
    def build_encoder_layer(self, args: TransformerEncoderConfig):
        layer = ConformerEncoderLayer(
            embed_dim=self.embedding_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attn_heads,
            dropout=args.dropout,
            depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
            activation_fn="swish",
            attn_type=args.attn_type,
            pos_enc_type=args.pos_enc_type,
            use_fp16=args.fp16,  # only used for rope
        )
        return layer
    
    def __init__(self, args: TransformerEncoderConfig):
        super().__init__(args)
        self.args = args
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.pos_enc_type = args.pos_enc_type
        max_source_positions = args.max_positions

        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                max_source_positions, self.embedding_dim
            )
        elif self.pos_enc_type == "rope":
            self.embed_positions = None
        else:
            raise Exception("Unsupported positional encoding type")

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

    def forward(self, x, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # B X T X C here
        position_emb = None
        if self.pos_enc_type == "rel_pos":
            position_emb = self.embed_positions(x)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    position_emb=position_emb,
                )
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results
        