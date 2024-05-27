import numpy as np
from scipy.signal import welch

def cal_psd(emg, sampling_rate: float=800.0):
    """Calculates the psd for a given emg signal.

    Args:
        emg (torch.Tensor): Given emg signal
        sampling_rate (float, optional): The sampling rate. Defaults to 800.0.

    Returns:
        f : ndarray
            Array of sample frequencies.
        Pxx : ndarray
            Power spectral density or power spectrum of x.
    """
    f, pxx_den = welch(emg, sampling_rate)
    
    return f, pxx_den