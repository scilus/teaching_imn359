import numpy as np

def snr(x, y):
    """
    snr - signal to noise ratio

    Parameters
    ----------
    x : fthe original clean signal (reference).
    y : the denoised signal.
    Returns
    -------
    snr : signal to noise ratio
    """

    snr = 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(x - y))

    return snr
