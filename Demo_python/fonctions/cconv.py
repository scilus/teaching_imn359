import numpy as np
from fonctions.general import circshift1d


def cconv(x, h, d):
    """
        Circular convolution along dimension d.
        h should be small and with odd size
    """
    if d == 2:
        # apply to transposed matrix
        return np.transpose(cconv(np.transpose(x), h, 1))
    y = np.zeros(x.shape)
    p = len(h)
    pc = int(round( float((p - 1) / 2 )))
    for i in range(0, p):
        y = y + h[i] * circshift1d(x, i - pc)
    return y