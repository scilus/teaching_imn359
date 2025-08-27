import numpy as np
# general.py TODO: try to not make use of transform.resize

def crop(M, n=None):
    """ crop - crop an image to reduce its size
    Only crops square black and white images for now.
    """
    # Check that image is square and black and white (no channels)
    assert M.shape[0] == M.shape[1]
    assert len(M.shape) == 2

    n0 = M.shape[0]

    if n is None:
        n = (n0/2)
    # Start and end of selection
    start_ind = int(np.floor((n0 - n) / 2))
    end_ind = int(-np.ceil((n0 - n) / 2))

    return M[start_ind:end_ind, start_ind:end_ind]


def circshift(x, p):
    """
        Circular shift of an array.
    """
    y = x.copy()
    y = np.concatenate((y[p[0]::, :], y[:p[0]:, :]), axis=0)
    if x.shape[1] > 0 and len(p) > 1:
        y = np.concatenate((y[:, p[0]::], y[:, :p[0]:]), axis=1)
    return y

def circshift1d(x, k):
    """
        Circularly shift a 1D vector
    """
    return np.roll(x, -k, axis=0)

def clamp(x, a=[], b=[]):
    """
     clamp - clamp a value

       y = clamp(x,a,b);

     Default is [a,b]=[0,1].

       Copyright (c) 2004 Gabriel Peyre
    """

    if a == []:
        a = 0.0
    if b == []:
        b = 1.0
    return np.minimum(np.maximum(x, a), b)

def rescale(f,a=0,b=1):
    """
        Rescale linearly the dynamic of a vector to fit within a range [a,b]
    """
    v = f.max() - f.min()
    g = (f - f.min()).copy()
    if v > 0:
        g = g / v
    return a + g*(b-a)


def reverse(x):
    """
        Reverse a vector.
    """
    return x[::-1]


def upsampling_2d(x, d):
    """
        up-sampling along dimension d by factor p=2
    """
    p = 2
    s = x.shape
    if d == 0:
        y = np.zeros((p * s[0], s[1]))
        y[::p, :] = x
    elif d == 1:
        y = np.zeros((s[0], p * s[1]))
        y[:, ::p] = x
    else:
        raise Exception('Not implemented')
    return y


def subsampling_2d(x, d):
    # subsampling along dimension d by factor p=2
    p = 2
    if d == 0:
        y = x[::p, :]
    elif d == 1:
        y = x[:, ::p]
    else:
        raise Exception('Not implemented')
    return y


def upsampling(x):
    """
        up-sampling along dimension d by factor p=2
    """
    p = 2
    s = x.shape
    y = np.zeros((p * s[0]))
    y[::p] = x

    return y


def subsampling(x):
    # subsampling along dimension d by factor p=2
    p = 2
    y = x[::p]

    return y
