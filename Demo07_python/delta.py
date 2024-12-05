import numpy as np

def delta(t, t0):
    d = np.zeros(len(t))

    j = 1

    for i in range(0, len(t)):
        if j > len(t0) - 1:
            continue
        else:
            if t[i] == t0[j]:
                d[i] = 1.0
                j += 1
            else:
                d[i] = 0.0

    return d
