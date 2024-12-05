# Compute Haar 1D
import numpy as np
import matplotlib.pyplot as plt


def computeHaar1D(signal, n):
    Jmax = np.log2(n) - 1
    Jmin = 0
    fw = signal.copy()
    fig, axs = plt.subplots(5, 1)
    axs[0].plot(signal)
    axs[0].set_title('Signal')

    for j in range(int(Jmax), Jmin-1, -1):
        A = fw[0:2**(int(j)+1)]
        Coarse = (A[::2] + A[1::2]) / np.sqrt(2)
        Detail = (A[::2] - A[1::2]) / np.sqrt(2)
        A = np.concatenate((Coarse, Detail))
        fw[0: 2**(j+1)] = A
        j1 = int(Jmax) - j
        if j > 4:
            axs[j1 + 1].plot(Coarse)
            axs[j1 + 1].set_title('Coarse, j = ' + str(j))
    plt.show()
    return fw