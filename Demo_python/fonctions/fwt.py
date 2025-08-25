# Compute FWT
import numpy as np
import matplotlib.pyplot as plt
from fonctions.general import subsampling
from fonctions.cconv import cconv

def fwt(signal, h, g, visu=False):
    n = signal.size
    Jmax = np.log2(n) - 1
    Jmin = 0
    fw = signal.copy()

    if visu :
        fig, axs = plt.subplots(5, 1)
        axs[0].plot(signal)
        axs[0].set_title('Signal')

    for j in range(int(Jmax), Jmin-1, -1):
        A = fw[0:2**(int(j)+1)]
        Coarse = subsampling(cconv(A, h, 0)) 
        Detail = subsampling(cconv(A, g, 0)) 
        A = np.concatenate((Coarse, Detail))
        fw[0: 2**(j+1)] = A
        j1 = int(Jmax) - j
        if visu and j > 4:
            axs[j1 + 1].plot(Coarse)
            axs[j1 + 1].set_title('Coarse, j = ' + str(j))
    plt.show()
    return fw
