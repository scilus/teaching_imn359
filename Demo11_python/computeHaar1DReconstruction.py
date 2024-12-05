# Compute Haar 1D
import numpy as np
import matplotlib.pyplot as plt


def computeHaar1DReconstruction(fw, n):
    Jmax = int(np.log2(n) - 1)
    Jmin = 0
    f1 = fw.copy()
    fig, axs = plt.subplots(4, 1)

    for j in range(Jmin, Jmax+1):
        Coarse = f1[:2**j].copy()
        Detail = f1[2**j:2**(j+1)].copy()
        f1[0:2**(j+1):2] = (Coarse + Detail) / np.sqrt(2)
        f1[1:2**(j+1):2] = (Coarse - Detail) / np.sqrt(2)

        j1 = Jmax - j
        if j1 < 4:
            axs[j1].plot(f1[:2**(j+1)], '.-')
            axs[j1].set_title('Partial Reconstruction, j = ' + str(j))
    plt.show()
    return f1
