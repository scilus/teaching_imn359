# Compute iFWT
import numpy as np
import matplotlib.pyplot as plt
from fonctions.general import upsampling
from fonctions.cconv import cconv 

def ifwt(fw, h, g, visu=False):
    n = fw.size
    Jmax = int(np.log2(n) - 1)
    Jmin = 0
    f1 = fw.copy()
    
    if visu: 
        fig, axs = plt.subplots(4, 1)

    for j in range(Jmin, Jmax+1):
        Coarse = f1[:2**j].copy()
        Detail = f1[2**j:2**(j+1)].copy()
        f1[0:2**(j+1)] = cconv(upsampling(Coarse), h, 0) + cconv(upsampling(Detail), g, 0)

        j1 = Jmax - j
        if visu and j1 < 4:
            axs[j1].plot(f1[:2**(j+1)], '.-')
            axs[j1].set_title('Partial Reconstruction, j = ' + str(j))
    plt.show()
    return f1
