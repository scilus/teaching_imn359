import matplotlib.pyplot as plt
import numpy as np

from fonctions.io import read_data
from fonctions.ondelette import computeHaar1D, computeHaar1DReconstruction
from fonctions.threshold import perform_thresholding

#### Demo des ondelettes de Haar 1D
############# 1 D Haar Wavelet DECOMPOSITION ################
# get signal
f = read_data('piece-regular_512.npy')
n = 512
plt.plot(f)
plt.title('Signal')
plt.show()

# Initialiser les coefficients avec le signal lui-même,
# Échelle 2^9 = 512 bins, résolution 2^{-9}
fw = f.copy()

# Échelle initiale j est l'échelle max
j = np.log2(n) - 1

# Les coefficients seront calculés itérativement jusqu'à
# ce qu'ils contiennent les ondelettes de Harr.
A = fw[0:2**(int(j)+1)]

# 256 x 1
# Calcul de la moyenne et des différences pour obtenir `coarse` et `details`.
# Normalisation par 1/sqrt(2) pour conserver l'orthogonalité.
Coarse = (A[::2] + A[1::2]) / 2
Detail = (A[::2] - A[1::2]) / 2

A = np.concatenate((Coarse, Detail))

fig, axs = plt.subplots(2, 1)
axs[0].plot(f, '.-')
axs[0].set_title('Signal')
axs[1].plot(Coarse, '.-')
axs[1].set_title('Coarse')
fig.tight_layout()
plt.show()

# On zoom dans une region de Coarse
s = 400
t = 40
fig, axs = plt.subplots(2, 1)
axs[0].plot(f, 'o-')
axs[0].set_xlim([s-t, s+t])
axs[0].set_ylim([0, 1])
axs[0].set_title('Signal (zoom)')
axs[1].plot(Coarse, 'o-')
axs[1].set_xlim([(s-t)/2, (s+t)/2])
axs[1].set_ylim([np.min(A), np.max(A)])
axs[1].set_title('Averages (zoom)')
fig.tight_layout()
plt.show()

# plot Details
fig, axs = plt.subplots(2, 1)
axs[0].plot(f, '.-')
axs[0].set_title('Signal')
axs[1].plot(Detail, '.-')
axs[1].set_title('Details')
fig.tight_layout()
plt.show()

# Signal et coarse et details
fig, axs = plt.subplots(2, 1)
axs[0].plot(f)
axs[0].set_title('Signal')
axs[1].plot(A)
axs[1].set_title('Transformed')
fig.tight_layout()
plt.show()

# plot `coarse` et `details`
fig, axs = plt.subplots(2, 1)
axs[0].plot(A[0:256])
axs[0].set_title('Coarse')
axs[1].plot(A[256:])
axs[1].set_title('Details')
fig.tight_layout()
plt.show()

#####################################################################
# Calcul des ondelettes de Haar pour tous les niveaux de la pyramide.
#####################################################################
fw = computeHaar1D(f, n)  # il y a un plt.show() dans la fonction

# On plot tous les coefficients
plt.plot(fw)
plt.title('Ondelettes 1D')
plt.show()

print('Energy of the signal       = ', np.linalg.norm(f) ** 2)
print('Energy of the coefficients = ', np.linalg.norm(fw) ** 2)

########################################################################
# Reconstruction 1D
#
# On initialise l'image pour reconstruire f1 en transformant les
# coefficients.
########################################################################

f1 = fw.copy()
j = 0 

# Exemple.
# Extraire les coefficients coarse et details
Coarse = f1[:2**j].copy()
Detail = f1[2**j:2**(j+1)].copy()

# Reconstruire le prochain niveau de la pyramide
f1[0:2**(j+1):2] = (Coarse + Detail) / np.sqrt(2)
f1[1:2**(j+1):2] = (Coarse - Detail) / np.sqrt(2)

# On utilise la fonction computeHaar1DReconstruction pour faire
# la reconstruction de maniere iterative.
f1 = computeHaar1DReconstruction(fw, n)  # plt.show() dans la fonction

# Original image versus Reconstructed image
print('Error |f-f1|/|f| = ', np.linalg.norm(f-f1)/np.linalg.norm(f))

fig, axs = plt.subplots(2, 1)
axs[0].plot(f)
axs[0].set_title('Original Signal')
axs[1].plot(f1)
axs[1].set_title('Reconstructed signal with full HAAR transform, error = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
fig.tight_layout()
plt.show()

##################
# Approximations #
##################
cut = 100
fw_a = np.zeros((fw.shape[0],))
fw_a[0:cut] = fw[0:cut]

fw_max = perform_thresholding(fw, cut, 'largest')

# Reconstruction avec les 100 premiers coefficients
f1_a = computeHaar1DReconstruction(fw_a, n)

# reconstruction avec les 100 plus gros coefficients
f1_max = computeHaar1DReconstruction(fw_max, n)

fig, axs = plt.subplots(3, 1)
axs[0].plot(f)
axs[0].set_title('Original Signal')
axs[1].plot(f1_a)
axs[1].set_title('Linear approx with ' + str(cut) + 'coefficients')
print('Error |f-f1_a|/|f| = ', np.linalg.norm(f-f1_a)/np.linalg.norm(f))
axs[2].plot(f1_max)
axs[2].set_title('Nonlinear approx with ' + str(cut) + 'coefficients')
print('Error |f-f1_max|/|f| = ', np.linalg.norm(f-f1_max)/np.linalg.norm(f))
fig.tight_layout()
plt.show()
