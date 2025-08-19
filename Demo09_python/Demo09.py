import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from dct import dct
from idct import idct

from scipy.fft import fft, fftshift, ifft

# DCT locale
######################################################
# Approximation avec des cosinus locaux
# 
# Ameliore l'approximation globale de la DCT
# On decoupe le signal en segments locaux  dans lesquels on
# fait la DCT
# Exemple avec un signal sonore
# ####################################################

x = np.load('bird.npy')
n = x.shape[0]
fc = dct(x)

# On peut écouter l'enregistrement audio
import sounddevice as sd
sd.play(x, n)

w = 16 # taille du segment

fc_a = np.zeros((n, 1))

# DCT locale
for i in range(int(n/w)):
    seli = np.array([i * w, i * w + w])
    fc_a[seli[0]:seli[1]] = dct(x[seli[0]:seli[1]])

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.abs(fc))
axs[0].set_title('DCT')
axs[1].plot(np.abs(fc_a))
axs[1].set_title('DCT locale')
fig.tight_layout()
plt.show()

# Reconstruction du signal avec full DCT
f1 = idct(fc)

# iDCT locale
fm_local = np.zeros((n, 1))
fc_a_4 = np.zeros((n, 1))
fm_a = np.zeros((n, 1))

# On garde les 4 plus basses fréquences de chaque segment (4*n/w au total)
s = 4
for i in range(int(n/w)):
    seli = np.array([i * w, i * w + w])
    sela = np.array([i * w, i * w + s])
    fc_a_4[sela[0]:sela[1]] = fc_a[sela[0]:sela[1]]
    fm_local[seli[0]:seli[1]] = idct(fc_a[seli[0]:seli[1]]) # on fait la iDCT locale non-tronquée aussi
    fm_a[seli[0]:seli[1]] = idct(fc_a_4[seli[0]:seli[1]])

f = x
coeff = 256

fc_2 = np.zeros((n, 1))
fc_2[0:coeff] = fc[0:coeff]
f1_2 = idct(fc_2)

ff = fft(x, axis=0)
ffamp = np.abs(fftshift(ff))
ffs = fftshift(ff)

ffs_2 =  np.zeros((n, 1))
ffs_2[int(n/2 - coeff):int(n/2 + coeff)] = ffs[int(n/2 - coeff):int(n/2 + coeff)]

xx = np.abs(ifft(fftshift(ffs_2), axis=0))

print('Error |f-f1|/|f| using a full Discrete Cosine basis = ', np.linalg.norm(f-f1)/np.linalg.norm(f))
print('Error |f-f1|/|f| using a full local Discrete Cosine basis = ', np.linalg.norm(f-fm_local)/np.linalg.norm(f))
print('Error |f-f1|/|f| using  a linear' + str(coeff) + ' coeffs local DCT approx = ', np.linalg.norm(f-fm_a)/np.linalg.norm(f))
print('Error |f-f1|/|f| using  a linear' + str(coeff) + ' coeffs discrete DCT approx = ', np.linalg.norm(f-f1_2)/np.linalg.norm(f))
print('Error |f-f1|/|f| using  a linear' + str(coeff) + ' coeffs Fourier approx = ', np.linalg.norm(f-xx)/np.linalg.norm(f))

fig, axs = plt.subplots(5, 1)
axs[0].plot(f)
axs[0].set_title('Full DCT, error = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
axs[1].plot(f1)
axs[1].set_title('Full locDCT, error = ' + str(np.linalg.norm(f-fm_local)/np.linalg.norm(f)))
axs[2].plot(fm_a)
axs[2].set_title('256 coeffs of localDCT (4 per bin), error = ' + str(np.linalg.norm(f-fm_a)/np.linalg.norm(f)))
axs[3].plot(f1_2)
axs[3].set_title('256 coeffs of DCT, error = ' + str(np.linalg.norm(f-f1_2)/np.linalg.norm(f)))
axs[4].plot(xx)
axs[4].set_title('256 coeffs of FFT, error = ' + str(np.linalg.norm(f-xx)/np.linalg.norm(f)))
fig.tight_layout()
plt.show()
