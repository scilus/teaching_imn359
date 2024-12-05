import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from dct import dct
from idct import idct
from dct2 import dct2
from idct2 import idct2

from scipy.fft import fft, fftshift, ifftshift, ifft

# DCT locale
######################################################
# Approximation avec des cosinus locaux
# 
# Ameliore l'approximation globale de la DCT
# On decoupe le signal en segments locaux  dans lesquels on
# fait la DCT
# JPEG est basée sur cette base
# ####################################################

x = loadmat('bird.mat')['x']
n = x.shape[0]
fc = dct(x)

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
axs[1].plot(np.abs(fc_a))
plt.show()

# Lets keep 4 lowest frequency coefficients in each bin (4* n/w in total)
# iDCT locale
f1 = idct(fc)
fm_local = fc_a
fc_a_4 = np.zeros((n, 1))
fm_a = np.zeros((n, 1))

s = 4
for i in range(int(n/w)):
    seli = np.array([i * w, i * w + w])
    sela = np.array([i * w, i * w + s])
    fc_a_4[sela[0]:sela[1]] = fm_local[sela[0]:sela[1]]
    fm_local[seli[0]:seli[1]] = idct(fm_local[seli[0]:seli[1]])
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
plt.show()
