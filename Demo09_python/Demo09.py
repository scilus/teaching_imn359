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
x_dct = dct(x)

# On peut écouter l'enregistrement audio
try:
    import sounddevice as sd
    sd.play(x, n)
except OSError:
    # possible que ça ne fonctionne pas s'il manque des librairies sur votre ordi
    print('Erreur avec l\'audio. Tant pis, on va se contenter de regarder des graphiques.')

# DCT locale
w = 16 # taille du segment
x_dct_local = np.zeros((n, 1))
for i in range(int(n/w)):
    seli = np.array([i * w, i * w + w])
    x_dct_local[seli[0]:seli[1]] = dct(x[seli[0]:seli[1]])

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.abs(x_dct))
axs[0].set_title('DCT')
axs[1].plot(np.abs(x_dct_local))
axs[1].set_title('DCT locale')
fig.tight_layout()
plt.show()

# Reconstruction du signal avec full DCT
x_idct = idct(x_dct)

# iDCT locale
x_idct_local = np.zeros((n, 1))
# DCT locale tronquée à s=4
x_dct_local_trunc4 = np.zeros((n, 1))
# iDCT de la DCT locale tronquée à s=4
x_idct_local_trunc4 = np.zeros((n, 1))

# On garde les 4 plus basses fréquences de chaque segment (4*n/w au total)
s = 4
for i in range(int(n/w)):
    seli = np.array([i * w, i * w + w])
    sela = np.array([i * w, i * w + s])
    x_dct_local_trunc4[sela[0]:sela[1]] = x_dct_local[sela[0]:sela[1]]
    x_idct_local[seli[0]:seli[1]] = idct(x_dct_local[seli[0]:seli[1]])
    x_idct_local_trunc4[seli[0]:seli[1]] = idct(x_dct_local_trunc4[seli[0]:seli[1]])

# Signal en entrée
f = x

# Nombre de coefficients des full DCT et FFT tronquées
coeff = 256

x_dct_trunc256 = np.zeros((n, 1))
x_dct_trunc256[0:coeff] = x_dct[0:coeff]
x_idct_trunc256 = idct(x_dct_trunc256)

x_fft = fft(x, axis=0)
x_fft_shift = fftshift(x_fft)

x_fft_shift_trunc256 =  np.zeros((n, 1), dtype=complex)
x_fft_shift_trunc256[int(n/2 - coeff/2):int(n/2 + coeff/2)] = x_fft_shift[int(n/2 - coeff/2):int(n/2 + coeff/2)]

x_ifft_shift_trunc256 = np.abs(ifft(fftshift(x_fft_shift_trunc256), axis=0))

print('Error |f-f1|/|f| using a full Discrete Cosine basis = ', np.linalg.norm(f-x_idct)/np.linalg.norm(f))
print('Error |f-f1|/|f| using a full local Discrete Cosine basis = ', np.linalg.norm(f-x_idct_local)/np.linalg.norm(f))
print('Error |f-f1|/|f| using  a linear' + str(coeff) + ' coeffs local DCT approx = ', np.linalg.norm(f-x_idct_local_trunc4)/np.linalg.norm(f))
print('Error |f-f1|/|f| using  a linear' + str(coeff) + ' coeffs discrete DCT approx = ', np.linalg.norm(f-x_idct_trunc256)/np.linalg.norm(f))
print('Error |f-f1|/|f| using  a linear' + str(coeff) + ' coeffs Fourier approx = ', np.linalg.norm(f-x_ifft_shift_trunc256)/np.linalg.norm(f))

fig, axs = plt.subplots(5, 1)
axs[0].plot(f)
axs[0].set_title('Full DCT, error = ' + str(np.linalg.norm(f-x_idct)/np.linalg.norm(f)))
axs[1].plot(x_idct)
axs[1].set_title('Full locDCT, error = ' + str(np.linalg.norm(f-x_idct_local)/np.linalg.norm(f)))
axs[2].plot(x_idct_local_trunc4)
axs[2].set_title('256 coeffs of localDCT (4 per bin), error = ' + str(np.linalg.norm(f-x_idct_local_trunc4)/np.linalg.norm(f)))
axs[3].plot(x_idct_trunc256)
axs[3].set_title('256 coeffs of DCT, error = ' + str(np.linalg.norm(f-x_idct_trunc256)/np.linalg.norm(f)))
axs[4].plot(x_ifft_shift_trunc256)
axs[4].set_title('256 coeffs of FFT, error = ' + str(np.linalg.norm(f-x_ifft_shift_trunc256)/np.linalg.norm(f)))
fig.tight_layout()
plt.show()
