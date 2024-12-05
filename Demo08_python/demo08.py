import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from dct import dct
from idct import idct
from dct2 import dct2
from idct2 import idct2
from snr import snr

from scipy.fft import fft2, fftshift, ifftshift, ifft2, rfft

from perform_thresholding import perform_thresholding

t1 = np.arange(0, 1+1/1000, 1 / 1000)  # ts = 0.001 seconde, fs = 1000
f1 = 8
f2 = 3

s1 = np.cos(2 * np.pi * f1 * t1) + np.cos(2 * np.pi * f2 * t1)
dct_s1 = dct(s1) # rfft(s1)

fig, axs = plt.subplots(1, 2)

axs[0].plot(s1)
axs[1].plot(np.squeeze(dct_s1))
plt.show()

# LA DCT
t = np.arange(1, 101)
x = t + 50 * np.cos(t * 2 * np.pi / 40)
X = dct(x)

fig, axs = plt.subplots(1, 2)

axs[0].plot(x)
axs[1].plot(np.squeeze(X))
plt.show()


# Piece-regular
x0 = loadmat("piece-regular.mat")['x0']
X0 = np.abs(fft2(x0))
X0_shift = np.abs(fftshift(fft2(x0)))
fig, axs = plt.subplots(3, 1)
axs[0].plot(X0)
axs[0].set_title('fft of piece regular')
axs[1].plot(X0_shift)
axs[1].set_title('fft shifted of piece regular')
axs[2].plot(dct(x0))
axs[2].set_title('dct of piece regular')
plt.show()

n = 500
length = x0.shape[0]
xf = fftshift(fft2(x0))
xc = dct(x0)
x_compressed = np.zeros((x0.shape[0], 1), dtype=complex)
x_compressed[int(length/2) - int(n/2):int(length / 2) + int(n/2)]= xf[int(length/2) - int(n/2):int(length / 2) + int(n/2)]
x_compressed_fft = np.abs(ifft2(ifftshift(x_compressed)))

x_dct = np.zeros((length, 1))
x_dct[0:n] = xc[0:n].reshape((n, 1))

fig, axs = plt.subplots(3, 1)
axs[0].plot(x0)
axs[0].set_title('Piece-regular')
axs[1].plot(x_compressed_fft)
axs[1].set_title(str(n) + ' Fourier coeffs: erreur: ' +
                 str(np.linalg.norm(x_compressed_fft - x0)/np.linalg.norm(x0)*100)+ '%')
axs[2].plot(idct(x_dct))
axs[2].set_title(str(n) + ' DCT coeffs: erreur: ' +
                 str(np.linalg.norm(idct(x_dct) - x0)/np.linalg.norm(x0)*100)+ '%')
plt.show()

# In 2D
# Fourier classique
M = loadmat('lena.mat')['M']
n = M.shape[0]
m = int(n/8)
F = fftshift(fft2(M))
F1 = np.zeros_like(F, dtype=complex)
sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int)
F1[sel[0] : sel[1], sel[0] : sel[1]] = F[sel[0] : sel[1], sel[0] : sel[1]]
f1 = np.real(ifft2(fftshift(F1)))

fig, axs = plt.subplots(2, 1)
axs[0].imshow(M)
axs[0].set_title('Image')
axs[1].imshow(f1)
axs[1].set_title('Approx, SNR= ' + str(snr(M, f1)) + ' dB')
plt.show()

# DCT 2D
Mc = dct2(M)
fig, axs = plt.subplots(2, 1)
axs[0].imshow(np.log(np.abs(Mc)))
axs[1].imshow(np.log(np.abs(F)))
plt.show()

fig, axs = plt.subplots(2, 1)
f2 = np.zeros((n,n))
f2[0:m, 0:m] = Mc[0:m, 0:m]
f2 = idct2(f2)

fig, axs = plt.subplots(2, 1)
axs[0].imshow(f1)
axs[0].set_title('Fourier Approx' + str(m**2) + ' coeffs, SNR= ' + str(snr(M, f1)) + ' dB')
axs[1].imshow(f2)
axs[1].set_title('DCT Approx' + str(m**2) + ' coeffs, SNR= ' + str(snr(M, f2)) + ' dB')
plt.show()


############################################################
# DCT locale
############################################################
# Approximation avec des cosinus locaux
# 
# Ameliore l'approximation globale de la DCT
# On decoupe le signal en segments locaux  dans lesquels on
# fait la DCT JPEG est basée sur cette approche
# ###########################################################

x0 = loadmat('piece-regular.mat')['x0']
n = x0.shape[0]
# n = 1024

coeff = 128
dct_full = dct(x0)
dct_linear = np.zeros((n, 1))
dct_linear[0:coeff] = dct_full[0:coeff]
f_dct_full = idct(dct_full)  # la full DCT reconstruction sans couper des coefficients 
f_dct_full_lin = idct(dct_linear)  # approximation linear du piece-regular prenant les 128 premiers coefficients 
f_dct_full_nonlin = idct(perform_thresholding(dct_full, coeff, 'largest')) # approximation non-linear prenant que les 128 plus grands coeffcients

# taille du segment
w = 32

# coefficients de la DCT locale 
locDCT_full = np.zeros((n, 1))

# DCT locale
for i in range(int(n/w)):
    # indices du segment i 
    seli_full = np.array([i * w, i * w + w])
    # DCT locale dans le segment i 
    locDCT_full[seli_full[0]:seli_full[1]] = dct(x0[seli_full[0]:seli_full[1]])
    
fig, axs = plt.subplots(3, 1)
axs[0].plot(x0)
axs[1].plot(np.abs(dct_full))
axs[2].plot(np.abs(locDCT_full))
plt.show()

# On a 32 segments de taille 32 (1024 / 32 = 32). Donc, si on garde 128 coefficients au total,
# on garde que s coefficients (128/32=4) par segment. 

# iDCT locale. On initialise tout à 0 
local_segment = np.zeros((n,1))
f_locDCT_full = np.zeros((n, 1))
f_locDCT_lin = np.zeros((n, 1))
f_locDCT_nonlin = np.zeros((n, 1))

s = int(coeff / (n / w))
for i in range(int(n/w)):
    # tous indices du segment i 
    seli_full = np.array([i * w, i * w + w])
    # que les s premiers indices du segment i
    seli_cut = np.array([i * w, i * w + s])

    # la full DCT local prends tous les coefficients s a chaque segment 
    f_locDCT_full[seli_full[0]:seli_full[1]] = idct(dct(x0[seli_full[0]:seli_full[1]]))

    # l'approximation lineaire garde que les s premiers coefficients
    # le reste des coefficients restent à zéro
    local_segment[seli_cut[0]:seli_cut[1]] = locDCT_full[seli_cut[0]:seli_cut[1]]
    f_locDCT_lin[seli_full[0]:seli_full[1]] = idct(local_segment[seli_full[0]:seli_full[1]])

    # l'approximation non-lineaire garde que les s plus grands coefficients dans chaque segment i
    # le reste des coefficients restent à 0
    f_locDCT_nonlin[seli_full[0]:seli_full[1]] = idct(perform_thresholding(locDCT_full[seli_full[0]:seli_full[1]], s, 'largest'))


f = x0
print('Erreur quadratique moyenne avec une full Discrete Cosine transform (DCT) = ',
      np.linalg.norm(f-f_dct_full)/np.linalg.norm(f))
print('Erreur quadratique moyenne avec une full locale Discrete Cosine transform (locDCT) = ',
      np.linalg.norm(f-f_locDCT_full)/np.linalg.norm(f))
print('Erreur quadratique moyenne avec ' + str(coeff) + ' coefficients lineair DCT coeffs = ',
      np.linalg.norm(f-f_dct_full_lin)/np.linalg.norm(f))
print('Erreur quadratique moyenne avec ' + str(coeff) + ' coefficients lineair locDCT coeffs = ',
      np.linalg.norm(f-f_locDCT_lin)/np.linalg.norm(f))
print('\n')
print('Erreur quadratique moyenne avec ' + str(coeff) + ' coefficients non-linear DCT coeffs = ',
      np.linalg.norm(f-f_dct_full_nonlin)/np.linalg.norm(f))
print('Erreur quadratique moyenne avec ' + str(coeff) + ' coefficients non-linear local DCT coeffs = ',
      np.linalg.norm(f-f_locDCT_nonlin)/np.linalg.norm(f))

fig, axs = plt.subplots(5, 1)
fig.tight_layout(pad=1.0)
axs[0].plot(f_locDCT_full)
axs[0].set_title('\n Reconstruction avec une full locDCT (tous les coefficients)')
axs[1].plot(f_dct_full_lin)
axs[1].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients lineaires de la full DCT. Erreur: '
                 + str(np.linalg.norm(f-f_dct_full_lin)/np.linalg.norm(f)))
axs[2].plot(f_dct_full_nonlin)
axs[2].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineaires de la full DCT. Erreur: '
                 + str( np.linalg.norm(f-f_dct_full_nonlin)/np.linalg.norm(f)))
axs[3].plot(f_locDCT_lin)
axs[3].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients lineares d\'une DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f-f_locDCT_lin)/np.linalg.norm(f)))
axs[4].plot(f_locDCT_nonlin)
axs[4].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineares d\'une DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f-f_locDCT_nonlin)/np.linalg.norm(f)))
plt.show()


# regardons de plus pres les coefficients coupés
local_lin = np.zeros((n,1))
local_nonlin = np.zeros((n,1))
for i in range(int(n/w)):
    # indices du segment i 
    seli_full = np.array([i * w, i * w + w])
    # DCT locale dans le segment i 
    locDCT_full[seli_full[0]:seli_full[1]] = dct(x0[seli_full[0]:seli_full[1]])

    # que les s premiers indices du segment i
    seli_cut = np.array([i * w, i * w + s])
    local_lin[seli_cut[0]:seli_cut[1]] = locDCT_full[seli_cut[0]:seli_cut[1]]
    
    # que les s plus grands coefficients du segment i
    local_nonlin[seli_full[0]:seli_full[1]] = perform_thresholding(locDCT_full[seli_full[0]:seli_full[1]], s, 'largest')
    

global_nonlin = perform_thresholding(locDCT_full, coeff, 'largest')

fig, axs = plt.subplots(6, 1)
axs[0].plot(np.abs(f))
axs[1].plot(np.abs(dct_full))
axs[2].plot(np.abs(locDCT_full))
axs[3].plot(np.abs(local_lin))
axs[4].plot(np.abs(local_nonlin))
axs[5].plot(np.abs(global_nonlin))
plt.show()


f_global_nonlin = np.zeros((n,1))
for i in range(int(n/w)):
    seli_full = np.array([i * w, i * w + w])
    f_global_nonlin[seli_full[0]:seli_full[1]] = idct(global_nonlin[seli_full[0]:seli_full[1]])


print('Erreur quadratique moyenne avec ' + str(coeff)
      + ' coefficients non-lineares pris sur la globalité de la local DCT coeffs = ',
      np.linalg.norm(f-f_global_nonlin)/np.linalg.norm(f))


fig, axs = plt.subplots(6, 1)
fig.tight_layout(pad=1.0)
axs[0].plot(f_locDCT_full)
axs[0].set_title('\n Reconstruction avec une full locDCT (tous les coefficients)')
axs[1].plot(f_dct_full_lin)
axs[1].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients lineaires de la full DCT. Erreur: '
                 + str(np.linalg.norm(f-f_dct_full)/np.linalg.norm(f)))
axs[2].plot(f_dct_full_nonlin)
axs[2].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineaires de la full DCT. Erreur: '
                 + str( np.linalg.norm(f-f_dct_full_nonlin)/np.linalg.norm(f)))
axs[3].plot(f_locDCT_lin)
axs[3].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients lineares d\'une DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f-f_locDCT_lin)/np.linalg.norm(f)))
axs[4].plot(f_locDCT_nonlin)
axs[4].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineares d\'une DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f-f_locDCT_nonlin)/np.linalg.norm(f)))
axs[5].plot(f_global_nonlin)
axs[5].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineares sur la globalite de la DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f-f_global_nonlin)/np.linalg.norm(f)))

plt.show()

# tester avec d'autres valeurs de coeff et w
# coeff = 256 est impressionnant!
