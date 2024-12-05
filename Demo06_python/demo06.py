import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from scipy.fft import fft2, fftshift, ifftshift, ifft2

from snr import snr

# Demo06
# Lena
lena = loadmat("lena.mat")['lena']

plt.imshow(lena)
plt.savefig('lena.jpg')
plt.show()

plt.clf()
plt.imshow(np.abs(fft2(lena)))
plt.savefig('lena_abs_fft2.jpg')
plt.show()
# on ne voit rien. Pourquoi? Zoomer dans les valeurs. Visualiser la matrice.
# La difference entre le min et max de la matrice est trop importante.
Lena = fft2(lena)
Lena_abs = np.abs(Lena)
Lena_abs.min()
Lena_abs.max()
Lena_abs.mean()

plt.clf()
# truc de visu: on prend le log des valeurs pour ramener sur une plage de valeur
# plus petite
plt.imshow(np.log(np.abs(fft2(lena))))
plt.savefig('lena_log_abs_fft2.jpg')
plt.show()

# on voit que les hautes frequences sont dans les 4 coins de l'image
# fftshift pour les ramener centré comme dans mes diapos
plt.clf()
plt.imshow(fftshift(np.log(Lena_abs)))
plt.savefig('lena_fftshift_log_abs_fft2.jpg')
plt.show()


## Mandrill
M = loadmat("mandrill.mat")['M']
# Compute and display the Fourier transform (display over a log scale).
# The function fftshift is useful to put the 0 low frequency in the middle.
# After fftshift, the zero frequency is located at position (n/2+1,n/2+1).

Mf = fft2(M)
Lf = fftshift(np.log(np.abs(Mf)))

fig, axs = plt.subplots(1,2)
axs[0].imshow(M, cmap='gray')
axs[1].imshow(Lf, cmap='gray')
plt.savefig('mandrill.jpg')
plt.show()

n = 512 # size of mandrill
m = n / 2 # Number of coefficients in X and Y
F1 = np.zeros((n, n), dtype=complex)


# on prend le centre du spectre du mandrill et le reste est rempli de 0's 
F = fftshift(fft2(M))
sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int) + 1
F1[sel[0] : sel[1], sel[0] : sel[1]] = F[sel[0] : sel[1], sel[0] : sel[1]]

fig, axs = plt.subplots(1,2)
axs[0].imshow(np.log(np.abs(F1)), cmap='gray')
axs[0].set_title('Cropped spectrum : ' + str(m*m/(n*n)*100) + '% of coefficients')
axs[1].imshow(Lf, cmap='gray')
axs[1].set_title('Original spectrum')
plt.savefig('mandrill_spectre.jpg')
plt.show()

M1 = np.real(ifft2(ifftshift(F1)))

fig, axs = plt.subplots(1,2)
axs[0].imshow(M, cmap='gray')
axs[0].set_title('Original image')
axs[1].imshow(M1, cmap='gray')
axs[1].set_title('Approx, SNR=' + str(np.round(snr(M,M1), 4)) + 'dB')
plt.savefig('mandrill_compressed.jpg')
plt.show()


# Less coefficients
m = n / 16 # 4 coefficients in X and Y
F1 = np.zeros((n, n), dtype=complex)

F = fftshift(fft2(M))
sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int) + 1
F1[sel[0] : sel[1], sel[0] : sel[1]] = F[sel[0] : sel[1], sel[0] : sel[1]]

fig, axs = plt.subplots(1,2)
axs[0].imshow(np.log(np.abs(F1)), cmap='gray')
axs[0].set_title('Cropped spectrum : ' + str(m*m/(n*n)*100) + '% of coefficients')
axs[1].imshow(Lf, cmap='gray')
axs[1].set_title('Original spectrum')
plt.savefig('mandrill_spectre2.jpg')
plt.show()

M1 = np.real(ifft2(ifftshift(F1)))

fig, axs = plt.subplots(1,2)
axs[0].imshow(M, cmap='gray')
axs[0].set_title('Original image')
axs[1].imshow(M1, cmap='gray')
axs[1].set_title('Approx, SNR=' + str(np.round(snr(M,M1), 4)) + 'dB')
plt.savefig('mandrill_compressed2.jpg')
plt.show()


# Try with cameraman
M = loadmat("cameraman.mat")['cameraman']
n = 512
m = n/4 # 2 coefficients in X and Y
F1 = np.zeros((n, n), dtype=complex)

Mf = fft2(M)
Lf = fftshift(np.log(np.abs(Mf)))
F = fftshift(fft2(M))
sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int) + 1
F1[sel[0] : sel[1], sel[0] : sel[1]] = F[sel[0] : sel[1], sel[0] : sel[1]]

fig, axs = plt.subplots(1,2)
axs[0].imshow(np.log(np.abs(F1)), cmap='gray')
axs[0].set_title('Cropped spectrum : ' + str(m*m/(n*n)*100) + '% of coefficients')
axs[1].imshow(Lf, cmap='gray')
axs[1].set_title('Original spectrum')
plt.savefig('cameraman_spectre.jpg')
plt.show()

M1 = np.real(ifft2(ifftshift(F1)))

fig, axs = plt.subplots(1,2)
axs[0].imshow(M, cmap='gray')
axs[0].set_title('Original image')
axs[1].imshow(M1, cmap='gray')
axs[1].set_title('Approx, SNR=' + str(np.round(snr(M,M1), 4)) + 'dB')
plt.savefig('cameraman_compressed.jpg')
plt.show()


# Add noise
sigma = 0.1
Mn = M + np.random.uniform(0,sigma,512)
plt.imshow(Mn, cmap="gray")
plt.savefig('cameraman_noisy.jpg')
plt.show()

Mnf = fft2(Mn)
Mnf_shift = fftshift(Mnf)
Lnf = fftshift(np.log(np.abs(Mnf)))

plt.clf()
plt.imshow(Lnf, cmap="gray")
plt.savefig('cameraman_spectre.jpg')
plt.show()

m = n/2
F1 = np.zeros((n, n), dtype=complex)
sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int) + 1
F1[sel[0] : sel[1], sel[0] : sel[1]] = Mnf_shift[sel[0] : sel[1], sel[0] : sel[1]]

fig, axs = plt.subplots(1,3)
axs[0].imshow(np.real(ifft2(fftshift(F1))), cmap='gray')
axs[0].set_title('Cropped Fourier : ' + str(m*m/(n*n)*100) + '% of coefficients')
axs[1].imshow(np.real(ifft2(Mnf)), cmap='gray')
axs[1].set_title('Full Fourier noisy')
axs[2].imshow(M, cmap='gray')
axs[2].set_title('Original')
plt.savefig('cameraman_compressed.jpg')
plt.show()
