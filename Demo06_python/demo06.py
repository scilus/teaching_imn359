"""
Demo06 -- Transformée de Fourier discrète 2D
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from scipy.fft import fft2, fftshift, ifftshift, ifft2

from snr import snr

# Demo06
# Lena
lena = np.load('lena.npy')

plt.imshow(lena)
plt.savefig('lena.jpg')
plt.show()
plt.clf()  # sert a "nettoyer" notre graphique avant d'en afficher un nouveau

# transformée de Fourier en 2D de lena
lena_TF = fft2(lena)

plt.imshow(np.abs(lena_TF))  # on affiche l'amplitude de la TF
plt.title("TF sans mise à l'échelle\n(Zoomez sur le coin supérieur gauche)")
plt.savefig('lena_abs_fft2.jpg')
plt.show()
plt.clf()

# On ne voit rien. Pourquoi? Zoomez dans les valeurs. Visualisez la matrice.
# La difference entre le min et max de la matrice est trop importante.
Lena_TF_abs = np.abs(lena_TF)
print("Amplitude minimum dans Fourier:", Lena_TF_abs.min())
print("Amplitude maximum dans Fourier:", Lena_TF_abs.max())
print("Amplitude moyenne dans Fourier:", Lena_TF_abs.mean())

# TRUC DE VISU: on prend le log des valeurs pour ramener sur
# une plage de valeur plus petite --> Uniquement pour la visualisation,
# NE JAMAIS TRAVAILLER SUR LE LOG DES AMPLITUDES DE LA TF!
plt.imshow(np.log(np.abs(lena_TF)))
plt.title("TF avec mise à l'échelle (on prend le log des amplitudes)")
plt.savefig('lena_log_abs_fft2.jpg')
plt.show()
plt.clf()

# on voit que les hautes frequences sont dans les 4 coins de l'image
# fftshift pour les ramener centré comme dans mes diapos
plt.imshow(fftshift(np.log(np.abs(lena_TF))))
plt.title("TF avec mise à l'échelle et recentrage des fréquences (fftshift)")
plt.savefig('lena_fftshift_log_abs_fft2.jpg')
plt.show()

################################
# Autre exemple avec le mandrill
################################
mandrill = np.load('mandrill.npy')

# Calculer et afficher la transformée de Fourier (sur une échelle
# logarithmique). La fonction fftshift permet de mettre la fréquence
# (0, 0) au centre de l'image. Après fftshift, la fréquence (0, 0) se
# trouve à la position (n/2, n/2).
mandrill_TF = fft2(mandrill)
mandrill_TF_pour_visu = fftshift(np.log(np.abs(mandrill_TF)))

fig, axs = plt.subplots(1,2)
axs[0].imshow(mandrill, cmap='gray')
axs[0].set_title("Image originale")
axs[1].imshow(mandrill_TF_pour_visu, cmap='gray')
axs[1].set_title("TF mise à l'échelle et shiftée")
plt.savefig('mandrill.jpg')
plt.show()

########################
# Exemple de compression
########################
n = mandrill.shape[0] # Dimension des côtés du mandrill
m = n / 2 # Nombre de coefficients à garder en X et Y
mandrill_TF_shift_masquee = np.zeros((n, n), dtype=complex)

# on prend le centre du spectre du mandrill et le reste est rempli de 0 
mandrill_TF_shift = fftshift(mandrill_TF)
sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int)
mandrill_TF_shift_masquee[sel[0]:sel[1], sel[0]:sel[1]] = mandrill_TF_shift[sel[0]:sel[1], sel[0]:sel[1]]

fig, axs = plt.subplots(1,2)
# On ajoute un petit epsilon (1e-8) pour ne pas avoir log(0)
axs[0].imshow(np.log(np.abs(mandrill_TF_shift_masquee) + 1e-8), cmap='gray')
axs[0].set_title('TF masquée : ' + str(m*m/(n*n)*100) + '% des coefficients')
axs[1].imshow(mandrill_TF_pour_visu, cmap='gray')
axs[1].set_title('TF originale')
plt.savefig('mandrill_spectre.jpg')
plt.show()

mandrill_compressee_256x256 = np.real(ifft2(ifftshift(mandrill_TF_shift_masquee)))

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(mandrill, cmap='gray')
axs[0].set_title('Image originale')
axs[1].imshow(mandrill_compressee_256x256, cmap='gray')
axs[1].set_title('Approximation, SNR=' + str(np.round(snr(mandrill, mandrill_compressee_256x256), 4)) + 'dB')
plt.savefig('mandrill_compressed.jpg')
plt.show()

###########################################
# Exemple avec encore moins de coefficients
###########################################
m = n / 16  # 32 coefficients en X et Y
mandrill_TF_shift_masquee = np.zeros((n, n), dtype=complex)

sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int)
mandrill_TF_shift_masquee[sel[0]:sel[1], sel[0]:sel[1]] = mandrill_TF_shift[sel[0]:sel[1], sel[0]:sel[1]]

fig, axs = plt.subplots(1,2)
axs[0].imshow(np.log(np.abs(mandrill_TF_shift_masquee) + 1e-8), cmap='gray')
axs[0].set_title('TF masquée: ' + str(m*m/(n*n)*100) + '% des coefficients')
axs[1].imshow(mandrill_TF_pour_visu, cmap='gray')
axs[1].set_title('TF originale')
plt.savefig('mandrill_spectre2.jpg')
plt.show()

mandrill_compressee_32x32 = np.real(ifft2(ifftshift(mandrill_TF_shift_masquee)))

fig, axs = plt.subplots(1,2)
axs[0].imshow(mandrill, cmap='gray')
axs[0].set_title('Image originale')
axs[1].imshow(mandrill_compressee_32x32, cmap='gray')
axs[1].set_title('Approximation, SNR=' + str(np.round(snr(mandrill, mandrill_compressee_32x32), 4)) + 'dB')
plt.savefig('mandrill_compressed2.jpg')
plt.show()

##############################
# Un exemple avec le cameraman
##############################
cameraman = np.load('cameraman.npy')
n = 512
m = n/4
cameraman_TF_shift_masquee = np.zeros((n, n), dtype=complex)

cameraman_TF = fft2(cameraman)
cameraman_TF_shift = fftshift(cameraman_TF)
sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int)
cameraman_TF_shift_masquee[sel[0]:sel[1], sel[0]:sel[1]] = cameraman_TF_shift[sel[0]:sel[1], sel[0]:sel[1]]

fig, axs = plt.subplots(1,2)
axs[0].imshow(np.log(np.abs(cameraman_TF_shift_masquee) + 1e-8), cmap='gray')
axs[0].set_title('TF masquée: ' + str(m*m/(n*n)*100) + '% des coefficients')
axs[1].imshow(np.log(np.abs(cameraman_TF_shift)), cmap='gray')
axs[1].set_title('TF originale')
plt.savefig('cameraman_spectre.jpg')
plt.show()

cameraman_compresse_128x128 = np.real(ifft2(ifftshift(cameraman_TF_shift_masquee)))

fig, axs = plt.subplots(1,2)
axs[0].imshow(cameraman, cmap='gray')
axs[0].set_title('Image originale')
axs[1].imshow(cameraman_compresse_128x128, cmap='gray')
axs[1].set_title('Approximation, SNR=' + str(np.round(snr(cameraman,cameraman_compresse_128x128), 4)) + 'dB')
plt.savefig('cameraman_compressed.jpg')
plt.show()

# Ajout d'un bruit aléatoire
sigma = 0.2
cameraman_bruit = cameraman + np.random.uniform(0, sigma, cameraman.shape)
plt.imshow(cameraman_bruit, cmap="gray")
plt.title('Cameraman avec du bruit')
plt.savefig('cameraman_noisy.jpg')
plt.show()

cameraman_bruit_TF = fft2(cameraman_bruit)
cameraman_bruit_TF_shift = fftshift(cameraman_bruit_TF)

plt.clf()
plt.imshow(np.log(np.abs(cameraman_bruit_TF_shift)), cmap="gray")
plt.title('TF cameraman avec bruit')
plt.savefig('cameraman_bruit_spectre.jpg')
plt.show()

m = n/2
cameraman_bruit_TF_shift_masquee = np.zeros((n, n), dtype=complex)
sel = np.array([n / 2 - m / 2, n / 2 + m / 2 + 1], dtype=int)
cameraman_bruit_TF_shift_masquee[sel[0]:sel[1], sel[0]:sel[1]] = cameraman_bruit_TF_shift[sel[0]:sel[1], sel[0]:sel[1]]

fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
axs[0].imshow(cameraman, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(cameraman_bruit, cmap='gray')
axs[1].set_title('Cameraman bruité')
axs[2].imshow(np.real(ifft2(fftshift(cameraman_bruit_TF_shift_masquee))), cmap='gray')
axs[2].set_title('Cameraman débruité avec : ' + str(m*m/(n*n)*100) + '% des coefficients')
plt.savefig('cameraman_debruitage.jpg')
plt.show()
