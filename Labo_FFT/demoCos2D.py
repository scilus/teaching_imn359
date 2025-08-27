import cmath as cm
import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import ifftshift, ifft2

# Demo texture
# On se donne un spectre de Fourier 2D vide
kx = 5
ky = 8
N = 129
# En se donnant une taille impaire, le centre de l'image est facile à trouver
# ici, ce sera à (65, 65)

I = np.zeros((N, N))
centrex = np.floor(N / 2).astype(int)
centrey = np.floor(N / 2).astype(int)

# Verfier que c'est bien le centre de l'image
I[centrex, centrey] = 1 
plt.imshow(I)
plt.show()

# On remet un zéro
I[centrex, centrey] = 0

# On commence par une texture en rangée, fréquence kx
# Il faut refaire les maths pour avoir la preuve que c'est bien N^2/2 ici
I[centrex - kx, centrey] = N * (N/2)
I[centrex + kx, centrey] = N * (N/2)
plt.clf()
plt.imshow(I)
plt.show()

Ishift = ifftshift(I)
plt.clf()
plt.imshow(Ishift)
plt.show()

# TFD inverse 2D. À noter que le résultat n'a PAS de complexe qui traine
texture = np.real(ifft2(Ishift))
plt.clf()
plt.imshow(texture, cmap='gray')
plt.show()

# Texture sur les colonnes
I = np.zeros((N, N))
I[centrex, centrey - ky] = N * (N/2)
I[centrex, centrey + ky] = N * (N/2)
plt.clf()
plt.imshow(I)
plt.show()

Ishift = ifftshift(I)
plt.clf()
plt.imshow(Ishift)
plt.show()

texture = np.real(ifft2(Ishift))
plt.clf()
plt.imshow(texture, cmap='gray')
plt.show()

# Texture sur les diagonales
I = np.zeros((N, N))
I[centrex + kx, centrey + ky] = N * (N/2)
I[centrex - kx, centrey - ky] = N * (N/2)
plt.clf()
plt.imshow(I)
plt.show()

Ishift = ifftshift(I)
plt.clf()
plt.imshow(Ishift)
plt.show()

texture = np.real(ifft2(Ishift))
plt.clf()
plt.imshow(texture, cmap='gray')
plt.show()
