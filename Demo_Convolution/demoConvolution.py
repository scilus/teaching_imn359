import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, fftshift, ifft2

from utils import add_square_around_zoom

##################
# CONVOLUTION 1D
##################

piece_regular = np.load('piece-regular.npy')

plt.plot(piece_regular)
plt.show()

moy3 = np.convolve(piece_regular, [1/3]*3)
moy20 = np.convolve(piece_regular, [1/20]*20)
moy50 = np.convolve(piece_regular, [1/50]*50,'same')

plt.figure()
plt.plot(piece_regular, label='Original')
plt.plot(moy3, label='Resultat moy 3')
plt.plot(moy20, label='Resultat moy 20')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(piece_regular, label='Original')
plt.plot(moy3, label='Resultat moy 3')
plt.plot(moy20, label='Resultat moy 20')
plt.plot(moy50, label='Resultat moy 50')
plt.legend(loc='upper right')
plt.show()

# La dérivée de f(x) à la coordonnée x=x0 s'approxime par
# df(x=x0) = f(x0 + 1)/2 - f(x0 - 1)/2. Puisque la convolution
# inverse les éléments du filtre, on passe le filtre [1/2, 0, -1/2].
deriv = np.convolve(piece_regular, [1 / 2, 0, -1 / 2])
_, axs = plt.subplots(2, 1)
axs[0].plot(piece_regular)
axs[0].set_title('Original')
axs[1].plot(deriv)
axs[1].set_title('Dérivée centrée')
plt.show()

##################
# CONVOLUTION 2D
##################

# Loading Lena
lena = np.load("lena_float.npy")
plt.imshow(lena, cmap='gray')
plt.show()
print("Résolution: {}".format(lena.shape))

# Exemple 1: Moyenne (Filtre 1D voisinage 3 vs voisinage 10)
filtre_g = [[1 / 3, 1 / 3, 1 / 3]]
moy3 = convolve2d(lena, filtre_g)

filtre_g = [[1 / 10] * 10]
Moy10 = convolve2d(lena, filtre_g)

_, axs = plt.subplots(1, 3)
axs[0].imshow(lena, cmap='gray')
axs[0].set_title('Lena')
axs[1].imshow(moy3, cmap='gray')
axs[1].set_title('Moyenne 3 voisins')
axs[2].imshow(Moy10, cmap='gray')
axs[2].set_title('Moyenne 10 voisins')
plt.show()

# Exemple 2: Moyenne (Filtre 2D voisinage 5x5)
filtre_g = [[1 / 25] * 5] * 5
moy3 = convolve2d(lena, filtre_g)

zoom1x = 100
zoom1y = 20
zoom2x = 250
zoom2y = 70
width = 100
lena_show_zooms = lena.copy()
lena_show_zooms = add_square_around_zoom(lena_show_zooms, zoom1x, zoom1y, width)
lena_show_zooms = add_square_around_zoom(lena_show_zooms, zoom2x, zoom2y, width)


_, axs = plt.subplots(3, 2)
axs[0][0].imshow(lena_show_zooms, cmap='gray')
axs[0][0].set_title('Lena')
axs[0][1].imshow(moy3, cmap='gray')
axs[0][1].set_title('Moyenne 5x5 voisins')
axs[1][0].imshow(lena[zoom1x:zoom1x + width, zoom1y:zoom1y + width], cmap='gray')
axs[1][0].set_title('Lena, zoom 1')
axs[1][1].imshow(moy3[zoom1x:zoom1x + width, zoom1y:zoom1y + width], cmap='gray')
axs[1][1].set_title('Moyenne 5x5 voisins, zoom 1')
axs[2][0].imshow(lena[zoom2x:zoom2x + width, zoom2y:zoom2y + width], cmap='gray')
axs[2][0].set_title('Lena, zoom 2')
axs[2][1].imshow(moy3[zoom2x:zoom2x + width, zoom2y:zoom2y + width], cmap='gray')
axs[2][1].set_title('Moyenne 5x5 voisins, zoom 2')
plt.show()


# Exemple 3: Derivée
filtre_g = [[-1 / 2, 0, 1 / 2]]
I_x_centered = convolve2d(lena, np.array(filtre_g).T)
I_y_centered = convolve2d(lena, filtre_g)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(I_x_centered, cmap='gray')
axs[0].set_title('Gradient centré, axe x')
axs[1].imshow(I_y_centered,  cmap='gray')
axs[1].set_title('Gradient centré, axe y')
plt.show()


# Exemple 4 : Convolution avec un filtre très gros. Mêmes dimensions que Lena.
filtre_g = np.zeros((512, 512))
filtre_g[246: 267, 246: 267] = np.ones((21, 21))
plt.figure()
plt.imshow(filtre_g)
plt.title("Filtre")
plt.show()

start = time.time()
C1 = convolve2d(lena, filtre_g, 'same')
end = time.time()
print("Time elapsed convolv: {}".format(end - start))  # --> Très long!!

# Beaucoup plus rapide d'utiliser le thm de convolution
start = time.time()
convolve_in_fft_world = fft2(lena) * fft2(filtre_g)
CC = np.real(fftshift(ifft2(convolve_in_fft_world)))
end = time.time()
print("Time elapsed with fft: {} seconds".format(end - start))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.log(np.abs(fftshift(fft2(lena)))))
axs[1].imshow(np.log(np.abs(fftshift(fft2(filtre_g)))))
plt.show()

plt.figure()
plt.imshow(CC)
plt.show()


# On devine que c'est la même chose que la boîte seule sans les zéro autour.
filtre_g = np.ones((21, 21))
C2 = convolve2d(lena, filtre_g)


fig, axs = plt.subplots(1, 2)
axs[0].imshow(CC)
axs[1].imshow(C2)
plt.show()


# Energy dans Lena
energy = np.sum(np.abs(pow(lena, 2)))
print(energy)

IF = fft2(lena)

energy2 = np.sum(np.abs(pow(IF, 2)))
print(energy2)

test = energy2 / pow(len(lena), 2)
print(test)
