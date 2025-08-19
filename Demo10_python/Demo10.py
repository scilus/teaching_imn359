import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

# On charge notre pyramide de résolutions
# img est un dictionnaire. L'image originale est donnée par img['f']
# et chaque niveau de la pyramide est donné par img['f1'], img['f2'],
# etc. (jusqu'à f7).
img = np.load('cameraman_pyramid.npz')

# On visualise la pyramide.
# Chaque resize a été obtenu par interpolation bicubique.
fig, axs = plt.subplots(2, 4)
fig.set_size_inches(12, 4)
axs[0][0].imshow(img['f'], cmap='gray')
axs[0][0].set_title('Original image 512x512')
axs[0][1].imshow(img['f1'], cmap='gray')
axs[0][1].set_title('Resized image 256x256')
axs[0][2].imshow(img['f2'], cmap='gray')
axs[0][2].set_title('Resized image 128x128')
axs[0][3].imshow(img['f3'], cmap='gray')
axs[0][3].set_title('Resized image 64x64')
axs[1][0].imshow(img['f4'], cmap='gray')
axs[1][0].set_title('Resized image 32x32')
axs[1][1].imshow(img['f5'], cmap='gray')
axs[1][1].set_title('Resized image 16x16')
axs[1][2].imshow(img['f6'], cmap='gray')
axs[1][2].set_title('Resized image 8x8')
axs[1][3].imshow(img['f7'], cmap='gray')
axs[1][3].set_title('Resized image 4x4')
fig.tight_layout()
plt.show()

# Calcul un filtre gaussien de largeur mu
n = 512  # largeur du filtre
mu = 5
t = np.arange(-n//2, n//2).reshape(n, -1)
h = np.exp(-(t**2) / (2 * mu ** 2))
h = h / np.sum(h)
plt.clf()
plt.plot(t, h)
plt.show()

# on fait une convolution séparable pour aller plus vite
fhy = convolve2d(img['f'], h, 'same')
fh = convolve2d(fhy, h.T, 'same')

fig, axs = plt.subplots(2, 1)
axs[0].imshow(img['f'], cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(img['f']-fh, cmap='gray')
axs[1].set_title('Laplacian representation level 0')
plt.show()
