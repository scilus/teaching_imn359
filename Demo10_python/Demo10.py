import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d

img = loadmat('cameraman_pyramid.mat')

# Pyramid representation of the cameran man
# Each resize was obtained with a bicubic interpolation
fig, axs = plt.subplots(2, 4)
axs[0][0].imshow(img['f'], cmap='gray')
axs[0][0].set_title('Original image 512x512')
axs[0][1].imshow(img['f1'], cmap='gray')
axs[0][1].set_title('Original image 256x256')
axs[0][2].imshow(img['f2'], cmap='gray')
axs[0][2].set_title('Original image 128x128')
axs[0][3].imshow(img['f3'], cmap='gray')
axs[0][3].set_title('Original image 64x64')
axs[1][0].imshow(img['f4'], cmap='gray')
axs[1][0].set_title('Original image 32x32')
axs[1][1].imshow(img['f5'], cmap='gray')
axs[1][1].set_title('Original image 16x16')
axs[1][2].imshow(img['f6'], cmap='gray')
axs[1][2].set_title('Original image 8x8')
axs[1][3].imshow(img['f7'], cmap='gray')
axs[1][3].set_title('Original image 4x4')
plt.show()

### 
n = 512
mu = 5

# Compute a Gaussian filter of width mu
t = np.arange(-256, 256).reshape(n, -1)
h = np.exp(-(t**2) / (2 * mu ** 2))
h = h / np.sum(h)
plt.clf()
plt.plot(t, h)
plt.show()

fhy = convolve2d(img['f'], h, 'same')
fh = convolve2d(fhy, h, 'same')

fig, axs = plt.subplots(2, 1)
axs[0].imshow(img['f'], cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(img['f']-fh, cmap='gray')
axs[1].set_title('Laplacian representation level 0')
plt.show()
