import cmath as cm
import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fft, fftshift, ifft

# Demo d'un cos qui oscille N fois par periode
k0 = 4
N = 1001
n = np.arange(0, N)
y = np.cos(2 * np.pi / N * k0 * n)

plt.plot(n , y, '.', linewidth=2)
plt.show()

Y = fft(y)
plt.clf()
plt.plot(np.abs(Y), 'o-', linewidth=2)
plt.show()

plt.clf()
plt.plot(fftshift(np.abs(Y)), 'o-', linewidth=2)
plt.show()

# demoCosinus 1D
N = 128
n = np.arange(0, N)
F = np.zeros(N)
plt.clf()
plt.plot(F, '.--', linewidth=2)
plt.show()

# On veut que le cosinus oscille k fois sur intervalle 128
# La preuve est dans les diapos
k = 5
F[k] = N / 2
F[N - k] = N / 2
iF = np.real(ifft(F))
plt.clf()
plt.plot(F, '.--', linewidth=2)
plt.show()
plt.clf()
plt.plot(n, iF, '.--', linewidth=2)
plt.show()

f = np.cos(2 * np.pi / N * k *n)
plt.clf()
plt.plot(n, f, 'g', linewidth=2)
plt.show()

fig, axs = plt.subplots(1,2)
axs[0].plot(F)
axs[1].plot(np.abs(fft(f)), 'r', linewidth=2)
plt.show()

fig, axs = plt.subplots(1,2)
axs[0].plot(iF)
axs[1].plot(np.real(ifft(fft(f))), 'r', linewidth=2)
plt.show()
