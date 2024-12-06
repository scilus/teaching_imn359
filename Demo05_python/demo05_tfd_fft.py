import matplotlib.pyplot as plt
import numpy as np


E = [[1, 1, 1, 1],
     [1, -1j, -1, 1j],
     [1, -1, 1, -1],
     [1, 1j, -1, -1j]]

g = [0, 1, 2, 3]

# TFD
G = np.dot(E, g)
print(G)

# iTFD
g3 = np.dot(G, np.conj(E))
print(g3)

g3 = 1/4*np.dot(G, np.conj(E))
print(g3)

g4 = np.fft.fft(g)
print(g4)


from scipy.fft import fft, fftshift, ifft 
g = np.load("piece_regular.npy")
G = fft(g)
t = np.linspace(0, 1, np.shape(g)[0])
f = np.linspace(-256, 256, 512)
plt.plot(t, g, '-o')
plt.savefig('demo05_fft_raw.jpg')
plt.show()

plt.plot(f, np.real(G), '-o')
plt.savefig('demo05_real_fft.jpg')
plt.show()

plt.plot(f, np.real(fftshift(G)), '-o')
plt.savefig('demo05_real_fftshift.jpg')
plt.show()

plt.plot(f, np.imag(fftshift(G)), '-o')
plt.savefig('demo05_imag_fftshift.jpg')
plt.show()

plt.plot(f, np.abs(fftshift(G)), '-o')
plt.savefig('demo05_abs_fftshift.jpg')
plt.show()


###################################
# Quelques propriétés importantes
###################################

# Le centre du spectre de Fourier est égal à la somme du signal piece-regular
# Faites les maths!
# G[0] = sum_i^N g[i] exp(0)
print(np.sum(g))
print(G[0])
print(np.real(G[0]))

# Théorème de Parseval - conservation d'énergie entre le signal dans le temps et le monde des fréquences
# quantité d'énergie dans le piece-regular (le signal)
# la somme au carré de toutes les intensités
# un peu équivalent à l'aire sous la courbe en valeur absolu
print(np.sum(g**2))

#quantité d'énergie dans le spectre de Fourier
print(np.sum(np.abs(G)**2))
# on divise par le nombre d'échantillons N dans le signal
# à cause des maths et du 1/N dans la transformée inverse
N = g.size
print(np.sum(np.abs(G)**2) / N)

# C'est la même chose pour les images comme on verra dans la demo06
