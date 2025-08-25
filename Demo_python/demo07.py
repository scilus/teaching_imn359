import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fft, fftshift

from fonctions.io import read_data
from fonctions.delta import delta

# Exemple d'échantillonnage
f0 = 4
t = np.linspace(0, 1, 1000) # 1000 échantillons/seconde
f = np.arange(-500, 500) # 1000 Hz
y = np.cos(2 * np.pi * f0 * t)

plt.clf()
plt.plot(t, y, linewidth=2)
plt.title(f"Signal avec une frequence de {f0} cycles/seconde")
plt.xlabel("Temps (s)")
plt.show()

plt.clf()
plt.plot(f, np.abs(fftshift(fft(y))), 'bo')
plt.title(f"TF du signal avec peak à la fréquence {f0}")
plt.xlabel("Frequence (Hz)")
plt.show()

# échantillonnage avec peigne de Dirac
ts = 20
ii = np.arange(1, len(t), ts)
t0 = t[ii]
d = delta(t, t0)

ys = y * d
t2 = t[ys != 0]
ys = ys[ys != 0]

fig, ax = plt.subplots(1, 2)
ax[0].plot(t, y, 'b-', t, d, 'ro-', linewidth=2)
ax[1].plot(t2, ys, 'bo-', linewidth=2)
plt.show()

# Exemple de repliement
lim = 16

# Différents temps d'échantillonnage
ts1 = 1/1000
ts2 = 1/100
ts3 = 1/50
ts4 = 1/8
ts5 = 1/lim

t1 = np.arange(0, 1 + ts1, ts1) # ts = 0.001 second, fs = 1000
t2 = np.arange(0, 1 + ts2, ts2) # ts = 0.01 second, fs = 100
t3 = np.arange(0, 1 + ts3, ts3) # ts = 0.05 second, fs = 20
t4 = np.arange(0, 1 + ts4, ts4) # ts = 1/8 second, fs = 8
t5 = np.arange(0, 1 + ts5, ts5) # jouez avec lim pour changer t5

# Frequence fondamentale
f1 = 8
f2 = 3

# Fonction échantillonnée pour différentes fréquences d'échantillonnage
s1 = np.cos(2 * np.pi * f1 * t1) + np.cos(2 * np.pi * f2 * t1)
s2 = np.cos(2 * np.pi * f1 * t2) + np.cos(2 * np.pi * f2 * t2)
s3 = np.cos(2 * np.pi * f1 * t3) + np.cos(2 * np.pi * f2 * t3)
s4 = np.cos(2 * np.pi * f1 * t4) + np.cos(2 * np.pi * f2 * t4)
s5 = np.cos(2 * np.pi * f1 * t5) + np.cos(2 * np.pi * f2 * t5)

fig, axs = plt.subplots(5,1)
axs[0].plot(t1, s1, '-b')
axs[1].plot(t2, s2, '-bo')
axs[2].plot(t3, s3, '-bo')
axs[3].plot(t4, s4, '-bo')
axs[4].plot(t5, s5, '-bo')
plt.show()

# Et dans Fourier...
fs1 = 1/ts1
fs2 = 1/ts2
fs3 = 1/ts3
fs4 = 1/ts4
fs5 = 1/ts5

f1 = np.arange(-fs1/2, fs1/2+1)
f2 = np.arange(-fs2/2, fs2/2+1)
f3 = np.arange(-fs3/2, fs3/2+1)
f4 = np.arange(-fs4/2, fs4/2+1)
f5 = np.arange(-fs5/2, fs5/2+1)

fig, axs = plt.subplots(4,1)
axs[0].plot(f1, np.real(fftshift(fft(s1))), 'o', linewidth=2)
axs[1].plot(f2, np.real(fftshift(fft(s2))), 'o', linewidth=2)
axs[2].plot(f3, np.real(fftshift(fft(s3))), 'o', linewidth=2)
axs[3].plot(f4, np.real(fftshift(fft(s4))), 'o', linewidth=2)
plt.show()

plt.clf()
plt.plot(f5, np.real(fftshift(fft(s5))), 'o', linewidth=2)
plt.show()
plt.clf()

# Echantillonnage du piece regular
M = read_data('piece-regular_1024.npy')

# C'est comme si on avait echantillonne le signal a tous les 1/1024 pts
# ou 1024 Hz
M = M.squeeze()
t = np.linspace(0, 1, np.shape(M)[0])
plt.plot(t, M, '--.', linewidth=2)
plt.show()

plt.clf()
# comme on a 1024 pts, ca met au monde 512 frequences, entre -512 et 512
f = np.linspace(-512, 512, 1024)
plt.plot(range(-512, 512), fftshift(np.abs(fft(M))), '--.r', linewidth=2)
plt.show()

# On voit que la fft est symetrique. En fait, on a que de l'information sur
# 1024/2 frequences a cause des coefficients complexe de la TF
plt.clf()
plt.plot(f, np.abs(fftshift(fft(M))))
plt.show()
