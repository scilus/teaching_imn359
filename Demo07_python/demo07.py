import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from scipy.fft import fft, fftshift

from delta import delta

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


plt.clf()
ts = 20
ii = np.arange(1, len(t), ts)
t0 = t[ii]
d = delta(t, t0)


plt.clf()
plt.plot(t, y, 'b-', t, d, 'ro-', linewidth=2)
plt.show()

ys = y * d
t2 = t[ys != 0]
ys = ys[ys != 0]

plt.clf()
plt.plot(t2, ys, 'bo-', linewidth=2)
plt.show()


# Exemple de repliement
lim = 17
t1 = np.arange(0, 1, 1/(1000+1)) # ts = 0.001 second, fs = 1000
t2 = np.arange(0, 1, 1/(100+1)) # ts = 0.01 second, fs = 100
t3 = np.arange(0, 1, 1/(50+1)) # ts = 0.05 second, fs = 20
t4 = np.arange(0, 1, 1/(8+1)) # ts = 1/8 second, fs = 8
t5 = np.arange(0, 1, 1/lim)

f1 = 8
f2 = 3

# Frequence fondamentale
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

f1 = range(-500, 500+1)
f2 = range(-50, 50+1)
f3 = range(-25, 25+1)
f4 = range(-4, 4+1)
f5 = range(int(-lim/2), int(lim/2)+1)

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



# Echantillonnage du piece regulare
M = loadmat("piece-regular.mat")['x0']

# C'est comme si on avait echantillonne le signal a tous les 1/1024 pts
# ou 1024 Hz
M = loadmat("piece-regular.mat")['x0']
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

