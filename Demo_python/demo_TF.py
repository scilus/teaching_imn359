import cmath as cm
import matplotlib.pyplot as plt
import numpy as np

# Exemple TF de exp(-alpha * t)
# Avec ses spectres d'amplitudes et de phases

t = np.arange(0, 101)

g = np.exp(-0.05 * t)
fig, axs = plt.subplots(3,2)

axs[0][0].plot(t, g, '.-', linewidth=2)
axs[0][0].set_xlabel('t')
axs[0][0].set_ylabel('g(t)')
axs[0][0].set_title('g(t) = exp(-0.05t)')

f = np.arange(-0.5, 0.51, 0.01)
G = []
for i in f:
    G.append(1.0 / np.sqrt(pow(0.05, 2) + 4.0 * pow(np.pi, 2.0) * pow(i, 2)) * cm.exp(1j * np.arctan(-2 * np.pi * i / 0.05)))

axs[0][1].plot(f, np.real(G), '.-', linewidth=2)
axs[0][1].set_xlabel('f')
axs[0][1].set_ylabel('Real(G(f))')
axs[0][1].set_title('Real part of G(f) = TF[g(t)]')

axs[1][0].plot(f, np.imag(G), '.-', linewidth=2)
axs[1][0].set_xlabel('f')
axs[1][0].set_ylabel('Im(G(f))')
axs[1][0].set_title('Imaginery part of G(f) = TF[g(t)]')

axs[1][1].plot(f, np.abs(G), '.-', linewidth=2)
axs[1][1].set_xlabel('f')
axs[1][1].set_ylabel('abs(G(f))')
axs[1][1].set_title('Modulus of G(f) = TF[g(t)]')

axs[2][0].plot(f, np.abs(G), '.-', linewidth=2)
axs[2][0].set_xlabel('f')
axs[2][0].set_ylabel('Spectre d amplitude')
axs[2][0].set_title('Spectre d amplitude')

axs[2][1].plot(f, np.arctan(np.imag(G) / np.real(G)), '.-', linewidth=2)
axs[2][1].set_xlabel('f')
axs[2][1].set_ylabel('Spectre de phase')
axs[2][1].set_title('Spectre de phase en radian')

plt.savefig('demo05.jpg')
plt.show()



