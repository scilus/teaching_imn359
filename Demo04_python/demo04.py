import matplotlib.pyplot as plt
import numpy as np

# Exemple 1 SF complexe
# f(t) = -2A/pi sum (1 / (4n^2 -1) e^(i2pi*nt))

A = 10
t = np.arange(-3.0, 3.01, 0.01)
f1 = np.abs(A * np.sin(np.pi * t)) # petit truc pour rendre sin periodique de periode T

# on initialise f à 0 et on évalue la série pour n = [-10, 10]
f=0
for n in range(-10, 11):
    f += (-2 * A / np.pi * (1 / (4 * n * n - 1)) + 0j) * np.exp(1j * 2 * np.pi * n * t)

plt.plot(t, f.real, 'b', linewidth=2)
plt.plot(t, f1, 'r', linewidth=3, linestyle='dotted')
plt.legend(['SF', 'Fonction originale'])
plt.savefig('demo04_ex1.jpg')
plt.show()

# Exemple 2 SF complexe
# f(t) = Ad/T sum( sinc( w*n*d / 2) * exp(i*w*n*t);
A = 1
T = 20  # (2 * d)
d = 10
w = 2 * np.pi / T
t = np.arange(-35, 35.1, 0.1)
f=0
for n in range(-10000, 10000):
    arg = w * n * d / 2
    if (arg != 0):
        f += A * d / T * np.sin(arg) / arg * np.exp(1j * w * n * t)
    else:
        f += A * d / T * np.exp(1j * w * n * t)

plt.clf()
plt.plot(t, f.real, linewidth=2)
plt.savefig('demo04_ex2.jpg')
plt.show()

# Spectre d'amplitude
# f(w) = |c_n| pour n valeur de w

w = np.arange(-1600 * np.pi, 1600 * np.pi, 10)
A = 1
T = 1/2
d = 1/20
arg = w * d /2
f = A * d / T * np.sin(arg) / arg

plt.clf()
plt.plot(w, f, '.-', linewidth=2)
plt.savefig('demo04_ex2_amp.jpg')
plt.show()

# TODO: What's this next demo about?
K = 100000
x = np.arange(-np.pi, np.pi, 0.01)
f = 0

for n in range(1, K):
    f += 2 * pow(-1, n+1) / n * np.sin(n * x)

plt.clf()
plt.plot(x, f, 'b')
plt.plot(x, x, 'r', linestyle='dotted', linewidth=3)
plt.title('Nombre d harmoniques n = {}'.format(str(K)))
plt.savefig('demo04_ex2_erreur.jpg')
plt.show()

err = np.mean((x-f)**2)
print('Erreur quadratique numerique = {}'.format(err))

# Gibbs ringing
x = np.arange(-20, 20.1, 0.1)
ordre_max = 50

f = 3/2

for n in range(1, ordre_max + 1):
    f += 3 * (1 - np.cos(n * np.pi)) / (n * np.pi) * np.sin(n * np.pi * x / 5)

plt.clf()
plt.plot(x, f, linewidth=2)
plt.savefig('demo04_gibbs.jpg')
plt.show()

