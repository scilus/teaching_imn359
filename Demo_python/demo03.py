"""
Séries de Fourier
"""
import matplotlib.pyplot as plt
import numpy as np

# Exemple 1 SF
# f(x) = 3/2 + sum 3(1 - cos(n pi))/ (n pi) sin( n pi x / 5 )

# valeurs pour lesquelles on évalue f(x)
x = np.arange(-20, 20, 0.1)

# on initialise f avec la valeur à l'extérieur de la somme (3/2)
f = 3/2

# borne maximale de la somme sur n
ordre_max = 10000

# on calcule la somme pour n=1..ordre_max
for n in range(1, ordre_max):
    f += 3 * (1 - np.cos(n * np.pi)) / (n * np.pi) * (np.sin(n * np.pi * x / 5))

plt.plot(x, f, linewidth=2)
plt.savefig("demo03_ex1.jpg")
plt.show()

# Exemple 2
# f(x) = 4pi^2/3 + sum[ 4/n^2 cos(nx) - 4pi/n sin(nx)]

x = np.arange(-6 * np.pi, 6 * np.pi, 0.01)
f = 4 * pow(np.pi, 2) / 3
ordre_max = 100000

for n in range(1, ordre_max):
    f += 4 / (n * n) * np.cos(n * x) - 4 * np.pi / n * np.sin(n * x)

plt.clf()
plt.plot(x, f, linewidth=2)
plt.savefig("demo03_ex2.jpg")
plt.show()
