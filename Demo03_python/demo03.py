import matplotlib.pyplot as plt
import numpy as np

# Exemple 1 SF
# f(x) = 3/2 + sum 3(1 - cos(n pi))/ (n pi) sin( n pi x / 5 )

x = np.arange(-20, 20, 0.1)
f = 3/2
ordre_max = 10000

for n in range(1, ordre_max):
    f += 3 * (1 - np.cos(n * np.pi)) / (n * np.pi) * (np.sin(n * np.pi * x / 5))

plt.plot(x, f, linewidth=2)
plt.savefig("demo03_ex1.jpg")
plt.show()























# Exemple 2
# f(x) = 4pi^2/3 + sum[ 4/n^2 cos(nx) - 4pi/n sin(nx)]

x = np.arange(-6 * np.pi, 6 * np.pi, 0.01)
f = 4 * pow(np.pi, 2) / 3

for n in range(1, 100000):
    f += 4 / (n * n) * np.cos(n * x) - 4 * np.pi / n * np.sin(n * x)

plt.clf()
plt.plot(x, f, linewidth=2)
plt.savefig("demo03_ex2.jpg")
plt.show()

