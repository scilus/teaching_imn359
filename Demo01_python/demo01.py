import numpy as np

from mes_fonctions import racine

# les nombres complexes
# Pour calculer les racines de nombres négatifs
# on utilise np.emath.sqrt() au lieu de np.sqrt(),
# cette dernière retournant NaN pour les nombres négatifs
z = np.emath.sqrt(-1)

print("z:", z)

# partie réelle
print("Partie réelle de z:", z.real)

# partie imaginaire
print("Partie imaginaire de z:", z.imag, "\n")

"""
On représente le nombre imaginaire "i" par le symbole "j",
acollé directement à un nombre.
"""

z1 = (-3 + 2j)
z2 = (1 - 1j)

# addition
print("z1 + z2: \n", z1+z2, "\n")

# multiplication
print("z1 * z2: \n", z1*z2, "\n")

# division
print("z1 / z2: \n", z1/z2, "\n")

# module (norme)
print("Module de z1: \n", np.abs(z1), "\n")

# conjugué complexe
print("Conjugué de z1: \n", np.conj(z1), "\n")


# Résoudre un systeme d'equations de la forme Ax=b
# revient à calculer x = inv(A)*b. La fonction .dot
# sert à calculer le produit matriciel.
A = [[2, -4], [1, 1]]
b = [8, 1]
x = np.dot(np.linalg.inv(A), b)
print("x: \n", x, "\n")


# Systeme d'equations complexes
A = [[3 + 2j, 1 + 3j],[2, 2 + 1j]]
b = [3, 5]
x = np.dot(np.linalg.inv(A), b)
print("x: \n", x, "\n")

print("Reponse de la question #1a. racine(-4): \n", racine(-4))
