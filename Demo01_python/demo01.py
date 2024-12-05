import cmath as cm
import numpy as np

from mes_fonctions import racine

# les nombres complexes
# Numpy ne supporte pas sqrt(-1)
z = cm.sqrt(-1)

# partie réelle
print("Partie réelle: \n", z.real, "\n")

# partie imaginaire
print("Partie imaginaire: \n", z.imag, "\n")
    
# addition
z1 = (-3 + 2j)
z2 = (1 - 1j)
print("z1+z2: \n", z1+z2, "\n")

# multiplication
z1 = (-3 + 2j)+(1 - 1j)
z2 = 1 - 5j
print("z1*z2: \n", z1*z2, "\n")

# division
print("z1/z2: \n", z1/z2, "\n")


# modulus
print("Module de z1: \n", np.abs(z1), "\n")

# conjugate
print("Conjugué de z1: \n", np.conj(z1), "\n")


# Systeme d'equations
A = [[2, -4], [1, 1]]
b = [8, 1]
x = np.divide(A, b)
print("A/b: \n", x, "\n")


# Systeme d'equations complexes
A = [[3 + 2j, 1 + 3j],[2, 2 + 1j]]
b = [3, 5]
x = np.divide(A, b)
print("A/b: \n", x, "\n")


print("Reponse de la question #1a. racine(-4): \n", racine(-4))










