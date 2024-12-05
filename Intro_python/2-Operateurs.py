
"""
2- Operateurs

Python supporte les opérations arithmétiques standards.
- opérations de base (+,-,*,/)
- expontenielle (**)
- parenthèse pour grouper des expressions

"""

print("7/45: ", 7/45)

print("(1+j) * (2+j): ", (1+1j) * (2+1j))

print("4**2 : ", 4**2)

print("(3+4*j)**2 : ", (3+4j)**2)

print("((2+3)*3)**0.1 : ", ((2+3)*3)**0.1)



"""
Fonctions fournies avec les librairies
Il faut d'une part importer les librairies. 
Pour les opérations mathématiques, les librairies
numpy et math ont les fonctions de base.

La documentation aide à trouver les bonnes fonctions.
"""

import numpy as np
import math

a = np.sqrt(2)
b = math.sqrt(2)

print("Racine de 2 (numpy): ", a)
print("Racine de 2 (math): ", b)



"""
La fonction transpose() de numpy avec les arrays
ou .T permet de transposer les matrices. Par contre,
lorsqu'on a un vecteur ligne de 1 dimension, pour
obtenir un vecteur colonne, il faut utiliser la
fonction reshape.

"""

m = np.array([[1, 2], [4, 3.2]])
m_transpose = m.transpose()
m_transpose_2 = m.T

print("Matrice': ", m)
print("Matrice transposée (.tranpose()): ", m_transpose)
print("Matrice transposée (.T): ", m_transpose_2)

v = np.array([1, 2, 3])
v_dim = v.shape

v_t = v.T
v_t_dim = v_t.shape

v_transpose = v.reshape((v_dim[0], 1))
v_transpose_dim = v_transpose.shape

print("Vecteur ligne: ", v)
print("Vecteur ligne dimension: ", v_dim)

print("Vecteur avec .T: ", v_t)
print("Vecteur .T dimension: ", v_t_dim)

print("Vecteur transpose avec reshape(): ", v_transpose)
print("Vecteur transpose dimension: ", v_transpose_dim)

"""
Addition/Soustraction de vecteurs/matrices
Il est possible d'ajouter ou de soustraire des vecteurs ou des matrices
si ces derniers ont la même dimension. L'adition se fait élément par
élément. Il est aussi possible de trouver la somme ou le produit de
tous les éléments d'un vecteur ou d'une matrice via les fonctions "sum"
et "prod".

"""

u = np.array([1, 2, 3, 5.4, -6.6])
v = np.array([2, 4, 6, 8, 10])

print("Vecteur u: ", u)
print("Vecteur v: ", v)

add = u + v
print("u + v: ", add)

s = u.sum()
p = u.prod()

print("Somme des éléments de u ", s)
print("Produit des éléments de u ", p)

"""
Fonctions "element-wise"

Toutes les fonctions de la librairie numpy ou presque qui fonctionnent sur un
scalaire seul peuvent aussi fonctionner sur une matrice ou un vecteur.
Si jamais une fonction ne semble pas compatible avec un vecteur,
n'hésitez-pas à consulter la documentation, la compatibilité avec les
types y sera inscrite.

"""

t = np.array([-2, 5, -3.6])
f = np.exp(t)
g = np.abs(t)

print("Vecteur t ", t)
print("Exponentielle de t ", f)
print("Valeur absolue de t ", f)

"""
Initialisation automatique
Il est courant de vouloir initialiser des grosses matrices
avec une valeur par défaut ou une séquence numérique régulière. Pour
ceci, python et numpy prévoit certaines fonctions d'initialisation, soit "ones",
"zeros", "full", "empty".

Ces fonctions initialisent une matrice ou un vecteur contenant des
valeurs analogues à leur nom. Elles prennent au minimum 1 paramètre
soit la taille de la matrice ou du vecteur en hauteur et en largeur
(dans cet ordre) dans un tuple.

Une matrice identité peut être initialisée de la même manière avec la fonction eye.
"""

o = np.ones((2, 3)) # Matrice de 2 lignes, 3 colonnes de 1
z = np.zeros((13, 1)) # Vecteur colonne de 13 éléments de 0
e = np.empty((1, 10)) # Vecteur ligne de 10 éléments vides 

print("Matrice de 1 : \n", o, "\n")
print("Vecteur de 0 : \n", z, "\n")
print("Vecteur de vide : \n", e, "\n")

id = np.eye(4) # Matrice identité de 4x4
print("Matrice identité : \n", id, "\n")


"""
Initialisation automatique de séquences

Tel qu'expliqué au point précédent, il est aussi possible d'initialiser
des séquences de nombre qui suivent un incrément particulier. Par
exemple, on peut définir une séquence avec la fonction arange ou linspace
et changer la taille avec reshape.
"""

a = np.arange(0, 20, 1) # Vecteur de longueur 20 d'éléments de 0 à 19

b = np.linspace(0, 10, 25) # Vectuer de longueur 25 d'éléments de 0 à 10

print("Vecteur avec arange: \n", a, "\n")

print("Vecteur avec linspace: \n", b, "\n")

r_a = a.reshape((4, 5))


print("Vecteur avec arange et reshape: \n", r_a, "\n")