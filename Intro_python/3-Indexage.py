
import numpy as np
"""
3- Indexage de vecteur

L'indexage en python est basé à  0, il faut donc faire
attention lorsqu'on accède à un élément d'une matrice ou d'un vecteur
Pour accéder à un élément précis d'une matrice, on utilise l'opérateur
[] en y insérant l'index désiré

"""

a = np.array([13, 5, 9, 10])

print("Vecteur: ", a)

print("a[0]: ", a[0])

print("a[1]:  \n", a[1], "\n")

print("a[3] \n", a[3], "\n")


"""
L'index peut aussi être un vecteur de plusieurs indexes, ce qui
retournera plusieurs valeurs correspondants aux indexes contenus dans le
vecteur.

"""

a = np.array([1, 2, 1, 3, 4, 5, 5, 4, 3])
a_sub = a[3:6]

print("Vecteur: \n", a, "\n")
print("Vecteur tronqué \n", a_sub, "\n")

a_end = a[4:]
a_beg = a[:5]

print("Vecteur coupé au début: \n", a_end, "\n")
print("Vecteur coupé à la fin \n", a_beg, "\n")

"""
Indexage de matrice

L'indexage de matrice peut se faire avec un seul indice linéaire ou 
une série de sous-indices. Lorsqu'un indice linéaire est utilisé, la
matrice est parcourue de haut en bas puis de gauche à droite. Lorsqu'une
série de sous-indices est utilisé, le premier indice correspond à la
ligne et le second à la colonne.

On peut vouloir aussi sélectionner seulement une ligne ou une colonne
de la matrice. Dans ce cas, on utilise un intervale ouvert ":" qui
signifie "tous les indices possibles".

"""

m = np.array([[1, 2], [4, 3.2]])
m_ligne = m[1]
m_colonne = m[:, 0]

m_el = m[0,1]

print("Matrice \n", m, "\n")
print("2e ligne de la matrice \n", m_ligne, "\n")
print("1ere colonne de la matrice \n", m_colonne, "\n")
print("2e élément de la matrice \n", m_el, "\n")


"""
Indexage avancé

Numpy contient des fonctions servant à retourner des indices en
particulier. Il est possible d'utiliser argwhere et where
afin de générer des formulations
extrêmement performantes sous python.

"""

n = np.array([[1, 0, 5], [5, 7, 0], [4, 3, -1]])

n_zero = np.argwhere(n==0)

print("Matrice \n", n, "\n")
print("Indices des éléments =0 \n", n_zero, "\n")