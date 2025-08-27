
import numpy as np
"""
3 - Indexage de vecteur

En programmation, l'indexage permet d'accéder à un ou des éléments d'un
tableau. L'indexage en python commence à 0. Pour accéder à un élément précis
d'une matrice, on utilise l'opérateur [] en y insérant l'index désiré

"""

a = np.array([13, 5, 9, 10])

print("Vecteur: ", a)

print("a[0]: ", a[0])

print("a[1]: ", a[1])

print("a[3]: ", a[3], "\n")


"""
L'index peut aussi définir une plage de valeurs. Pour définir
une plage de valeurs allant de l'index a (inclus) à l'index b (exclus),
on utilise la notation [a:b].

"""

a = np.array([1, 2, 1, 3, 4, 5, 5, 4, 3])
a_sub = a[3:6]  # de l'index 3 (inclus) à l'index 6 (exclus)

print("Vecteur: ", a)
print("Vecteur tronqué: ", a_sub)

# Si on omet le premier indice de la plage de valeur, python
# assume qu'on commence de l'index 0. À l'inverse, si on omet
# le dernier indice de la plage de valeurs, on finit au dernier
# index du tableau.
a_end = a[4:]
a_beg = a[:5]

print("Vecteur coupé au début: ", a_end)
print("Vecteur coupé à la fin: ", a_beg, "\n")

"""
Indexage de matrice

L'indexage de matrice peut se faire avec un seul indice linéaire ou 
une série de sous-indices. Lorsqu'un indice linéaire est utilisé, la
matrice est parcourue de haut en bas puis de gauche à droite. Lorsqu'une
série de sous-indices est utilisé, le premier indice correspond à la
ligne et le second à la colonne.

On peut aussi vouloir sélectionner seulement une ligne ou une colonne
de la matrice. Dans ce cas, on utilise un interval ouvert ":" qui
signifie "du premier au dernier indice possible".

"""

m = np.array([[1, 2], [4, 3.2]])
m_ligne = m[1]
m_colonne = m[:, 0]

m_el = m[0,1]

print("Matrice \n", m, "\n")
print("2e ligne de la matrice \n", m_ligne, "\n")
print("1ere colonne de la matrice \n", m_colonne, "\n")
print("Élément à la 1ere ligne, 2e colonne \n", m_el, "\n")


"""
Indexage avancé

Numpy contient des fonctions servant à retourner des indices en
particulier. Il est possible d'utiliser argwhere et where
afin de générer des formulations extrêmement performantes sous python.

"""

n = np.array([[1, 0, 5], [5, 7, 0], [4, 3, -1]])

n_zero = np.argwhere(n==0)

print("Matrice \n", n, "\n")
print("Indices des éléments qui valent 0 \n", n_zero, "\n")