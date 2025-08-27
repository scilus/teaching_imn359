import numpy as np

"""
1 - Variables

En programmation, on utilise des "variables" pour garder en mémoire
la valeur des éléments impliquées dans un traitement ou calcul.

Contrairement à des langages typés comme C et C++, en python les
variables sont faiblement typées et n'ont pas besoin d'être explicitement
initialisée. Il est donc possible de simplement ajouter une nouvelle
variable (sortie de nulle part) en plein milieu d'une séquence de
commande et cette dernière sera allouée.

Python possède quelques types de base, mais ces derniers sont
habituellement abstraits pour rester simple et transparent. Les
différences principales se situent au niveau des variables de type
numérique et des variables textuelles, la plupart des types de variable
sont compatible à l'exception du passage implicite numérique/texte ou
inversement.

Parmi les types supportés, on note les complexes, les entiers, les
variables symboliques, les réels, etc.

Pour créer une variable, on peut simplement assigner une valeur à un
nom.

"""

# initialisation de variables
var1 = 3.14  # nombre à virgule flottante
var2 = 2  # nombre entier
chaineDeTexte = 'Salut la planète!'  # chaine de caractères

# type de chacune des variables
print("Type de var1: \n", type(var1), "\n")
print("Type de var2: \n", type(var2), "\n")
print("Type de chaineDeTexte : \n", type(chaineDeTexte), "\n")

"""
La seule contrainte pour les noms de variable est que ces dernières
doivent commencer par une lettre. Par la suite, n'importe quelle
combinaison de lettre ou de '_' peuvent être utilisées.

*IMPORTANT*: Les variables python suivent la casse, la variable "var1"
ne serait donc pas identique à la variable "Var1".

"""

var1 = 3
Var1 = 2

print("var1= ", var1, "\n")
print("Var1= ", Var1, "\n")

"""
Une variable peut aussi avoir une valeur explicitement définie comme
étant le résultat d'une fonction utilisant des valeurs connues.

"""
a = 10
c = 1.3 * 45 - 2 * a

"""
Tableux, Matrices et Vecteur
============================

Il existe en python, comme dans plusieurs langages de programmation,
la notion de tableau ou de matrice. Cette structure est en fait la
force de Python puisque Python est optimisé pour les calculs sur de
grosses séries de données contenues dans des tableaux.

Notons que sous Python, une matrice et un vecteur sont tous deux
considérés comme étant des tableaux.

Vecteur
-------

Un vecteur est une matrice dont une des deux dimensions vaut 1. On
fait la distinction entre le vecteur ligne (de taille 1 x N) et le
vecteur colonne (de taille N x 1).

Les vecteurs sont définis à l'aide des crochets rectangulaires. Les
valeurs du vecteur sont listées et séparées par des virgules. Il
faut utiliser la libraire numpy qui supporte les arrays.

"""

row = np.array([[1, 2, 3, 5.4, -6.6]])
column = np.array([[1], [4], [10.2], [-5], [3]])
print("Le vecteur ligne est:\n", row, "\n")
print("Le vecteur colonne est:\n", column, "\n")

"""
Taille d'un vecteur
-------------------

La taille d'un vecteur peut être obtenue via la fonction "vecteur.shape".
La taille retournée est un tuple qui dépend de la dimension du tableau,
selon l'ordre (nombre_de_lignes, nombre_de_colonnes).

"""

print("Taille du vecteur ligne: \n", row.shape, "\n")

print("Taille du vecteur colonne: \n", column.shape, "\n")

"""
La fonction "len" permet aussi d'obtenir le NOMBRE DE LIGNES
d'un tableau.
"""

print("len() du vecteur ligne: \n", len(row), "\n")

print("len() du vecteur colonne: \n", len(column), "\n")

"""
Matrices

La création d'une matrice est très similaire à la création d'un
vecteur, excepté qu'on combine ici la notion de ligne et de colonne.
Par exemple, pour faire une matrice 3x2, on utiliserait le code suivant

"""

a = np.array([[1, 2], [3, 4], [5, 6]])

print("Matrice 3x2: \n", a, "\n")


"""
Finalement, on peut créer un tableau à partir de tableaux de même
taille. Ici, on crée un tableau d où les valeurs de chaque ligne
proviennent respectivement des tableaux a, b et c.
"""

a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])

d = np.array([a, b, c])

print("Matrice : \n", d, "\n")
