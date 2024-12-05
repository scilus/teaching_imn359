import numpy as np

"""
1- Variables

Sous python, les variables sont faiblement typées et n'ont pas besoin
d'être explicitement initialisée. Il est donc possible de simplement
ajouter une nouvelle variable (sortie de nulle part) en plein milieu
d'une séquence de commande et cette dernière sera allouée.

Python possède quelques types de base mais ces derniers sont
habituellement abstraits pour rester simple et transparents. Les
différences principales se situent au niveau des variables de type
numérique et des variables textuelles, la plupart des types de variable
sont compatible à l'exception du passage implicite numérique/texte ou
inversement.

Parmi les types supportés, on note les complexes, les entiers, les
variables symboliques, les réels, etc.

Pour créer une variable, on peut simplement assigner une valeur à un
nom.

"""

var1 = 3.14
chaineDeTexte = 'Salut la planète!'

print("Type de var1: \n", type(var1), "\n")
print("Type de chaineDeTexte : \n", type(chaineDeTexte), "\n")

"""
La seule contrainte pour les noms de variable est que ces dernières
doivent commencer par une lettre. Par la suite, n'importe quelle
combinaison de lettre ou de '_' peuvent être utilisées.

*IMPORTANT* : Les variables matlab suivent la casse, la variable "var1"
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

Il existe, comme dans plusieurs langages de programmation,
la notion de tableau ou de matrice. Cette structure est en fait la
force de Python puisque Python est optimisé pour
les calculs sur de grosses séries de données contenues dans des
tableaux.

Notons que sous Python, une matrice ou un vecteur sont tous deux
considérés comme étant des tableaux.

Vecteur Ligne

Un vecteur ligne se défini à l'aide des crochets rectangulaires. Les
valeurs du vecteur sont listées et séparées par des virgules. Il
faut utiliser la libraire numpy qui supporte les arrays.

"""

row = np.array([1, 2, 3, 5.4, -6.6])
print("Le vecteur ligne est:", row, "\n")

"""
Vecteur colonne

Python fait une différence entre les vecteurs ligne et vecteurs
colonnes. Assurez-vous donc que ces derniers sont clairement défini.

"""

column = np.array([[1], [4], [10.2], [-5], [3]])

"""
Taille d'un vecteur

La taille d'un vecteur peut être obtenue via la fonction
"vecteur.shape".
La taille retournée est un tuple qui dépend de la dimension du array
sous la forme "ligne colonne".

"""

print("Taille du vecteur ligne: \n", row.shape, "\n")

print("Taille du vecteur colonne: \n", column.shape, "\n")

"""
Si vous souhaitez uniquement avoir la longueur du vecteur, vous pouvez
aussi utiliser la fonction "len", vous ne pourrez cependant plus
faire la différence entre un vecteur ligne ou colonne
"""

print("Longueur du vecteur ligne: \n", len(row), "\n")

print("Longueur du vecteur colonne: \n", len(column), "\n")

"""
Matrices

La création d'une matrice est très similaire à la création d'un
vecteur, excepté qu'on combine ici la notion de ligne et de colonne.
Par exemple, pour faire une matrice 2x2, on utiliserait le code suivant

"""

a = np.array([[1, 2], [3, 4]])

print("Matrice 2x2: \n", a, "\n")

a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])

d = np.array([a, b])

print("Matrice : \n", d, "\n")
