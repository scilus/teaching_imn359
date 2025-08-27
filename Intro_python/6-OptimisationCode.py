import numpy as np
"""
6 - Optimisation du code

Tel que brièvement expliqué plus tôt, la librairie numpy est hautement
optimisée pour travailler directement sur des matrices et des vecteurs.
Il est donc beaucoup plus performant d'effectuer des opérations globales
sur des vecteurs plutôt que de faire une boucle "for" parcourant chaque
élément du vecteur individuellement.

Dans ce contexte, il peut parfois être nécessaire de seulement
effectuer des actions sur certaines parties du vecteur. La fonction
"where" présentée plus tôt devient particulièrement indétressante.

Apprendre à vectoriser les processus python est une pratique très
importante pour arriver à bien utiliser ce langage de programmation.

"""

x = np.array([0.0, 0.01, 0.21, 0.33, 0.44, 0.45, 0.61, 0.7, 0.12, 0.19])

# Retourne un vecteur de booleen ou les valeurs
# au-dessus de 0.4 sont a True et les autres a False
val_sup = x > 0.4 

# Retourne un vecteur de booleen ou les valeurs
# inferieur a 0.6 sont a True et les autres a False
val_inf = x < 0.6

print('Vecteur: \n', x, '\n')
print('Vecteur de booléens où x > 0.4: \n', val_sup, '\n')
print('Vecteur de booléens où x < 0.6: \n', val_inf, '\n')

# l'opérateur "&" est un element-wise "and"
indices = np.argwhere((x > 0.4) & (x < 0.6))

print('Indices du vecteur où 0.4 < x < 0.6: \n', indices, '\n')

"""
À la ligne précédente, ce qui arrive vraiment est :
1- x>0.4 retourne un vecteur avec des 1 et des 0 selon si la valeur est >
    0.4
2- x<0.6 retourne un vecteur avec des 1 et des 0 selon si la valeur est <
    0.6
3- le & combine les deux vecteurs
4 - argwhere retourne l'indice de tous les endroits à 1 (true)


Éviter les boucles
==================

Il est souvent possible d'optimiser le code en éliminant les boucles.

Par exemple, le code :


Peut être remplacé par le code :
"""
count_boucle = 0
for n in range(len(x)):
    if x[n] > 0.4:
       count_boucle = count_boucle + 1

count_numpy = len(np.argwhere(x>0.4))

print("Compte du nombre d'indices < 0.4 avec boucle: ", count_boucle)
print("Compte du nombre d'indices < 0.4 sans boucle: ", count_numpy)
