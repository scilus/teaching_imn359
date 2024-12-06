import numpy as np

"""
4 - Graphiques

La librairie la plus utilisée pour faire des graphiques
avec Python est matplotlib. L'interface de matplotlib pour
produire les graphiques est pyplot.

"""

import matplotlib.pyplot as plt

"""
Il est possible de générer le graphique d'un vecteur en fonction de ses
indexes, par exemple :

"""

vec = np.array([1, 2, 3, 0, 0, 2])
plt.plot(vec)
plt.title('Graphe avec un seul vecteur')
plt.show()

"""
On peut faire de même mais en spécifiant explicitement les valeurs sur
Y et les valeurs sur X à utiliser. Dans ce contexte, faites attention!
X et Y doivent être de la même taille, sinon python produira une erreur.

"""
x = np.linspace(0, 4*np.pi, 10)
y = np.sin(x)

plt.plot(y) # sans specifier x
plt.title('Graphe sin sans spécifier x')
plt.show()

plt.plot(x, y) # en spécifiany x (regardez l'axe des x, les valeurs sont différentes)
plt.title('Graphe de sin en spécifiant x et y')
plt.show()

"""
Configuration des options d'affichage
=====================================

Sans trop entrer dans les détails, notons qu'il est possible de
configurer l'apparence de votre graphique. Par exemple, il est possible
de configurer le trait du graphique avec un code relativement simple où on
indique la couleur, le type de point et le type de ligne à afficher.

L'argument "linestyle" permet d'indiquer le type de trait à afficher.

"""
plt.plot(x, y, color='green', linestyle='-.')
plt.title("Graphe de sin en changeant l'affichage")
plt.show()

plt.plot(x, y, color='red', linestyle='--')
plt.title("Graphe de sin en changeant l'affichage")
plt.show()


"""
Les options de configuration sont beaucoup plus avancées en réalité,
pour les voir, allez regarder l'aide :).

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

"""

"""
Visualiser une matrice

Une matrice peut être visualisée comme une grille de valeurs 2D en
utilisant la fonction imshow. On peut charger une image en format .npy
en utilisant la fonction load() de la librairie numpy.

"""

mat = np.random.rand(8, 8)
plt.imshow(mat)
plt.show()

lena = np.load("lena_uint8.npy")

plt.imshow(lena)
plt.show()

"""
Il est possible de changer les couleurs en niveaux de gris
plutôt que d'afficher des teintes de bleu/jaune.

"""
plt.imshow(lena, cmap='gray')
plt.show()

"""
Graphiques de surfaces 3D

Il est aussi possible d'afficher des graphiques en 3D sous forme de
surfaces tridimensionnelles. On utilise la fonction "meshgrid" qui nous
génère nos points en X et en Y.
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)

plt.show()
