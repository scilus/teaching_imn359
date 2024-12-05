import numpy as np

"""
4- Graphes

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
On peut faire demême mais en spécifiant spécifiquement les valeurs sur
Y et les valeurs sur X à utiliser. Par exemple :

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
Dans ce contexte, faites attention! X et Y doivent être de la même
taille sinon python produira une erreur.

"""

"""
Configuration des options d'affichage.

Sans entrer trop dans le détail, notons qu'il est possible de
configurer l'affichage de votre graphe. Par exemple, il est possible de
configurer le trait du graph avec un code relativement simple où on
indique la couleur, le type de point et le type de ligne à afficher.

"""
plt.plot(x, y, color='green', linestyle='-.')
plt.title("Graphe de sin en changeant l'affichage")
plt.show()

plt.plot(x, y, color='red', linestyle='--')
plt.title("Graphe de sin en changeant l'affichage")
plt.show()


"""
Les options de configuration sont beaucou plus avancées en réalité,
pour les voir, allez regarder l'aide :).

"""

"""
Visualiser une matrice

Une matrice peut être visualisée comme une grille de valeurs 2D en
utilisant la fonction imshow. Pour une image en format .mat,
il faut aller la chercher en utilisant fonction loadmat dans la librairie
scipy.io

"""
from scipy.io import loadmat

mat = np.random.rand(8, 8)
plt.imshow(mat)
plt.show()

lena = loadmat("lena.mat")['lena']

plt.imshow(lena)
plt.show()

"""
Il est possible de changer la carte en quelque
chose de plus intuitif plutôt qu'avoir un arc-en-ciel.

"""
plt.imshow(lena, cmap='gray')
plt.show()

"""
Graphes de surfaces 3D

Il est aussi possible d'afficher des graphes en 3D sous forme de
surfaces tridimensionnelles. On utilise la fonction "meshgrid" qui nous
génère  nos points en X et en Y.
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

