import matplotlib.pyplot as plt
import numpy as np

# valeurs pour lesquelles on évalue nos fonctions
t = np.arange(-150, 150, 0.01)

# fonctions cos pour différentes fréquences évaluées aux
# valeurs dans t.
y1 = np.cos(t / 3)
y2 = np.cos(t / 4)

ys = np.cos(t / 3) + np.cos(t / 4)

# dessiner y1 en vert (g), y2 en rouge (r) et ys en bleu (b)
# avec une épaisseur de ligne de 2
plt.plot(t, y1, 'g', t, y2, 'r', t, ys, 'b', linewidth=2)

# l'axe des x s'appellera "Temps" et celui des y, "Amplitude"
plt.xlabel("Temps")
plt.ylabel("Amplitude")

# le nom de chaque courbe, dans le même ordre qu'à la ligne 16
plt.legend(['cos(t/3)','cos(t/4)','cos(t/3)+cos(t/4)'])

# le titre du graphique
plt.title('My cosines')

# on utilise savefig pour sauvegarder la figure dans un fichier
plt.savefig('demo02.png')

# plt.show() permet l'affiche interactif du graphique
plt.show()
