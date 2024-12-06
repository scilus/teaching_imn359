"""
5 - Fonctions personnalisées

Qui dit langage de programmation dit fonctions personnalisées.

On définit une fonction comme suit:

def maFonction(in1, in2, ...):
    # corps de la fonction
    return out1, out2, ...

La fonction peut être placée dans un fichier ".py" qui doit
être importée dans le fichier principal.

Une fonction n'a pas besoin de valeur de retour, il suffit simplement
d'assigner les valeurs souhaitées au variables dans le vecteur de
retour.

Toutes les variables définies dans la fonction ont une portée se
limitant à cette dernière.

"""

from Fonctions import HelloWorld, Retourner42

HelloWorld()

value_f = Retourner42(False)
value_t = Retourner42(True)

print('Retour de la fonction Retourner42(False):', value_f)
print('Retour de la fonction Retourner42(True):', value_t)

"""
Structures de contrôle logique
==============================

Python possède des structures de contrôles logique propre à ce
qu'on retrouve habituellement dans un langage de programmation. La
syntaxe change légèrement par contre par rapport au C++

IF :

    if condition:
        commandes


ELSE :

    if condition:
        commandes1
    else:
        commandes2


ELSEIF:

    if condition1:
        commandes1
    elif condition2:
        commandes2
    else:
        commandes3

Python peut aussi faire des boucles "for". Dans ce cas-ci, il n'y a pas
de variable incrémentée explicitement à chaque fois, on passe plutôt un
vecteur à la boucle for avec range(start, stop, step) :

    for n in range(100):
        commandes

    for n in range(3, 6):
        commandes


Finalement, python supporte aussi la structure de contrôle "WHILE".

    while condition:
        commandes

Opérateurs relationels et logique
    Comme tout langage de programmation, python possède aussi des
    opérateurs pour évaluer les relations entre les valeurs. Ces opérateurs
    retournent False (faux) ou True (vrai).


Opérateurs relationels

    == égal
    != non égal
    >  plus grand que
    <  plus petit que
    >= plus grand ou égal
    <= plus petit ou égal


Les opérateurs logiques utilisent ce fonctionnement pour déterminer
si une expression booléene est vraie ou fausse.

Opérateurs logiques

    si A ET B -> if A and B:
    si A ou B -> if A or B:
    si A dans T -> if A in T:
    si A n'est pas dans T -> if A not in T:

"""
