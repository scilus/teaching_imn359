import numpy as np

"""
5- Fonctions personnalisées

Python possède aussi son propre langage de programmation. Qui dit
langage de programmation dit donc fonctions personnalisées.

La signature de fonction se donne telle que :

def [out1,out2,out3,etc] = funName(in1,in2)

La fonction peut être placée dans un fichier ".py" qui doit
être importée du fichier.

Une fonction n'a pas besoin de valeur de retour, il suffit simplement
d'assigner les valeurs souhaitées au variables dans le vecteur de
retour.

Toutes les variables définies dans la fonction ont une portée ne
dépassant pas cette dernière.

"""

from HelloWorldFunction import HelloWorldFunction
from ReturnFordyTou import ReturnFordyTou

HelloWorldFunction()

value_f = ReturnFordyTou(False)
value_t = ReturnFordyTou(True)

print('Retour de la fonction ReturnFordyTou(False) \n', value_f, '\n')
print('Retour de la fonction ReturnFordyTou(True) \n', value_t, '\n')

"""
Structures de contrôle logique

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

Opérateurs relationels
    Comme tout langage de programmation, python possède aussi des
    opérateurs pour évaluer les relations entre les valeurs. Ces opérateurs
    retournent zéro (faux) ou une valeur non zéro (vrai).


    Opérateurs relationels : 

    == égal
    != non égal
    >  plus grand que
    <  plus petit que
    >= plus grand ou égal
    <= plus petit ou égal


    Les opérateurs logiques utilisent ce fonctionnement pour
    déterminer si oui ou non une expression booléene est vraie ou fausse.

    Opérateurs logiques et binaires
    and    & 
    or     |
    in not ~

"""
