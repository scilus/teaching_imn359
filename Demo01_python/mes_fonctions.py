import numpy as np

def racine(x):
    """
    Implementation de la racine carrée supportant
    les valeurs négatives.

    Parametres
    ----------
    x : le nombre dont on veut connaître la racine carrée

    Retour
    -------
    r : la racine carrée de x
    """

    r = np.emath.sqrt(x)
    return r


def fact(number):
    """
    Fonction pour calculer la factorielle d'un nombre

    Parameters
    ----------
    number : Nombre à évaluer

    Returns
    -------
    response : Factorielle du nombre
    """

    response = 1

    for i in range(number, 1, -1):
        response *= i

    return response


def fact_recursif(number):
    """
    Fonction factorielle, implémentation récursive.

    Parameters
    ----------
    number : Nombre à évaluer

    Returns
    -------
    response : Factorielle du nombre
    """

    if number < 2:
        response = 1
    else:
        response = number * fact_recursif(number -1)

    return response
