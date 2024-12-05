import cmath as cm

def racine(x):
    """
    For example, racine(-4) will return 0+2i

    Parameters
    ----------
    x : the number to compute the square root
    Returns
    -------
    r : the square root of x
    """
    
    r = cm.sqrt(x)
    return r


def fact(number):
    """
    Function to compute factorial of a number

    Parameters
    ----------
    number : the number to compute the factorial operator on
    Returns
    -------
    response : factorial of number
    """

    response = 1

    for i in range(number, 1, -1):
        response *= i

    return response

def fact_recursif(number):
    """
    Function to compute factorial of a number recursively

    Parameters
    ----------
    number : the number to compute the factorial operator on
    Returns
    -------
    response : factorial of number
    """

    if number < 2:
        response = 1
    else:
        response = number * fact_recursif(number -1)

    return response