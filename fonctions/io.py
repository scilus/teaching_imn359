from importlib.resources import files
import numpy as np

def read_data(filename):
    fname_full = files('fonctions.data').joinpath(filename)
    return np.load(fname_full)
