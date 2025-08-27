from pathlib import Path
import numpy as np

def read_data(filename):
    fname_full = Path(__file__).parent.joinpath('data').joinpath(filename)
    return np.load(fname_full)
