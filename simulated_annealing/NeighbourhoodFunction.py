import numpy as np

def shift_all_randomly_normal(vector):

    rndm = np.random.normal(loc=0, scale=1, size=len(vector))
    vector = np.add(vector, rndm)

    return vector