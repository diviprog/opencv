import numpy as np

def rotate(matrix):
    return np.rot90(matrix, -1)


def transpose(matrix):
    return np.transpose(matrix)