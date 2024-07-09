import numpy as np
import cv2


def rotate_90(image):
    return flip_hor(transpose(image))

def rotate_270(image):
    return flip_vert(transpose(image))

def rotate_180(image):
    return flip_hor(flip_vert(image))

def transpose(image):
    height,width,channel = image.shape
    transposed = np.zeros((width,height,channel), dtype=image.dtype)
    for i in range(height):
        for j in range(width):
            transposed[j,i] = image[i,j]

    return transposed


def flip_hor(image):
    height,width,channel = image.shape
    flipped = np.zeros(image.shape,dtype=image.dtype)
    for i in range(height):
        for j in range(width):
            flipped[i,j] = image[i,width-j-1]

    return flipped

def flip_vert(image):
    height,width,channel = image.shape
    flipped = np.zeros(image.shape,dtype=image.dtype)
    for i in range(height):
        for j in range(width):
            flipped[i,j] = image[height-i-1,j]

    return flipped