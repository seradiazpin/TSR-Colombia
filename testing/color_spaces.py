from __future__ import division
import numpy as np

def RGB2III(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    width, height = img.shape[:2]

    I1 = (1 / 3) * (np.float32(red) + np.float32(green) + np.float32(blue))
    I2 = (1 / 2) * (np.float32(red) - np.float32(blue))
    I3 = (1 / 4) * (2 * np.float32(green) - np.float32(red) - np.float32(blue))
    return I1, I2, I3


def III2RGB(I1, I2, I3, img):
    width, height = img.shape[:2]

    R = I1 + I2 - (2 / 3) * I3
    G = I1 + (4 / 3) * I3
    B = I1 - I2 - (2 / 3) * I3
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 1]
    return R, G, B

def RGB2LUX(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    #width, height = img.shape[:2]

    L = (((red + 1) ** 0.3) * ((green + 1) ** 0.6) * ((blue + 1) ** 0.6)) - 1
    if ((L >= red).all()):
        U = 128 * ((red + 1) / (L + 1))
    else:
        U = 256 - 128 * ((L + 1) / (red + 1))
    if ((L >= blue).all()):
        X = 128 * ((blue + 1) / (L + 1))
    else:
        X = 256 - 128 * ((L + 1) / (blue + 1))
    return L, U, X


def LUX2RGB(L, U, X):
    if not U.all() < 128:
        R = ((U * (L + 1)) / 128) - 1
    else:
        R = ((128) * (L + 1) / (256 - U)) - 1
    if (X.all() < 128):
        B = ((X * (L + 1)) / 128) - 1
    else:
        B = ((128) * (L + 1) / (256 - X)) - 1
    G = ((L + 1) / ((R + 1) ** (0.3) * (B + 1) ** (0.6))) ** (5 / 3) - 1
    return R, G, B
