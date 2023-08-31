import numpy as np
from scipy import signal, ndimage

def run(img):
    new_img = np.zeros(img.shape).astype(float)

    S = (img[:,:,0] + img[:,:,1] + img[:,:,2]).astype(float)

    # Predict G: Horizontal and Vertical
    S_H = ndimage.convolve1d(S, [-1, 2, 2, 2, -1]) / 4
    S_V = ndimage.convolve1d(S.T, [-1, 2, 2, 2, -1]).T / 4

    Grad_H1 = ndimage.convolve1d(S, [-1, 0, 1])
    Grad_H2 = ndimage.convolve1d(S, [-1, 0, 2, 0, -1])
    Grad_H = np.abs(Grad_H1) + np.abs(Grad_H2)

    Grad_V1 = ndimage.convolve1d(S.T, [-1, 0, 1]).T
    Grad_V2 = ndimage.convolve1d(S.T, [-1, 0, 2, 0, -1]).T
    Grad_V = np.abs(Grad_V1) + np.abs(Grad_V2)

    new_img[:, :, 1] = (S_H + S_V) / 2
    new_img[Grad_H < Grad_V, 1] = S_H[Grad_H < Grad_V]
    new_img[Grad_H > Grad_V, 1] = S_V[Grad_H > Grad_V]
    new_img[0::2, 0::2, 1] = img[0::2, 0::2, 1]
    new_img[1::2, 1::2, 1] = img[1::2, 1::2, 1]

    # # Predict R in B (or B in R)
    # I = np.zeros(5, 5)
    # I[0, 0] = -1, I[1, 1] = 2, I[2, 2] = 2, I[3, 3] = 2, I[4, 4] = -1
    # S_45 = signal.convolve2d(S, I, mode='same') / 4

    # I = np.zeros(5, 5)
    # I[0, 4] = -1, I[1, 3] = 2, I[2, 2] = 2, I[3, 1] = 2, I[4, 0] = -1
    # S_135 = signal.convolve2d(S, I, mode='same') / 4

    # I = np.zeros(5, 5), I[0, 0] = -1, I[2, 2] = 2, I[4, 4] = -1
    # Grad_45_1 = signal.convolve2d(S, I, mode='same')
    # I = np.zeros(5, 5), I[1, 1] = -1, I[3, 3] = 1
    # Grad_45_2 = signal.convolve2d(S, I, mode='same')
    # Grad_45 = np.abs(Grad_45_1) + np.abs(Grad_45_2)

    # I = np.zeros(5, 5), I[0, 4] = -1, I[2, 2] = 2, I[4, 0] = -1
    # Grad_135_1 = signal.convolve2d(S, I, mode='same')
    # I = np.zeros(5, 5), I[1, 3] = -1, I[3, 1] = 1
    # Grad_135_2 = signal.convolve2d(S, I, mode='same')
    # Grad_135 = np.abs(Grad_135_1) + np.abs(Grad_135_2)

    # new_img[:, :, 0] = (S_45 + S_135) / 2
    # new_img[Grad_45 < Grad_135, 0] = S_45[Grad_45 < Grad_135]
    # new_img[Grad_45 > Grad_135, 0] = S_135[Grad_45 > Grad_135]
    # new_img[0::2, 1::2, 0] = img[0::2, 0::2, 1]


    # new_img = (new_img + 0.5).clip(0, 255.5)
    return new_img.astype(np.uint8)
