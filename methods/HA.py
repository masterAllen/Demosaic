import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d

def run(img):
    new_img = np.copy(img).astype(float)

    R = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    B = img[:, :, 2].astype(float)

    S = R + G + B
    # Step 1: Interplote G in R/B
    # 0 means 0', 90 means 90'
    S0 = convolve1d(S, [-1, 2, 2, 2, -1]) / 4
    S90 = convolve1d(S.T, [-1, 2, 2, 2, -1]).T / 4

    # g means gradient
    g0 = np.abs(convolve1d(S, [-1, 0, 1])) + np.abs(convolve1d(S, [-1, 0, 2, 0, -1]))
    g90 = np.abs(convolve1d(S.T, [-1, 0, 1]).T) + np.abs(convolve1d(S.T, [-1, 0, 2, 0, -1]).T)

    new_img[:, :, 1] = (S0 + S90) / 2
    new_img[g0 < g90, 1] = S0[g0 < g90]
    new_img[g0 > g90, 1] = S90[g0 > g90]

    new_img[0::2, 1::2, 1] = G[0::2, 1::2]
    new_img[1::2, 0::2, 1] = G[1::2, 0::2]

    # Step 2: Interplote R in B, Interplote B in R
    G = new_img[:, :, 1]

    I1 = np.zeros((3, 3)); I1[0, 2] = I1[2, 0] = 1
    I2 = np.zeros((3, 3)); I2[0, 2] = I2[2, 0] = -1; I2[1, 1] = 2
    S45 = convolve2d(S, I1, mode='same') / 2 + convolve2d(G, I2, mode='same') / 4

    I1 = np.zeros((3, 3)); I1[0, 0] = I1[2, 2] = 1
    I2 = np.zeros((3, 3)); I2[0, 0] = I2[2, 2] = -1; I2[1, 1] = 2
    S135 = convolve2d(S, I1, mode='same') / 2 + convolve2d(G, I2, mode='same') / 4
    
    I1 = np.zeros((3, 3)); I1[0, 2] = I1[2, 0] = 1
    I2 = np.zeros((3, 3)); I2[0, 2] = I2[2, 0] = -1; I2[1, 1] = 2
    g45 = np.abs(convolve2d(S, I1, mode='same')) + np.abs(convolve2d(G, I2, mode='same'))

    I1 = np.zeros((3, 3)); I1[0, 0] = I1[2, 2] = 1
    I2 = np.zeros((3, 3)); I2[0, 0] = I2[2, 2] = -1; I2[1, 1] = 2
    g135 = np.abs(convolve2d(S, I1, mode='same')) + np.abs(convolve2d(G, I2, mode='same'))

    new_img[:, :, 0] = (S45 + S135) / 2
    new_img[g45 < g135, 0] = S45[g45 < g135]
    new_img[g45 > g135, 0] = S135[g45 > g135]
    new_img[0::2, 0::2, 0] = R[0::2, 0::2]

    new_img[:, :, 2] = (S45 + S135) / 2
    new_img[g45 < g135, 2] = S45[g45 < g135]
    new_img[g45 > g135, 2] = S135[g45 > g135]
    new_img[1::2, 1::2, 2] = B[1::2, 1::2]

    # Step 3: Interplote R in G, Interplote B in G
    new_img[0::2, 1::2, 0] = S0[0::2, 1::2]
    new_img[1::2, 0::2, 0] = S90[1::2, 0::2]

    new_img[1::2, 0::2, 2] = S0[1::2, 0::2]
    new_img[0::2, 1::2, 2] = S90[0::2, 1::2]

    return new_img.clip(0, 255).astype(np.uint8)
