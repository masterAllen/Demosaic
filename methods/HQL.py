import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d

# alpha * delta_R, beta * delta_G, gama * delta_B, alpha = 1/2, beta = 5/8, gama=3/4
# Then it can be converted to convolution
def run(img):
    new_img = np.copy(img).astype(float)

    R = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    B = img[:, :, 2].astype(float)
    S = R + G + B

    # 1. Predict G
    I = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]])
    new_img[:, :, 1] = convolve2d(S, I, mode='same') / 8
    new_img[0::2, 1::2, 1] = G[0::2, 1::2]
    new_img[1::2, 0::2, 1] = G[1::2, 0::2]

    # 2. Predict R and B
    I = np.array([[0, 0, 1, 0, 0], [0, -2, 0, -2, 0], [-2, 8, 10, 8, -2], [0, -2, 0, -2, 0], [0, 0, 1, 0, 0]])
    new_img[0::2, 1::2, 0] = (convolve2d(S, I, mode='same') / 16)[0::2, 1::2]
    I = np.array([[0, 0, -2, 0, 0], [0, -2, 8, -2, 0], [1, 0, 10, 0, 1], [0, -2, 8, -2, 0], [0, 0, -2, 0, 0]])
    new_img[1::2, 0::2, 0] = (convolve2d(S, I, mode='same') / 16)[1::2, 0::2]
    I = np.array([[0, 0, -3, 0, 0], [0, 4, 0, 4, 0], [-3, 0, 12, 0, -3], [0, 4, 0, 4, 0], [0, 0, -3, 0, 0]])
    new_img[1::2, 1::2, 0] = (convolve2d(S, I, mode='same') / 16)[1::2, 1::2]

    # 3. Predict B
    I = np.array([[0, 0, 1, 0, 0], [0, -2, 0, -2, 0], [-2, 8, 10, 8, -2], [0, -2, 0, -2, 0], [0, 0, 1, 0, 0]])
    new_img[1::2, 0::2, 2] = (convolve2d(S, I, mode='same') / 16)[1::2, 0::2]
    I = np.array([[0, 0, -2, 0, 0], [0, -2, 8, -2, 0], [1, 0, 10, 0, 1], [0, -2, 8, -2, 0], [0, 0, -2, 0, 0]])
    new_img[0::2, 1::2, 2] = (convolve2d(S, I, mode='same') / 16)[0::2, 1::2]
    I = np.array([[0, 0, -3, 0, 0], [0, 4, 0, 4, 0], [-3, 0, 12, 0, -3], [0, 4, 0, 4, 0], [0, 0, -3, 0, 0]])
    new_img[0::2, 0::2, 2] = (convolve2d(S, I, mode='same') / 16)[0::2, 0::2]

    return new_img.clip(0, 255).astype(np.uint8)
