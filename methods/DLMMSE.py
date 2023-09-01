import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d

def run(img):
    new_img = np.copy(img).astype(float)

    R = new_img[:, :, 0]; G = new_img[:, :, 1]; B = new_img[:, :, 2]
    S = R + G + B

    # Step 1: Interploate G
    # Step 1.1: Interploate by a simple method, then compute G-R and G-B (2-2 ~ 2-6 in paper)
    H = convolve1d(S, [-1, 2, 2, 2, -1]) / 4
    V = convolve1d(S.T, [-1, 2, 2, 2, -1]).T / 4

    # H: G R G R ...; B G B G ...
    # S: R G R G ...; G B G B ...
    # We need: G-R, G-R ..; G-B, G-B
    delta_H = H - S; delta_H[0::2, 1::2] = -delta_H[0::2, 1::2]; delta_H[1::2, 0::2] = -delta_H[1::2, 0::2]
    delta_V = V - S; delta_V[0::2, 1::2] = -delta_V[0::2, 1::2]; delta_V[1::2, 0::2] = -delta_V[1::2, 0::2] 

    # Step 1.2: Use a low-pass filter (3-4)
    # TODO: How to change this gaussian smooth filter adaptively?
    # sigma=2
    gaussian_filter = [4, 9, 15, 23, 26, 23, 15, 9, 4]
    gaussian_H = convolve1d(delta_H, gaussian_filter) / np.sum(gaussian_filter)
    gaussian_V = convolve1d(delta_V.T, gaussian_filter).T / np.sum(gaussian_filter)

    # Step 1.3: Calucate mean_x, var_x, var_v (2-12, 3-6 ~ 3-10)
    # 2L+1, L=4
    mean_filter = [1 for _ in range(2*4+1)]
    mean_H = convolve1d(gaussian_H, mean_filter) / np.sum(mean_filter)
    mean_V = convolve1d(gaussian_V.T, mean_filter).T / np.sum(mean_filter)

    var_value_H = convolve1d(np.square(gaussian_H - mean_H), mean_filter) / np.sum(mean_filter) + 1e-10
    var_value_V = convolve1d(np.square(gaussian_V - mean_V).T, mean_filter).T / np.sum(mean_filter) + 1e-10

    var_noise_H = convolve1d(np.square(gaussian_H - delta_H), mean_filter) / np.sum(mean_filter) + 1e-10
    var_noise_V = convolve1d(np.square(gaussian_V - delta_V).T, mean_filter).T / np.sum(mean_filter) + 1e-10

    # Step 1.4: make delta more precise by 2-12 in paper
    new_H = mean_H + var_value_H / (var_noise_H + var_value_H) * (delta_H - mean_H)
    new_V = mean_V + var_value_V / (var_noise_V + var_value_V) * (delta_V - mean_V)

    # Step 1.5: combine delta of two direcitons to make more precise by 3-11 and 4-7 in paper
    # TODO: Sometimes var is negative, which cannot happen in reality. 
    # I use abs and it works well. But is it OK?
    var_x_H = np.abs(var_value_H - var_value_H / (var_value_H + var_noise_H)) + 1e-10
    var_x_V = np.abs(var_value_V - var_value_V / (var_value_V + var_noise_V)) + 1e-10

    w_H = var_x_V / (var_x_H + var_x_V)
    w_V = var_x_H / (var_x_H + var_x_V)
    final_result = w_H * new_H + w_V * new_V

    # Step 1.6: add delta, ok~
    new_img[0::2, 0::2, 1] = (R + final_result)[0::2, 0::2]
    new_img[1::2, 1::2, 1] = (B + final_result)[1::2, 1::2]

    # Step 2: Interploate R and B
    G = new_img[:, :, 1]
    
    # Step 2.1: R in B or B in R (Figure.6)
    # we can use G-S to get both R and B. For clarity, we dont do that.
    kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    delta_GR = convolve2d(G-R, kernel, mode='same') / 4 
    delta_GB = convolve2d(G-B, kernel, mode='same') / 4
    new_img[1::2, 1::2, 0] = (G - delta_GR)[1::2, 1::2]
    new_img[0::2, 0::2, 2] = (G - delta_GB)[0::2, 0::2]

    # Step 2.2: R/B in G
    R = new_img[:, :, 0]; B = new_img[:, :, 2]
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    delta_GR = convolve2d(G-R, kernel, mode='same') / 4 
    delta_GB = convolve2d(G-B, kernel, mode='same') / 4

    new_img[0::2, 1::2, 0] = (G - delta_GR)[0::2, 1::2]
    new_img[1::2, 0::2, 0] = (G - delta_GR)[1::2, 0::2]
    new_img[0::2, 1::2, 2] = (G - delta_GB)[0::2, 1::2]
    new_img[1::2, 0::2, 2] = (G - delta_GB)[1::2, 0::2]

    return (new_img + 0.5).clip(0, 255.5).astype(np.uint8)
