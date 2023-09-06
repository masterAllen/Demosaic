import numpy as np
from scipy import signal, ndimage
from . import GBTF

def run(img, ok_G=None):
    def guide_filter(M, I, p, h, v, eps=0.01):
        sum_kernel = np.ones((h, v))

        N1 = signal.convolve2d(M, sum_kernel, mode='same')
        N2 = signal.convolve2d(np.ones(I.shape), sum_kernel, mode='same')

        nI = I * M
        mean_I = signal.convolve2d(nI, sum_kernel, mode='same') / N1
        mean_p = signal.convolve2d(p, sum_kernel, mode='same') / N1
        corr_I = signal.convolve2d(nI*nI, sum_kernel, mode='same') / N1
        corr_Ip = signal.convolve2d(nI*p, sum_kernel, mode='same') / N1

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = signal.convolve2d(a, sum_kernel, mode='same') / N2
        mean_b = signal.convolve2d(b, sum_kernel, mode='same') / N2

        return mean_a * I + mean_b

    # Step 1: Use GBTF to infer G
    if ok_G is None:
        ok_G = GBTF.run(img)[:, :, 1].astype(float)
        ok_G = (ok_G + 0.5).clip(0, 255)

    R = img[:, :, 0].astype(float)
    B = img[:, :, 2].astype(float)

    # Step 2: Use guide filter to predict R or B
    R_Mask = np.zeros(ok_G.shape)
    R_Mask[0::2, 0::2] = 1
    new_R = guide_filter(R_Mask, ok_G, R, 11, 11)

    B_Mask = np.zeros(ok_G.shape)
    B_Mask[1::2, 1::2] = 1
    new_B = guide_filter(B_Mask, ok_G, B, 11, 11)

    # Step 3: Calculate delta of the positions which had data before
    delta_R = (R - new_R) * R_Mask
    delta_B = (B - new_B) * B_Mask

    # Step 4: Interploate the RI
    intp = np.array([[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4]])
    delta_R = signal.convolve2d(delta_R, intp, mode='same') / signal.convolve2d(R_Mask, intp, mode='same')
    delta_B = signal.convolve2d(delta_B, intp, mode='same') / signal.convolve2d(B_Mask, intp, mode='same')

    # Step 5: Add, done, nice job!
    new_R = new_R + delta_R
    new_B = new_B + delta_B

    # Actually, this must be true
    # new_R[0::2, 0::2] = R[0::2, 0::2]
    # new_B[1::2, 1::2] = B[1::2, 1::2]

    new_R = (new_R + 0.5).clip(0, 255)
    new_B = (new_B + 0.5).clip(0, 255)

    new_img = np.dstack((new_R, ok_G, new_B))
    return new_img.astype(np.uint8)
