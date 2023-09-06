import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d

from . import RI

def run(img):
    def guide_filter(M, I, p, h, v, eps=0.01):
        sum_kernel = np.ones((h, v))

        N1 = convolve2d(M, sum_kernel, mode='same')
        N2 = convolve2d(np.ones(I.shape), sum_kernel, mode='same')

        ok_I = I * M; ok_p = p * M
        mean_I = convolve2d(ok_I, sum_kernel, mode='same') / N1
        mean_p = convolve2d(ok_p, sum_kernel, mode='same') / N1
        corr_I = convolve2d(ok_I*ok_I, sum_kernel, mode='same') / N1
        corr_Ip= convolve2d(ok_I*ok_p, sum_kernel, mode='same') / N1

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = convolve2d(a, sum_kernel, mode='same') / N2
        mean_b = convolve2d(b, sum_kernel, mode='same') / N2

        return mean_a * I + mean_b

    def IRI_onedirection(R, G, B, times):
        GR = np.copy(G); GR[1::2, :] = 0
        GB = np.copy(G); GB[0::2, :] = 0

        R_Mask = np.zeros(R.shape[0:2]); R_Mask[0::2, 0::2] = 1
        B_Mask = np.zeros(B.shape[0:2]); B_Mask[1::2, 1::2] = 1
        GR_Mask = np.zeros(GR.shape[0:2]); GR_Mask[0::2, 1::2] = 1
        GB_Mask = np.zeros(GB.shape[0:2]); GB_Mask[1::2, 0::2] = 1

        # Step 1: Linear Interploation in Value
        R = convolve1d(R, [1, 2, 1]) / 2
        B = convolve1d(B, [1, 2, 1]) / 2
        GR = convolve1d(GR, [1, 2, 1]) / 2
        GB = convolve1d(GB, [1, 2, 1]) / 2

        delta_G = None
        for i in range(0, times):
            # save old value
            old_G = GR + GB

            # Step 2: Use guide filter to predict 
            t = 3
            new_GR = guide_filter(GR_Mask, R, GR, 2*t+1, 4*t+1); new_GR[1::2, :] = 0
            new_GB = guide_filter(GB_Mask, B, GB, 2*t+1, 4*t+1); new_GB[0::2, :] = 0
            new_R = guide_filter(R_Mask, GR, R, 2*t+1, 4*t+1); new_R[1::2, :] = 0
            new_B = guide_filter(B_Mask, GB, B, 2*t+1, 4*t+1); new_B[0::2, :] = 0

            # Step 3: Linear Interploation in Residual
            delta_R = (new_R - R) * R_Mask
            delta_B = (new_B - B) * B_Mask
            delta_GR = (new_GR - GR) * GR_Mask
            delta_GB = (new_GB - GB) * GB_Mask

            delta_R = convolve1d((new_R - R) * R_Mask, [1, 2, 1]) / 2
            delta_B = convolve1d((new_B - B) * B_Mask, [1, 2, 1]) / 2
            delta_GR = convolve1d((new_GR - GR) * GR_Mask, [1, 2, 1]) / 2
            delta_GB = convolve1d((new_GB - GB) * GB_Mask, [1, 2, 1]) / 2

            delta_G = delta_GR + delta_GB

            # Step 4: Refine images
            R = new_R - delta_R
            B = new_B - delta_B
            GR = new_GR - delta_GR
            GB = new_GB - delta_GB

            # Step 5: Check MAD(Mean absolute difference)
            mad = np.sum(np.abs(GR + GB - old_G)) / GR.size
            # print(mad)
            if mad < 0.5:
                break

        gaussian_filter = [4, 9, 15, 23, 26, 23, 15, 9, 4]
        var = np.abs(convolve1d(delta_G, [-1, 0, 1]))
        var = convolve1d(var, gaussian_filter) / np.sum(gaussian_filter)
        w = var * var + 1e-10

        return 1/w, GR+GB

    # Horizontal
    R = img[:, :, 0].astype(float)
    B = img[:, :, 2].astype(float)
    G = img[:, :, 1].astype(float)
    wh, Gh = IRI_onedirection(R, G, B, 4)
    wv, Gv = IRI_onedirection(R.T, G.T, B.T, 4)
    wv = wv.T; Gv = Gv.T

    # generate new G
    new_G = ((wh * Gh + wv * Gv) / (wh + wv)).clip(0, 255)

    # predict R and B by RI, so just call it :)
    return RI.run(img, ok_G=new_G)
