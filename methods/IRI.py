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

    def IRI_onedirection(now_img, times):
        R = now_img[:, :, 0].astype(float)
        B = now_img[:, :, 2].astype(float)
        G = now_img[:, :, 1].astype(float)
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
            t = 4
            new_GR = guide_filter(GR_Mask, R, GR, 2*t+1, 4*t+1); new_GR[1::2, :] = 0
            new_GB = guide_filter(GB_Mask, B, GB, 2*t+1, 4*t+1); new_GB[0::2, :] = 0

            new_R = guide_filter(R_Mask, GR, R, 2*t+1, 4*t+1); new_R[1::2, :] = 0
            new_B = guide_filter(B_Mask, GB, B, 2*t+1, 4*t+1); new_B[0::2, :] = 0

            # Step 3: Linear Interploation in Residual
            delta_R = (new_R - R) * R_Mask
            delta_B = (new_B - B) * B_Mask
            delta_GR = (new_GR - GR) * GR_Mask
            delta_GB = (new_GB - GB) * GB_Mask

            delta_R = convolve1d(delta_R, [1, 2, 1]) / 2
            delta_B = convolve1d(delta_B, [1, 2, 1]) / 2
            delta_GR = convolve1d(delta_GR, [1, 2, 1]) / 2
            delta_GB = convolve1d(delta_GB, [1, 2, 1]) / 2

            delta_G = delta_GR + delta_GB

            # Step 4: Refine images
            R = new_R - delta_R
            B = new_B - delta_B
            GR = new_GR - delta_GR
            GB = new_GB - delta_GB

            # Step 5: Check MAD(Mean absolute difference)
            mad = np.sum(np.abs(GR + GB - old_G)) / GR.size
            # print(mad)
            if mad < 1.0:
                break

        gaussian_filter = [4, 9, 15, 23, 26, 23, 15, 9, 4]
        w = np.abs(convolve1d(delta_G, [-1, 0, 1]))
        w = convolve1d(w, gaussian_filter) / np.sum(gaussian_filter)
        w = w + 1e-10

        new_img = np.dstack((R, GR+GB, B))
        return 1/w, new_img

    # Horizontal
    wh, img1 = IRI_onedirection(img, 3)
    # Vertical
    img2 = np.transpose(img, (1, 0, 2))
    wv, img2 = IRI_onedirection(img2, 3)
    img2 = np.transpose(img2, (1, 0, 2))
    wv = np.transpose(wv, (1, 0))

    # print(metrics(img1.astype(np.uint8), src_img))
    # print(metrics(img2.astype(np.uint8), src_img))

    # Get G
    new_G = ((wh * img1[:, :, 1] + wv * img2[:, :, 1]) / (wh + wv)).clip(0, 255)

    # predict R and B by RI, so just call it :)
    return RI.run(img, ok_G=new_G)
