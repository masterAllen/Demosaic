import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

from . import RI

def run(img):
    def guide_filter_MLRI(M, I, p, h, v, eps=0.01):
        ok_I = I * M; ok_p = p * M

        sum_kernel = np.ones((h, v))
        N1 = convolve2d(M, sum_kernel, mode='same')

        lap_I = convolve1d(ok_I, [-1, 0, 2, 0, -1]) * M
        lap_p = convolve1d(ok_p, [-1, 0, 2, 0, -1]) * M

        # make the derivate of eq.6 and eq.7 to zero, get a and b
        sum_lap_II = convolve2d(lap_I * lap_I, sum_kernel, mode='same')
        sum_lap_Ip = convolve2d(lap_I * lap_p, sum_kernel, mode='same')
        a = sum_lap_Ip / (sum_lap_II + 1e-32)

        sum_I = convolve2d(ok_I, sum_kernel, mode='same')
        sum_p = convolve2d(ok_p, sum_kernel, mode='same')
        b = (sum_p - a * sum_I) / N1

        # mean_a = convolve2d(a*M, sum_kernel, mode='same') / N1
        # mean_b = convolve2d(b*M, sum_kernel, mode='same') / N1
        # return mean_a * I + mean_b

        # calculate W in eq.10
        W = ok_p - a * ok_I - b
        W = np.square(convolve2d(W, sum_kernel, mode='same')) / N1
        W = 1 / (W + 1e-32)

        sum_W = convolve2d(W, sum_kernel, mode='same')
        mean_a = convolve2d(W * a, sum_kernel, mode='same') / sum_W
        mean_b = convolve2d(W * b, sum_kernel, mode='same') / sum_W

        return mean_a * I + mean_b
        # return a * I + b

    def compute_delta(R, G, B):
        R_Mask = np.zeros(R.shape); R_Mask[0::2, 0::2] = 1
        B_Mask = np.zeros(B.shape); B_Mask[1::2, 1::2] = 1

        GR = np.zeros(G.shape); GR[0::2, :] = G[0::2, :]
        GB = np.zeros(G.shape); GB[1::2, :] = G[1::2, :]
        GR_Mask = np.zeros(G.shape); GR_Mask[0::2, 1::2] = 1
        GB_Mask = np.zeros(G.shape); GB_Mask[1::2, 0::2] = 1

        # Step 1: Interploate by linear
        intp_R = convolve1d(R, [1, 2, 1]) / 2
        intp_G = convolve1d(G, [1, 2, 1]) / 2
        intp_B = convolve1d(B, [1, 2, 1]) / 2

        # Step 2: Use guide filter
        new_R = guide_filter_MLRI(R_Mask, intp_G, R, 7, 7)
        delta_R = convolve1d(new_R*R_Mask - R, [1, 2, 1]) / 2
        new_R = new_R - delta_R

        new_GR = guide_filter_MLRI(GR_Mask, intp_R, GR, 7, 7)
        delta_GR = convolve1d(new_GR*GR_Mask - GR, [1, 2, 1]) / 2
        new_GR = new_GR - delta_GR

        new_B = guide_filter_MLRI(B_Mask, intp_G, B, 7, 7)
        delta_B = convolve1d(new_B*B_Mask - B, [1, 2, 1]) / 2
        new_B = new_B - delta_B

        new_GB = guide_filter_MLRI(GB_Mask, intp_B, GB, 7, 7)
        delta_GB = convolve1d(new_GB*GB_Mask - GB, [1, 2, 1]) / 2
        new_GB = new_GB - delta_GB

        # Step 3: calculate R-G
        delta_GR = (new_GR - new_R); delta_GR[1::2, :] = 0
        delta_GB = (new_GB - new_B); delta_GB[0::2, :] = 0

        return (delta_GR, delta_GB)

    R = img[:, :, 0].astype(float)
    B = img[:, :, 2].astype(float)
    G = img[:, :, 1].astype(float)

    new_R = np.copy(R); new_B = np.copy(B); new_G = np.copy(G)

    # Step 1-3: Compute (G-R), (G-B) in Horizontal and Vertical (eq.6-10, eq.14 in paper)
    delta_GR_H, delta_GB_H = compute_delta(R, G, B)
    delta_GR_V, delta_GB_V = compute_delta(R.T, G.T, B.T)
    delta_GR_V = delta_GR_V.T; delta_GB_V = delta_GB_V.T

    # Step 4: Calcuate W in GBTF
    D_H = np.absolute(convolve1d((delta_GR_H + delta_GB_H), [-1, 0, 1]))
    D_V = np.absolute(convolve1d((delta_GR_V + delta_GB_V).T, [-1, 0, 1])).T

    now_kernel = np.zeros((9, 9)); now_kernel[2:7, 0:5] = 1
    W_W = convolve2d(D_H, now_kernel, mode='same')
    now_kernel = np.zeros((9, 9)); now_kernel[2:7, 4:9] = 1
    W_E = convolve2d(D_H, now_kernel, mode='same')
    now_kernel = np.zeros((9, 9)); now_kernel[0:5, 2:7] = 1
    W_N = convolve2d(D_V, now_kernel, mode='same')
    now_kernel = np.zeros((9, 9)); now_kernel[4:9, 2:7] = 1
    W_S = convolve2d(D_V, now_kernel, mode='same')

    W_W += 1e-10; W_E += 1e-10; W_N += 1e-10; W_S += 1e-10
    W_W = 1 / np.square(W_W); W_E = 1 / np.square(W_E);
    W_N = 1 / np.square(W_N); W_S = 1 / np.square(W_S);
    W_T = W_W + W_E + W_N + W_S

    # Step 5: Compute delta finally(eq.15 in paper)
    # TODO: I find f used in paper is worse than the orignal.
    # Is it because there is something wrong with my realization?
    # f = [0.01, 0.08, 0.35, 0.56]
    f = np.ones(4) / 4

    # Important: convolve is inverse! [a1, a2, a3] * [b1, b2, b3] = a1*b3 + a2*b2 + a3*b1
    # If f0*a0+..+f3*a3+0*a4+..+0*a6, then convolve [0,..,f3,..,f0]
    now_kernel = np.zeros(7); now_kernel[3:7] = f[::-1]
    V1 = convolve1d(delta_GR_V.T, now_kernel).T
    now_kernel = np.zeros(7); now_kernel[0:4] = f
    V2 = convolve1d(delta_GR_V.T, now_kernel).T
    now_kernel = np.zeros(7); now_kernel[3:7] = f[::-1]
    V3 = convolve1d(delta_GR_H, now_kernel)
    now_kernel = np.zeros(7); now_kernel[0:4] = f
    V4 = convolve1d(delta_GR_H, now_kernel)
    delta_GR = (V1*W_N + V2*W_S + V3*W_W + V4*W_E) / W_T

    now_kernel = np.zeros(7); now_kernel[3:7] = f[::-1]
    V1 = convolve1d(delta_GB_V.T, now_kernel).T
    now_kernel = np.zeros(7); now_kernel[0:4] = f
    V2 = convolve1d(delta_GB_V.T, now_kernel).T
    now_kernel = np.zeros(7); now_kernel[3:7] = f[::-1]
    V3 = convolve1d(delta_GB_H, now_kernel)
    now_kernel = np.zeros(7); now_kernel[0:4] = f
    V4 = convolve1d(delta_GB_H, now_kernel)
    delta_GB = (V1*W_N + V2*W_S + V3*W_W + V4*W_E) / W_T

    # Step 6: Recover G
    new_G[0::2, 0::2] = R[0::2, 0::2] + delta_GR[0::2, 0::2]
    new_G[1::2, 1::2] = B[1::2, 1::2] + delta_GB[1::2, 1::2]
    new_G = (new_G + 0.5).clip(0, 255)

    # Step 7: Recover R and B using RI
    return RI.run(img, ok_G=new_G)
