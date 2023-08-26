import numpy as np
from scipy import signal, ndimage

def run(img):
    S = img.astype(float)

    # RGB -> RBG
    S[:, :, [1, 2]] = S[:, :, [2, 1]]
    R, B, G = S[:,:,0], S[:,:,1], S[:,:,2]

    G_Final = np.copy(G)
    C_Final = [np.copy(R), np.copy(B)]

    conv_kernel_1 = [-1, 2, 2, 2, -1]
    H = 1/4 * ndimage.convolve1d((R+G+B), conv_kernel_1)
    G_H = np.copy(G)
    G_H[0::2, 0::2] = H[0::2, 0::2]
    G_H[1::2, 1::2] = H[1::2, 1::2]

    R_H = np.copy(R)
    R_H[0::2, 1::2] = H[0::2, 1::2]
    B_H = np.copy(B)
    B_H[1::2, 0::2] = H[1::2, 0::2]

    V = 1/4 * ndimage.convolve1d((R+G+B).T, conv_kernel_1).T
    G_V = np.copy(G)
    G_V[0::2, 0::2] = V[0::2, 0::2]
    G_V[1::2, 1::2] = V[1::2, 1::2]

    R_V = np.copy(R)
    R_V[1::2, 0::2] = V[1::2, 0::2]
    B_V = np.copy(B)
    B_V[0::2, 1::2] = V[0::2, 1::2]

    delta_H = G_H - R_H - B_H
    delta_V = G_V - R_V - B_V

    conv_kernel_2 = [-1, 0, 1]
    D_H = np.absolute(ndimage.convolve1d(delta_H, conv_kernel_2))
    D_V = np.absolute(ndimage.convolve1d(delta_V.T, conv_kernel_2)).T

    conv_kernel_3 = np.array([[0 for i in range(9)] for j in range(9)])

    conv_kernel_3[:, :] = 0
    conv_kernel_3[2:7, 0:5] = 1
    W_W = signal.convolve2d(D_H, conv_kernel_3, mode='same')

    conv_kernel_3[:, :] = 0
    conv_kernel_3[2:7, 4:9] = 1
    W_E = signal.convolve2d(D_H, conv_kernel_3, mode='same')

    conv_kernel_3[:,  :] = 0
    conv_kernel_3[0:5, 2:7] = 1
    W_N = signal.convolve2d(D_V, conv_kernel_3, mode='same')

    conv_kernel_3[:,  :] = 0
    conv_kernel_3[4:9, 2:7] = 1
    W_S = signal.convolve2d(D_V, conv_kernel_3, mode='same')

    # 如果某个方向是零，表示这一大片都是一个颜色，此时需要直接根据该方向的结果来作为最后的预测值
    # 因此，设定很小的一个值，取倒数后很大，之后计算的时候，其他几个方向都像被忽略不算一样，从而实现目标
    W_W[W_W==0] = 1e-10
    W_E[W_E==0] = 1e-10
    W_N[W_N==0] = 1e-10
    W_S[W_S==0] = 1e-10

    W_W = 1 / np.square(W_W)
    W_E = 1 / np.square(W_E)
    W_N = 1 / np.square(W_N)
    W_S = 1 / np.square(W_S)

    f = [1, 2, 3, 4, 5]
    conv_kernel_4 = np.array([0 for i in range(9)])

    delta_Final = list()
    for c in range(2):
        now_delta_V = np.copy(delta_V)
        now_delta_H = np.copy(delta_H)

        now_delta_H[c::2, :] = delta_H[c::2, :]
        now_delta_V[:, c::2] = delta_V[:, c::2]

        conv_kernel_4[:] = 0
        conv_kernel_4[0:5] = f
        V1 = ndimage.convolve1d(now_delta_V.T, conv_kernel_4).T
        conv_kernel_4[:] = 0
        conv_kernel_4[4:9] = f[::-1]
        V2 = ndimage.convolve1d(now_delta_V.T, conv_kernel_4).T
        conv_kernel_4[:] = 0
        conv_kernel_4[0:5] = f
        V3 = ndimage.convolve1d(now_delta_H, conv_kernel_4)
        conv_kernel_4[:] = 0
        conv_kernel_4[4:9] = f[::-1]
        V4 = ndimage.convolve1d(now_delta_H, conv_kernel_4)

        delta_Final.append(1/np.sum(f) * (V1 * W_N + V2 * W_S + V3 * W_E + V4 * W_W) / (W_N + W_S + W_E + W_W))

    [delta_R, delta_B] = delta_Final

    G_Final[0::2, 0::2] = R[0::2, 0::2] + delta_R[0::2, 0::2]
    G_Final[1::2, 1::2] = B[1::2, 1::2] + delta_B[1::2, 1::2]

    conv_kernel_5 = np.array([[0 for _ in range(7)] for _ in range(7)])
    conv_kernel_5[0, 2] = conv_kernel_5[0, 4] = conv_kernel_5[2, 0] = conv_kernel_5[2, 6] = -1
    conv_kernel_5[6, 2] = conv_kernel_5[6, 4] = conv_kernel_5[4, 0] = conv_kernel_5[4, 6] = -1
    conv_kernel_5[2, 2] = conv_kernel_5[2, 4] = conv_kernel_5[4, 2] = conv_kernel_5[4, 4] = 10

    conv_kernel_6 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    for c in range(2):
        C_adjust = G_Final - signal.convolve2d(delta_Final[c], conv_kernel_5, mode='same') / 32
        C_Final[c][1-c::2, 1-c::2] = C_adjust[1-c::2, 1-c::2]

        C_adjust = G - signal.convolve2d(G_Final - C_Final[c], conv_kernel_6, mode='same') / 4

        C_Final[c][c::2, 1-c::2] = C_adjust[c::2, 1-c::2]
        C_Final[c][1-c::2, c::2] = C_adjust[1-c::2, c::2]

    new_img = np.dstack((C_Final[0], G_Final, C_Final[1]))
    new_img = (new_img + 0.5).clip(0, 255)

    return new_img.astype(np.uint8)
