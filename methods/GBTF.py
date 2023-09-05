import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

def run(img):
    S = img.astype(float)
    R, G, B = S[:,:,0], S[:,:,1], S[:,:,2]
    new_R = np.copy(R); new_G = np.copy(G); new_B = np.copy(B)

    # Step 1: Interploate
    H = convolve1d((R+G+B), [-1, 2, 2, 2, -1]) / 4
    G_H = np.copy(G); R_H = np.copy(R); B_H = np.copy(B)
    G_H[0::2, 0::2] = H[0::2, 0::2]
    G_H[1::2, 1::2] = H[1::2, 1::2]
    R_H[0::2, 1::2] = H[0::2, 1::2]
    B_H[1::2, 0::2] = H[1::2, 0::2]

    V = convolve1d((R+G+B).T, [-1, 2, 2, 2, -1]).T / 4
    G_V = np.copy(G); R_V = np.copy(R); B_V = np.copy(B)
    G_V[0::2, 0::2] = V[0::2, 0::2]
    G_V[1::2, 1::2] = V[1::2, 1::2]
    R_V[1::2, 0::2] = V[1::2, 0::2]
    B_V[0::2, 1::2] = V[0::2, 1::2]

    # Step 2: Compute delta, compute gradient
    delta_H = G_H - R_H - B_H
    delta_V = G_V - R_V - B_V

    D_H = np.absolute(convolve1d(delta_H, [-1, 0, 1]))
    D_V = np.absolute(convolve1d(delta_V.T, [-1, 0, 1])).T

    # Step 3: Compute coeffecient (eq.6 in paper)
    now_kernel = np.zeros((9, 9)); now_kernel[2:7, 0:5] = 1
    W_W = convolve2d(D_H, now_kernel, mode='same')
    now_kernel = np.zeros((9, 9)); now_kernel[2:7, 4:9] = 1
    W_E = convolve2d(D_H, now_kernel, mode='same')
    now_kernel = np.zeros((9, 9)); now_kernel[0:5, 2:7] = 1
    W_N = convolve2d(D_V, now_kernel, mode='same')
    now_kernel = np.zeros((9, 9)); now_kernel[4:9, 2:7] = 1
    W_S = convolve2d(D_V, now_kernel, mode='same')

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
    W_T = W_W + W_E + W_N + W_S

    # Step 4: Compute delta finally(eq.5 in paper)
    f = [1, 1, 1, 1, 1]; f = f / np.sum(f)

    each_delta = list()
    for c in range(2):
        now_delta_H = np.zeros_like(delta_H)
        now_delta_V = np.zeros_like(delta_V)

        now_delta_H[c::2, :] = delta_H[c::2, :]
        now_delta_V[:, c::2] = delta_V[:, c::2]

        now_kernel = np.zeros((9)); now_kernel[0:5] = f
        V1 = convolve1d(now_delta_V.T, now_kernel).T
        now_kernel = np.zeros((9)); now_kernel[4:9] = f[::-1]
        V2 = convolve1d(now_delta_V.T, now_kernel).T
        now_kernel = np.zeros((9)); now_kernel[0:5] = f
        V3 = convolve1d(now_delta_H, now_kernel)
        now_kernel = np.zeros((9)); now_kernel[4:9] = f[::-1]
        V4 = convolve1d(now_delta_H, now_kernel)

        each_delta.append((V1*W_N + V2*W_S + V3*W_E + V4*W_W) / W_T)

    [delta_GR, delta_GB] = each_delta

    # Step 5: recover G now~ (Eq.8 in paper) 
    new_G[0::2, 0::2] = R[0::2, 0::2] + delta_GR[0::2, 0::2]
    new_G[1::2, 1::2] = B[1::2, 1::2] + delta_GB[1::2, 1::2]

    # Step 6: recover R in B (B in R)(Eq.9 in paper)
    prb = np.zeros((7, 7))
    prb[0, 2] = prb[0, 4] = prb[2, 0] = prb[2, 6] = -1
    prb[6, 2] = prb[6, 4] = prb[4, 0] = prb[4, 6] = -1
    prb[2, 2] = prb[2, 4] = prb[4, 2] = prb[4, 4] = 10

    now_kernel = np.ones((7, 7))

    # Can simplify eq.9 to this
    # now_delta = convolve2d(delta_GR, now_kernel, mode='same') / np.sum(now_kernel)
    now_delta = convolve2d(delta_GR, prb, mode='same') / np.sum(prb)
    new_R[1::2, 1::2] = (new_G - now_delta)[1::2, 1::2]
    now_delta = convolve2d(delta_GB, prb, mode='same') / np.sum(prb)
    new_B[0::2, 0::2] = (new_G - now_delta)[0::2, 0::2]

    # Step 7: recover R/B in G (Eq.10 in paper)
    now_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    now_R = G - convolve2d(new_G - new_R, now_kernel, mode='same') / 4
    new_R[0::2, 1::2] = now_R[0::2, 1::2]; new_R[1::2, 0::2] = now_R[1::2, 0::2]
    now_B = G - convolve2d(new_G - new_B, now_kernel, mode='same') / 4
    new_B[0::2, 1::2] = now_B[0::2, 1::2]; new_B[1::2, 0::2] = now_B[1::2, 0::2]

    new_img = np.dstack((new_R, new_G, new_B))
    new_img = (new_img + 0.5).clip(0, 255)

    return new_img.astype(np.uint8)
