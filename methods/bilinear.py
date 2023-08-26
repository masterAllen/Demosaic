import numpy as np
from scipy import signal, ndimage

def run(img):
    new_img = np.zeros(img.shape).astype(float)

    conv_kernel_1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    conv_kernel_2 = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])

    # 需要先设置为 float, 否则溢出!
    # new_img = np.zeros_like(img)
    # test = (signal.convolve2d(img[:, :, 0], conv_kernel_1, mode='same'))
    # print(test[1, 1])
    # new_img[:, :, 0] = test
    # print(new_img[1, 1])

    new_img[:, :, 0] = signal.convolve2d(img[:, :, 0], conv_kernel_1, mode='same') / 4
    new_img[:, :, 1] = signal.convolve2d(img[:, :, 1], conv_kernel_2, mode='same') / 4
    new_img[:, :, 2] = signal.convolve2d(img[:, :, 2], conv_kernel_1, mode='same') / 4

    new_img = (new_img + 0.5).clip(0, 255.5)
    return new_img.astype(np.uint8)
