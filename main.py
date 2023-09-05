import cv2
import numpy as np
from matplotlib import pyplot as plt

from methods import Bilinear, HQL, HA
from methods import GBTF, RI, DLMMSE, IRI

def metrics(raw_img, new_img):
    result = []
    for c in range(0, 3):
        mse = np.mean(np.power((raw_img[5:-5, 5:-5, c] - new_img[5:-5, 5:-5, c]), 2))
        if mse == 0:
            result.append('perfect!!')
        else:
            result.append(10 * np.log10(255*255/mse))

    return result

def make_bayer(img):
    new_img = np.zeros_like(img)

    new_img[0::2, 0::2, 0] = img[0::2, 0::2, 0]
    new_img[0::2, 1::2, 1] = img[0::2, 1::2, 1]
    new_img[1::2, 0::2, 1] = img[1::2, 0::2, 1]
    new_img[1::2, 1::2, 2] = img[1::2, 1::2, 2]

    return new_img

for picname in ['./kodim19.png']:
    src_img = cv2.imread(picname)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    bayer_img = make_bayer(src_img)
    plt.imshow(bayer_img), plt.show()

    bilinear_img = Bilinear.run(bayer_img)
    plt.imshow(bilinear_img), plt.show()
    print('Bilinear: ', metrics(src_img, bilinear_img))

    hql_img = HQL.run(bayer_img)
    plt.imshow(hql_img), plt.show()
    print('HQL: ', metrics(src_img, hql_img))

    ha_img = HA.run(bayer_img)
    plt.imshow(ha_img), plt.show()
    print('HA: ', metrics(src_img, ha_img))

    dlmmse_img = DLMMSE.run(bayer_img)
    plt.imshow(dlmmse_img), plt.show()
    print('DLMMSE: ', metrics(src_img, dlmmse_img))

    gbtf_img = GBTF.run(bayer_img)
    plt.imshow(gbtf_img), plt.show()
    print('GBTF: ', metrics(src_img, gbtf_img))

    ri_img = RI.run(bayer_img)
    plt.imshow(ri_img), plt.show()
    print('RI: ', metrics(src_img, ri_img))

    iri_img = IRI.run(bayer_img)
    plt.imshow(iri_img), plt.show()
    print('IRI: ', metrics(src_img, iri_img))
