import cv2
import os
from src.utils2 import *

I = cv2.imread('../data/lenna.png')


# sigmas = [1, 10, 50]
# B = [cv2.GaussianBlur(I, (101, 101), sigmaX=i) for i in sigmas]
# cv2.imshow('image', I)
# [cv2.imshow(f'blur-{i}', b) for i, b in zip(sigmas, B)]
# [cv2.imwrite(os.path.join(RESULTS_DIR, 'lenna', f'gaussian-{i}') + PNG, b) for i, b in zip(sigmas, B)]
# cv2.waitKey(0)


def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)


masks = [100, 150, 200, 254]
J = [burnV2(I, mask=np.zeros(I.shape, dtype=np.uint8) + mask) for mask in masks]
[cv2.imshow(f'result-{mask}', j) for mask, j in zip(masks, J)]
cv2.waitKey(0)
[cv2.imwrite(os.path.join(RESULTS_DIR, 'lenna', f'burn-{mask}') + PNG, j) for mask, j in zip(masks, J)]
