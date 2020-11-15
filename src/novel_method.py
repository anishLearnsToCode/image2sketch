from src.utils2 import *
import cv2
import numpy as np
import pprint


I = cv2.imread('../data/flower-rose.jpeg')
IMAGE_NAME = 'flower-rose'

J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', J)
L = J / 255
print(L)
cv2.imshow('normalized', L)
G = [inverse_gaussian(EPSILON, sigma, 1, L) for sigma in SIGMA]
# cv2.imshow('lenna', J)
[cv2.imshow(f'inverse-gaussian-{sigma}', g / np.max(g)) for sigma, g in zip(SIGMA, G)]
# cv2.waitKey(0)
# g = G[2]
# g = g / np.max(g)
G = [np.array((g / np.max(g)) * 255, dtype=np.uint8) for g in G]
# g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
# print(G[1])
# cv2.imshow('my', G[2])
cv2.waitKey(0)
# cv2.imwrite(os.path.join(RESULTS_DIR, IMAGE_NAME, get_params_dir_name(), f'gaussian-inverse-{10 * SQRT_2PI}') + PNG, g)

make_dir_if_absent(os.path.join(RESULTS_DIR, IMAGE_NAME, get_params_dir_name()))
cv2.imwrite(os.path.join(RESULTS_DIR, IMAGE_NAME, get_params_dir_name(), 'grayscale') + PNG, J)
[cv2.imwrite(os.path.join(RESULTS_DIR, IMAGE_NAME, get_params_dir_name(), f'gaussian-inverse-{sigma * SQRT_2PI}') + PNG, g) for sigma, g in zip(SIGMA, G)]
# cv2.imshow('gaussian-inv-1', G[1])
# cv2.waitKey(0)
