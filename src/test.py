import numpy as np
import cv2
from src.utils import *
from src.pencil_sketch import PencilSketch
from src.control_parameters import *


I = cv2.imread('../data/flower-rose.jpeg')
IMAGE_NAME = 'flower-rose'
CURRENT_BOUNDS = BOUNDS_NORMAL
ASSETS_DIR = os.path.abspath('../assets')
result_path = os.path.join(RESULTS_DIR, IMAGE_NAME, get_params_dir_name(), bounds_dir_name(CURRENT_BOUNDS), VERTEX_COLORING, 'result') + PNG
concat_path = os.path.join(ASSETS_DIR, f'{IMAGE_NAME}-result') + PNG
L3_PATH = os.path.join(RESULTS_DIR, IMAGE_NAME, get_params_dir_name(), bounds_dir_name(CURRENT_BOUNDS), VERTEX_COLORING, 'lattice-2') + JPG

create_vertex_shaded_image(I, image_name=IMAGE_NAME, bounds=CURRENT_BOUNDS)
S = PencilSketch(I, bg_gray='').render()
L3 = cv2.imread(L3_PATH)
R = get_linear_combination(I, image_name=IMAGE_NAME, weights=(0.05, 0.05, HAND_DRAWN), bounds=CURRENT_BOUNDS)
cv2.imshow('original', I)
cv2.imshow('current-soa', S)
cv2.imshow('novel', R)
# cv2.imshow('lattice-3', L3)
cv2.waitKey(0)
cv2.imwrite(result_path, R)

R = np.concatenate((I, R), axis=1)
# cv2.imshow('concat', R)
# cv2.waitKey(0)
cv2.imwrite(concat_path, R)
