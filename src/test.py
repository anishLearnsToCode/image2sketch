from src.utils2 import *
from src.control_parameters import *


I = cv2.imread('../data/bridge.jpeg')
IMAGE_NAME = 'bridge'
CURRENT_BOUNDS = BOUNDS_6
W = (0.2, 0.05, HAND_DRAWN)

ASSETS_DIR = os.path.abspath('../assets')
concat_path = os.path.join(ASSETS_DIR, f'{IMAGE_NAME}-result') + JPG
soa_path = os.path.join(RESULTS_DIR, IMAGE_NAME, 'blend') + PNG
novel_path = os.path.join(RESULTS_DIR, IMAGE_NAME, get_params_dir_name(), 'novel') + PNG

S, R = generate_pencil_sketch_of_image(I, image_name=IMAGE_NAME, bounds=CURRENT_BOUNDS, weights=W, brightness=0)
cv2.imshow('original', I)
cv2.imshow('soa', S)
cv2.imshow('novel', R)
cv2.waitKey(0)

cv2.imwrite(novel_path, R)
cv2.imwrite(soa_path, S)
