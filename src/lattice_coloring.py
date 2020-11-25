from src.utils import *
import numpy as np
import cv2


I = cv2.imread('../data/dolphin-2.jpg')
IMAGE_NAME = 'dolphin-2'
CURRENT_BOUNDS = BOUNDS_NORMAL
compute_and_save_lattice_color_images(I, IMAGE_NAME, CURRENT_BOUNDS)
