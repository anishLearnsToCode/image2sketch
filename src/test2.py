import cv2
import numpy as np
import pprint
import os
from src.utils2 import *
from src.control_parameters import *


I = cv2.imread('../data/switzerland/swiss-3.jpg')
IMAGE_NAME = 'swiss-3'
CURRENT_BOUNDS = BOUNDS_6
S, R = generate_pencil_sketch_of_image(I, image_name=IMAGE_NAME, bounds=CURRENT_BOUNDS, weights=(0.03, 0.03, HAND_DRAWN))
cv2.imshow('original', I)
cv2.imshow('soa', S)
cv2.imshow('novel', R)
cv2.waitKey(0)
