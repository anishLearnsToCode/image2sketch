import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils2 import *


img = cv2.imread('../data/flower-2.jpg', 0)
edges = cv2.Canny(img, 100, 200)
edges = np.abs(np.array(edges, dtype=np.int) - 255)
edges = np.array(edges, dtype=np.uint8)
print(edges)
cv2.imshow('edges', edges)
cv2.waitKey(0)

cv2.imwrite(os.path.join(RESULTS_DIR, 'flower-2', 'canny-edge') + PNG, edges)

# plt.subplot(121), plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
