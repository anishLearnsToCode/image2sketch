import cv2
from src.control_parameters import *


class PencilSketch:
    """Pencil sketch effect
        A class that applies a pencil sketch effect to an image.
        The processed image is overlayed over a background image for visual
        effect.
    """

    def __init__(self, I, bg_gray='../data/pencilsketch_bg.jpg'):
        """Initialize parameters
            :param (width, height): Image size.
            :param bg_gray: Optional background image to improve the illusion
                            that the pencil sketch was drawn on a canvas.
        """
        self.I = I
        self.width, self.height, _= I.shape

        # try to open background canvas (if it exists)
        self.canvas = cv2.imread(bg_gray, cv2.CV_8UC1)
        if self.canvas is not None:
            self.canvas = cv2.resize(self.canvas, (self.width, self.height))

    def render(self):
        """Applies pencil sketch effect to an RGB image
            :param img_rgb: RGB image to be processed
            :returns: Processed RGB image
        """
        blur_factor = (SKETCH_DENSITY, ) * 2
        img_gray = cv2.cvtColor(self.I, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, blur_factor, 0, 0)
        # cv2.imshow('im blur', img_blur)
        img_blend = cv2.divide(img_gray, img_blur, scale=256)

        # if available, blend with background canvas
        if self.canvas is not None:
            img_blend = cv2.multiply(img_blend, self.canvas, scale=1. / 256)

        return cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)
