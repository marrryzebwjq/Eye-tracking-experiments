"""
A python implementation of the model of R. Achanta for visual saliency.

The original code is available at http://ivrg.epfl.ch/supplementary_material/RK_CVPR09/index.html

Modified by Alexandre Bruckert - 2022

Relevant publication :
R. Achanta, S. Hemami, F. Estrada, S. Süsstrunk, Frequency-tuned Salient Region Detection,
IEEE CVPR, 2009.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

class Achanta:
    """
    Implementation of Achanta's saliency model.
    """

    def __init__(self, kernel_size=3):
        self.SM = None
        self.kernel_size = kernel_size

    def get_salmap(self, img):
        # Blur the image with either a 3x3 or 5x5 kernel
        filtered_img = gaussian_filter(img, sigma=self.kernel_size, mode='mirror', truncate=1)

        # RGB to CIELab color space conversion
        lab_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab_img)
        L_mean = np.mean(L)
        A_mean = np.mean(A)
        B_mean = np.mean(B)

        # Compute the saliency map out of it
        self.SM = (L-L_mean)**2 + (A-A_mean)**2 + (B-B_mean)**2
        return self.SM

