import cv2
import numpy as np
def sharpen(im):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(im, -1, kernel)
    return im
