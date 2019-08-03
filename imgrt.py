from scipy import ndimage
import cv2
#image=cv2.imread(image)
def rotate(image):
    image = ndimage.rotate(image,90)
    return image

