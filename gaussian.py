import cv2
import numpy as np
from PIL import Image


def get_gaussian():
    im = Image.open('./saved_img/input.jpg')
    im_array = np.asarray(im)

    kernel1d = cv2.getGaussianKernel(17, 5)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())

    low_im_array = cv2.filter2D(im_array, -1, kernel2d)

    low_im = Image.fromarray(low_im_array)
    low_im.save('output.jpg')
