import cv2
import numpy as np
from matplotlib import pyplot as plt
def show_image(img):
    plt.imshow(img,"gray")
    plt.show()
def pre_process_image(img, skip_dilate=False):
    """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

    # Gaussian blur with a kernal size (height, width) of 9.
    # Note that kernal sizes must be positive and odd and the kernel must be square.
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
        #Dilate the image to increase the size of the grid lines.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc

img = cv2.imread('idealsudoko.jpeg', cv2.IMREAD_GRAYSCALE)
processed = pre_process_image(img)

# See: https://gist.github.com/mineshpatel1/22e86200eee86ebe3e221343b26fc3f3#file-show_image-py
show_image(processed)
