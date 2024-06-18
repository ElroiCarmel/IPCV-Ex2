import cv2

from ex2_utils import *
import matplotlib.pyplot as plt
import time


def conv1Demo():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3])
    print(f"My convolution: {conv1D(a, b)}")
    print(f"Numpy convolution: {np.convolve(a, b)}")


def conv2Demo():
    rand_img = np.random.randint(low=0, high=255, size=(10, 10)).astype(np.uint8)
    ker = np.ones((5, 5)) / 25

    print(f"The random image: {rand_img}")
    my_convolved = conv2D(rand_img, ker)
    my_convolved = np.round(my_convolved).astype(np.uint8)
    cv_convolved = cv2.filter2D(rand_img, -1, ker, borderType=cv2.BORDER_REPLICATE)

    print(f"Check if the same: {np.array_equal(my_convolved, cv_convolved)}")


def derivDemo():
    img = cv2.imread('boxman.jpeg', cv2.IMREAD_GRAYSCALE)
    directions, magnitude, x_der, y_der = convDerivative(img)
    cv2.imshow("The derivative magnitude", magnitude)
    cv2.waitKey(0)


def blurDemo():
    img = cv2.imread('beach.jpeg', cv2.IMREAD_GRAYSCALE)
    ker_size = 5
    my_blurred = blurImage1(img, ker_size)
    cv_blurred = blurImage2(img, ker_size)
    cv2.imshow("The blurred image", my_blurred)
    cv2.waitKey(0)
    cv2.imshow("The CV-blurred image", cv_blurred)
    cv2.waitKey(0)


def edgeDemo():
    # Using sobel on codeMonkey
    img = cv2.imread('codeMonkey.jpeg', cv2.IMREAD_GRAYSCALE)
    cv_sobel, my_sobel = edgeDetectionSobel(img, 0.35)
    cv2.imshow("CV-Sobel image", cv_sobel)
    cv2.waitKey(0)
    cv2.imshow("My-Sobel image", my_sobel)
    cv2.waitKey(0)
    # Using Zero-Crossing Simple
    zero_crossing = edgeDetectionZeroCrossingSimple(img)
    cv2.imshow("zero-crossing image", zero_crossing)
    cv2.waitKey(0)


def houghDemo():
    pass


def main():
    # conv1Demo()
    # conv2Demo()
    # derivDemo()
    # blurDemo()
    edgeDemo()
    houghDemo()


if __name__ == '__main__':
    main()