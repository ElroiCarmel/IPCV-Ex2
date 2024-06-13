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
    pass


def blurDemo():
    pass


def edgeDemo():
    pass


def houghDemo():
    pass


def main():
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()


if __name__ == '__main__':
    main()
