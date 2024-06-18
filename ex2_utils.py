import numpy as np
import cv2


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    Result is same as calling np.convolve(signal, kernel, 'full')
    :param inSignal: 1-D array
    :param kernel1: 1-D array as kernel
    :return: The convolved array
    """
    n, m = len(inSignal), len(kernel1)
    size = n + m - 1
    ans = np.zeros(size, dtype=int)
    for i in range(n):
        j, k = i, 0
        while j > -1 and k < m:
            ans[i] += inSignal[j] * kernel1[k]
            j, k = j - 1, k + 1
    for i in range(n, size):
        j, k = n - 1, i - n + 1
        while j > -1 and k < m:
            ans[i] += inSignal[j] * kernel1[k]
            j, k = j - 1, k + 1
    return ans


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2-D image
    :param kernel2: kernel
    :return: The convolved image
    """
    h, w = inImage.shape
    output = np.zeros((h, w))
    center = int(kernel2.shape[0] / 2)

    for r in range(h):
        for c in range(w):
            for i in range(-center, center + 1):
                for j in range(-center, center + 1):
                    row, column = np.clip(r - i, 0, h - 1), np.clip(c + j, 0, w - 1)
                    output[r, c] += inImage[row, column] * kernel2[center - i, center + j]
    return output.clip(0, 255)


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale image
    :return: (directions, magnitude,x_der,y_der)
    """
    shape = inImage.shape
    der_kernel = [1, 0, -1]

    x_der = np.zeros(shape, dtype=int)
    for r in range(shape[0]):
        x_der[r] = np.convolve(inImage[r], der_kernel, mode="same")

    y_der = np.zeros(shape, dtype=int)
    for c in range(shape[1]):
        y_der[:, c] = np.convolve(inImage[:, c], der_kernel, mode="same")

    magnitude = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2)).astype(np.uint8)

    directions = np.arctan2(y_der, x_der)
    return directions, magnitude, x_der, y_der

def getGaussianKernel(n: int) -> np.ndarray:
    """
    Generate a Gaussian kernel
    :param n: The desired kernel size. N should be odd.
    :return: An approximation of the Gaussian kernel using the binomial kernel
    """
    ker = np.zeros((n, n))

    temp = [1]
    for i in range(1, n):
        temp = np.convolve(temp, [1,1])

    for i in range(n):
        ker[i] = temp[i] * temp

    return ker/np.power(2, 2*n-2)


def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """

    gaussian_kernel = getGaussianKernel(kernel_size)
    return cv2.filter2D(in_image, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    ker = cv2.getGaussianKernel(kernel_size, sigma=-1)
    return cv2.filter2D(in_image, -1, ker, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    # First my solution
    x_ker = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
    y_ker = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
    # I am using the cv2.filter only because it's faster!
    x_der = cv2.filter2D(img, cv2.CV_64F, x_ker, borderType=cv2.BORDER_REPLICATE)
    y_der = cv2.filter2D(img, cv2.CV_64F, y_ker, borderType=cv2.BORDER_REPLICATE)
    magnitude = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))
    magnitude /= 255
    magnitude[magnitude < thresh], magnitude[magnitude >= thresh] = 0, 255
    # Open-cv solution
    cv_sol = cv2.Sobel(img, -1, 1, 1, borderType=cv2.BORDER_REPLICATE)
    return cv_sol, magnitude.astype(np.uint8)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    # youtube.com/watch?v=uNP6ZwQ3r6A
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    der = cv2.filter2D(img, cv2.CV_16S, kernel, borderType=cv2.BORDER_REPLICATE)
    pos_mask = der > 0
    zero_crossing_x = np.logical_xor(pos_mask[:, 1:], pos_mask[:, :-1])
    zero_crossing_y = np.logical_xor(pos_mask[1:, :], pos_mask[:-1, :])

    zero_crossings = np.zeros_like(img, dtype=bool)
    zero_crossings[:, 1:] |= zero_crossing_x
    zero_crossings[1:, :] |= zero_crossing_y

    ans = np.where(zero_crossings, 255, 0).astype(np.uint8)
    return ans


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detects edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge map
    """
    return edgeDetectionZeroCrossingSimple(blurImage1(img, 21))




if __name__ == '__main__':
    img = cv2.imread('codeMonkey.jpeg', cv2.IMREAD_GRAYSCALE)
