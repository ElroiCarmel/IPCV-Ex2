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

    magnitude = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))

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
        temp = np.convolve(temp, [1, 1])

    for i in range(n):
        ker[i] = temp[i] * temp

    return ker / np.power(2, 2 * n - 2)


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
    magnitude = np.where(magnitude < thresh, 0, 255)
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
    return edgeDetectionZeroCrossingSimple(blurImage1(img, 15))


def edgeDetectionCanny(img: np.ndarray, thrs_high: float, thrs_low: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_high: T1
    :param thrs_low: T2
    :return: opencv solution, my implementation
    """
    # 1. Smooth the image with a Gaussian kernel
    smoothed = blurImage2(img, 5)
    # 2. Compute the partial derivatives lx, ly
    #   - Use simple derivative kernels

    # 3. Compute the magnitude and the direction of the gradients
    directions, magnitude, _, _ = convDerivative(smoothed)
    # 4. Quantize the gradient directions
    # Convert to from radians to degree
    directions = np.rad2deg(directions)
    # Since directions can be negative apply module
    directions = np.mod(directions, 180)
    # Round to the nearest 45 degree
    directions = np.round(directions / 45) * 45
    directions = directions.astype(int)
    # 5. Perform non-maximum suppression
    h, w = magnitude.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            deg = directions[i, j]
            x_mov, y_mov = conv_degree(deg)
            mag = magnitude[i, j]
            adjacent = magnitude[[i, i+y_mov, i-y_mov], [j, i+x_mov, j-x_mov]]
            if mag != np.max(adjacent):
                magnitude[i, j] = 0
    # 6. Hysteresis
    max_mag = np.max(magnitude)
    magnitude = magnitude / max_mag # Normalize
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            mag = magnitude[i, j]
            if mag >= thrs_high:
                continue
            elif mag < thrs_low:
                magnitude[i, j] = 0
            else:
                adjacent = magnitude[i-1:i+2, j-1:j+2]
                if np.sum(adjacent) - mag <= 0:
                    magnitude[i, j] = 0
    magnitude[magnitude != 0] = 255
    cv_sol = cv2.Canny(img, thrs_low * max_mag, thrs_high * max_mag)
    return cv_sol, magnitude.astype(np.uint8)

def conv_degree(degree: int) -> (int, int):
    """
    Helper function to convert a degree to movement on the x and y axes
    :param degree: 0/45//90/135/180
    :return: a tuple (x, y) which represents the movement
    """
    if degree == 0:
        return 1, 0
    elif degree == 45:
        return 1, -1
    elif degree == 90:
        return 0, -1
    elif degree == 135:
        return -1, -1
    elif degree == 180:
        return -1, 0


def houghCircles(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    edges = cv2.Canny(img, 500, 800)
    h, w = img.shape
    edge_mask = edges > 0
    y_indices, x_indices = np.indices((h, w))
    y_indices, x_indices = y_indices[edge_mask], x_indices[edge_mask]
    votes = np.zeros(shape=(h, w, max_radius - min_radius + 1), dtype=int)

    for radius in range(min_radius, max_radius + 1):
        for theta in range(360):
            x_circle = x_indices + radius * np.cos(theta)
            y_circle = y_indices - radius * np.sin(theta)
            x_circle = np.round(x_circle).astype(int)
            y_circle = np.round(y_circle).astype(int)
            mask_y, mask_x = np.logical_and(y_circle >= 0, y_circle < h), np.logical_and(x_circle >= 0, x_circle < w)
            mask_common = np.logical_and(mask_y, mask_x)
            votes[y_circle[mask_common], x_circle[mask_common], radius - min_radius] += 1


    ans = []
    for i in range(max_radius-min_radius+1):
        ans.append(votes[:,:,i].copy().astype(np.uint8))
    return ans


