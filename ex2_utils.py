import numpy as np


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
        j, k = n-1, i - n + 1
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

