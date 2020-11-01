import os.path as osp

import math
import numpy as np
import cv2

from skimage.metrics import mean_squared_error

import sys
sys.path.append('..')


def median_filter(image, kernel_size: int):
    dst = image.copy()

    radius = kernel_size // 2
    temp = image[0, 0].copy()
    pixels = [temp] * kernel_size * kernel_size

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            i = 0
            for px in range(-radius, radius + 1):
                for py in range(-radius, radius + 1):
                    ind_x = max(0, min(x + px, image.shape[0] - 1))
                    ind_y = max(0, min(y + py, image.shape[1] - 1))
                    pixels[i] = image[ind_x, ind_y]
                    i += 1

            pixels.sort(key=lambda r:
                        0.299 * r[0] + 0.587 * r[1] + 0.114 * r[2])
            dst[x, y] = pixels[kernel_size * kernel_size // 2]
    return dst


def MedianBlur(image_path: str, kernel_size: int = 3):
    img = cv2.imread(image_path)

    median = cv2.medianBlur(img, kernel_size)
    custom_median = median_filter(img, kernel_size)

    # create_two_windows(median, custom_median)
    print('Compare median and custom_median:')
    print(mean_squared_error(median, custom_median))
    print('Compare median and source:')
    print(mean_squared_error(median, img))


def gaussian_filter(image, kernel_size: int, sigma: float):
    dst = image.copy()

    kernel = np.ones((kernel_size, kernel_size))
    radius = kernel_size // 2
    norm = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            kernel[i + radius, j + radius] = math.exp(-(i * i + j * j) /
                                                      (2 * sigma * sigma))
            norm += kernel[i + radius, j + radius]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            kernel[i, j] /= norm

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            val = 0
            for px in range(-radius, radius + 1):
                for py in range(-radius, radius + 1):
                    ind_x = max(0, min(x + px, image.shape[0] - 1))
                    ind_y = max(0, min(y + py, image.shape[1] - 1))
                    val += (kernel[px + radius, py + radius] *
                            image[ind_x, ind_y])
            dst[x, y] = val
    return dst


def GaussianBlur(image_path: str):
    img = cv2.imread(image_path)

    gauss = cv2.GaussianBlur(img, (3, 3), 1)
    custom_gauss = gaussian_filter(img, 3, 1)

    # create_two_windows(gauss, custom_gauss)
    print('Compare gauss and custom_gauss:')
    print(mean_squared_error(gauss, custom_gauss))
    print('Compare gauss and source:')
    print(mean_squared_error(gauss, img))


def blur_filter(image, kernel_size: int):
    dst = image.copy()

    radius = kernel_size // 2

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            val = 0
            for px in range(-radius, radius + 1):
                for py in range(-radius, radius + 1):
                    ind_x = max(0, min(x + px, image.shape[0] - 1))
                    ind_y = max(0, min(y + py, image.shape[1] - 1))
                    val += image[ind_x, ind_y] / (kernel_size * kernel_size)
            dst[x, y] = val
    return dst


def Blur(image_path: str):
    img = cv2.imread(image_path)

    blur = cv2.blur(img, (3, 3))
    custom_blur = blur_filter(img, 3)

    # create_two_windows(blur, custom_blur)
    print('Compare blur and custom_blur:')
    print(mean_squared_error(blur, custom_blur))
    print('Compare blur and source:')
    print(mean_squared_error(blur, img))


if __name__ == "__main__":
    try:
        image_path = sys.argv[1]
        assert osp.isfile(image_path), '{} is not a file!'.format(image_path)
    except (IndexError, AssertionError):
        print('path to the image is not valid! The default path was set!')
        image_path = './src/cat.jpg'

    MedianBlur(image_path)
    GaussianBlur(image_path)
    Blur(image_path)
