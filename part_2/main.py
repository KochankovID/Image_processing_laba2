import os.path as osp

import math
import numpy as np
import cv2
import time

from skimage.metrics import mean_squared_error

import sys
sys.path.append('..')

from opencv_scripts.windows_manager import create_two_windows
from part_1.main import get_gauss_noise


def median_filter(image, kernel_size: int):
    radius = kernel_size // 2

    temp = image.copy()
    upper_border = np.repeat(np.expand_dims(temp[0, :], axis=0),
                             radius, axis=0)
    lower_border = np.repeat(np.expand_dims(temp[-1, :], axis=0),
                             radius, axis=0)
    temp = cv2.vconcat([upper_border, temp, lower_border])
    left_border = np.repeat(np.expand_dims(temp[:, 0], axis=1),
                            radius, axis=1)
    right_border = np.repeat(np.expand_dims(temp[:, -1], axis=1),
                             radius, axis=1)
    temp = cv2.hconcat([left_border, temp, right_border])

    dst = temp.copy()
    for x in range(radius, temp.shape[0] - radius):
        for y in range(radius, temp.shape[1] - radius):
            red_array = temp[(x - radius):(x + radius + 1),
                             (y - radius):(y + radius + 1), 2]
            red_median = np.median(red_array)
            green_array = temp[(x - radius):(x + radius + 1),
                               (y - radius):(y + radius + 1), 1]
            green_median = np.median(green_array)
            blue_array = temp[(x - radius):(x + radius + 1),
                              (y - radius):(y + radius + 1), 0]
            blue_median = np.median(blue_array)
            dst[x, y] = [blue_median, green_median, red_median]
    return dst[radius:-radius, radius:-radius]


def MedianBlur(image_path: str, kernel_size: int = 3):
    img = cv2.imread(image_path)
    noise = get_gauss_noise(img)

    start = time.time()
    median = cv2.medianBlur(img, kernel_size)
    end = time.time()
    print('\nTime:')
    print(end - start)
    print('Compare median and noise:')
    print(mean_squared_error(median, noise))
    print('Compare median and source:')
    print(mean_squared_error(median, img))

    start = time.time()
    custom_median = median_filter(img, kernel_size)
    end = time.time()
    print('\nTime:')
    print(end - start)
    print('Compare custom_median and noise:')
    print(mean_squared_error(custom_median, noise))
    print('Compare custom_median and source:')
    print(mean_squared_error(custom_median, img))

    print('\nCompare median and custom_median:')
    print(mean_squared_error(median, custom_median))
    create_two_windows(median, custom_median)


def gaussian_filter(image, kernel_size: (int, int), sigma: float):
    radius_x = kernel_size[0] // 2
    radius_y = kernel_size[1] // 2

    kernel = np.ones((kernel_size[0], kernel_size[1], 3), np.float32)
    norm = 0
    for i in range(-radius_x, radius_x + 1):
        for j in range(-radius_y, radius_y + 1):
            kernel[i + radius_x, j + radius_y] = math.exp(-(i * i + j * j) /
                                                          (2 * sigma * sigma))
            norm += kernel[i + radius_x, j + radius_y]
    kernel /= norm

    temp = image.copy()
    upper_border = np.repeat(np.expand_dims(temp[0, :], axis=0),
                             radius_x, axis=0)
    lower_border = np.repeat(np.expand_dims(temp[-1, :], axis=0),
                             radius_x, axis=0)
    temp = cv2.vconcat([upper_border, temp, lower_border])
    left_border = np.repeat(np.expand_dims(temp[:, 0], axis=1),
                            radius_y, axis=1)
    right_border = np.repeat(np.expand_dims(temp[:, -1], axis=1),
                             radius_y, axis=1)
    temp = cv2.hconcat([left_border, temp, right_border])

    dst = temp.copy()
    for x in range(radius_x, temp.shape[0] - radius_x):
        for y in range(radius_y, temp.shape[1] - radius_y):
            dst[x, y] = (temp[(x - radius_x):(x + radius_x + 1),
                              (y - radius_y):(y + radius_y + 1)] * kernel).sum(axis=(0, 1))

    return dst[radius_x:-radius_x, radius_y:-radius_y]


def GaussianBlur(image_path: str):
    img = cv2.imread(image_path)
    noise = get_gauss_noise(img)

    start = time.time()
    gauss = cv2.GaussianBlur(img, (3, 3), 1)
    end = time.time()
    print('\nTime:')
    print(end - start)
    print('Compare gauss and noise:')
    print(mean_squared_error(gauss, noise))
    print('Compare gauss and source:')
    print(mean_squared_error(gauss, img))

    start = time.time()
    custom_gauss = gaussian_filter(img, (3, 3), 1)
    end = time.time()
    print('\nTime:')
    print(end - start)
    print('Compare custom_gauss and noise:')
    print(mean_squared_error(custom_gauss, noise))
    print('Compare custom_gauss and source:')
    print(mean_squared_error(custom_gauss, img))

    print('\nCompare gauss and custom_gauss:')
    print(mean_squared_error(gauss, custom_gauss))
    create_two_windows(gauss, custom_gauss)


def blur_filter(image, kernel_size: (int, int)):
    radius_x = kernel_size[0] // 2
    radius_y = kernel_size[1] // 2

    kernel = np.ones((kernel_size[0], kernel_size[1], 3), np.float32)
    norm = kernel_size[0] * kernel_size[1]
    kernel /= norm

    temp = image.copy()
    upper_border = np.repeat(np.expand_dims(temp[0, :], axis=0),
                             radius_x, axis=0)
    lower_border = np.repeat(np.expand_dims(temp[-1, :], axis=0),
                             radius_x, axis=0)
    temp = cv2.vconcat([upper_border, temp, lower_border])
    left_border = np.repeat(np.expand_dims(temp[:, 0], axis=1),
                            radius_y, axis=1)
    right_border = np.repeat(np.expand_dims(temp[:, -1], axis=1),
                             radius_y, axis=1)
    temp = cv2.hconcat([left_border, temp, right_border])

    dst = temp.copy()
    for x in range(radius_x, temp.shape[0] - radius_x):
        for y in range(radius_y, temp.shape[1] - radius_y):
            dst[x, y] = (temp[(x - radius_x):(x + radius_x + 1),
                              (y - radius_y):(y + radius_y + 1)] * kernel).sum(axis=(0, 1))

    return dst[radius_x:-radius_x, radius_y:-radius_y]


def Blur(image_path: str):
    img = cv2.imread(image_path)
    noise = get_gauss_noise(img)

    start = time.time()
    blur = cv2.blur(noise, (3, 3))
    end = time.time()
    print('\nTime:')
    print(end - start)
    print('Compare blur and noise:')
    print(mean_squared_error(blur, noise))
    print('Compare blur and source:')
    print(mean_squared_error(blur, img))

    start = time.time()
    custom_blur = blur_filter(noise, (3, 3))
    end = time.time()
    print('\nTime:')
    print(end - start)
    print('Compare custom_blur and noise:')
    print(mean_squared_error(custom_blur, noise))
    print('Compare custom_blur and source:')
    print(mean_squared_error(custom_blur, img))

    print('\nCompare blur and custom_blur:')
    print(mean_squared_error(blur, custom_blur))
    create_two_windows(blur, custom_blur)


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
