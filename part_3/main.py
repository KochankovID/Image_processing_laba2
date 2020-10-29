import numpy as np
import cv2
import time

import sys


def cv_median_filter(image1) -> None:
    start = time.time()
    new_image = cv2.medianBlur(image1, 3, None)
    end = time.time()
    print('median filter time: ', end - start)
    create_two_windows(image1, new_image, 'original image', 'new image')


def create_two_windows(image1: np.ndarray, image2: np.ndarray,
                       title1: str = 'Left window', title2: str = 'Right window') -> None:
    
    cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(title2, cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow(title1, 100, 0)
    cv2.moveWindow(title2, 705, 0)

    image1 = cv2.resize(image1, dsize=(600, 600))
    image2 = cv2.resize(image2, dsize=(600, 600))

    cv2.imshow(title1, image1)
    cv2.imshow(title2, image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        image_path = sys.argv[1]
    except IndexError:
        print('path to the image is not valid! The default path was set!')
    image = cv2.imread(image_path)
    cv_median_filter(image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
