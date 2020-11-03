import cv2
import numpy as np


def create_two_windows(image1: np.ndarray, image2: np.ndarray,
                       title1: str = 'Left window', title2: str = 'Right window') -> None:
<<<<<<< HEAD
    '''Create two opencv windows with images and titels'''
=======
    """Create two opencv windows with images and titels."""
>>>>>>> main
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
