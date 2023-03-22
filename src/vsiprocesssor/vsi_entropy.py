from skimage.filters.rank import entropy
from skimage.morphology import disk
from cv2 import COLOR_BGR2GRAY, cvtColor


def vsi_has_sufficient_information(image, threshold: float = 3.0, disk_radius: int = 20):
    """
    Checks if the image has sufficient information.
    :param image: the image to check
    :param threshold: the threshold for the entropy
    :param disk_radius: the radius of the disk for the entropy
    :return: True if the image is empty, False otherwise
    """
    return entropy(cvtColor(image, COLOR_BGR2GRAY), disk(disk_radius)).mean() > threshold
