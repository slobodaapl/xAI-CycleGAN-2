"""
Stain normalization based on the method of:

M. Macenko et al., A method for normalizing histology slides for quantitative analysis, in 2009 IEEE International
Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

import numpy as np
from sklearn.decomposition import SparseCoder


def standardize_brightness(image):
    """

    :param image:
    :return:
    """
    p = np.percentile(image, 90)
    return np.clip(image * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(image):
    """
    Remove zeros, replace with 1's.
    :param image: uint8 array
    :return:
    """
    mask = (image == 0)
    image[mask] = 1
    return image


def RGB_to_OD(image):
    """
    Convert from RGB to optical density
    :param image:
    :return:
    """
    image = remove_zeros(image)
    return -1 * np.log(image / 255)


def OD_to_RGB(optical_density_image):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * optical_density_image)).astype(np.uint8)


def normalize_rows(matrix):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return matrix / np.linalg.norm(matrix, axis=1)[:, None]


def get_stain_matrix(image, beta=0.15, alpha=1):
    """
    Get stain matrix (2x3)
    :param image:
    :param beta:
    :param alpha:
    :return:
    """
    # 1. Convert to Optical Density
    optical_density = RGB_to_OD(image).reshape((-1, 3))

    # 2. Remove OD less than beta
    optical_density = (optical_density[(optical_density > beta).any(axis=1), :])

    # 3.1 Get eigenvectors
    _, eigen_vectors = np.linalg.eigh(np.cov(optical_density, rowvar=False))

    # 3.2 Make sure the vectors are ponting the right way
    eigen_vectors = eigen_vectors[:, [2, 1]]
    if eigen_vectors[0, 0] < 0:
        eigen_vectors[:, 0] *= -1
    if eigen_vectors[0, 1] < 0:
        eigen_vectors[:, 1] *= -1

    # 4. Project
    That = np.dot(optical_density, eigen_vectors)

    # 5. Calculate angle of each point in respect to the vector directions
    phi = np.arctan2(That[:, 1], That[:, 0])

    # 6. Find extremes
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    # 7. Min-max vectors coreesponding to Haematoxylin and Eosin
    v1 = np.dot(eigen_vectors, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(eigen_vectors, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # 8. Order H fist, E second
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])

    return normalize_rows(HE)


def get_concentrations(image, stain_matrix, lamda=0.01):
    """
    Get concentrations, a npix x 2 matrix
    :param image:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    optical_density = RGB_to_OD(image).reshape((-1, 3))
    coder = SparseCoder(dictionary=stain_matrix,
                        transform_algorithm='lasso_lars',
                        positive_code=True,
                        transform_alpha=lamda)
    return coder.transform(optical_density)


class MacenkoNormalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target):
        target = standardize_brightness(target)
        # Steps 1. - 8. of the Macenko algorithm
        self.stain_matrix_target = get_stain_matrix(target)

        # 9. Determine concentrations of individual stains
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)

    def target_stains(self):
        return OD_to_RGB(self.stain_matrix_target)

    def transform(self, image):
        image = standardize_brightness(image)
        stain_matrix_source = get_stain_matrix(image)
        source_concentrations = get_concentrations(image, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (maxC_target / maxC_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(image.shape))).astype(
            np.uint8)

    def hematoxylin(self, image):
        image = standardize_brightness(image)
        h, w, _ = image.shape
        stain_matrix_source = get_stain_matrix(image)
        source_concentrations = get_concentrations(image, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H
