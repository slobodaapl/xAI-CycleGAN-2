import numpy as np
import torch
import torchvision.transforms.functional as tf
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from typing import Literal
from enum import Enum
from functools import partial


# obtain a noise mask with given mean and std
def get_mask_noise(image, mean=1.0, std=0.02):  # A simple noise mask, with mean and std as parameters
    return torch.normal(torch.full(image.shape, mean), std).cuda()


# obtain a entropy mask with given disk size
def get_mask_entropy(image, disk_size=9):  # A simple entropy mask, with disk size as parameter
    # check if image is tensor, if yes convert to numpy, else leave as is
    if type(image) is torch.Tensor:
        mask = np.zeros_like(image)
        image = tf.rgb_to_grayscale(image)
        image = image.detach().cpu().numpy().astype(np.uint8)

        for i in range(image.shape[0]):
            mask[i] = np.repeat(np.expand_dims(entropy(np.squeeze(image[i]), disk(disk_size)), 0), 3, axis=0)

        mask = (((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) - 0.5) * 2
        return torch.tensor(mask, dtype=torch.float32)

    elif type(image) is np.ndarray:
        image = np.moveaxis(image, 2, 0)
        image = rgb2gray(image, channel_axis=0) * 255
        image = image.astype(np.uint8)
        image = np.repeat(np.expand_dims(entropy(image, disk(disk_size)), 0), 3, axis=0)
        image = ((image - np.min(image))/(np.max(image) - np.min(image)) - 0.5) * 2

        return torch.tensor(image, dtype=torch.float32)

    else:
        exception = Exception('image type is not supported')
        raise exception


# obtain a rectangular binary mask with given ratio of coverage
def get_mask_rec_binary(image, ratio=0.9):  # A simple rectangular binary mask, with ratio controlling the region size
    if type(image) is np.ndarray:
        binary_rec_mask = np.zeros((image.shape[2], image.shape[0], image.shape[1]))
    elif type(image) is torch.Tensor:
        binary_rec_mask = np.zeros((image.shape[1], image.shape[2], image.shape[3]))
    else:
        exception = Exception('image type is not supported')
        raise exception

    left_bound_y = int((1 - ratio) * binary_rec_mask.shape[1])
    up_bound_y = int(ratio * binary_rec_mask.shape[1])
    left_bound_x = int((1 - ratio) * binary_rec_mask.shape[2])
    up_bound_x = int(ratio * binary_rec_mask.shape[2])

    binary_rec_mask[:, left_bound_y:up_bound_y, left_bound_x:up_bound_x] = 1

    return torch.tensor(binary_rec_mask, dtype=torch.float32)


# enum of mask types
class MaskType(Enum):
    binary_rec = partial(get_mask_rec_binary)
    entropy = partial(get_mask_entropy)
    noise = partial(get_mask_noise)


# get mask of specified type
def get_mask(image, mask_type: Literal['binary_rec', 'entropy', 'noise'], options: dict = None):
    """
    This function returns a mask of the specified type.
    :param image: The image to create the mask for.
    :param mask_type: The type of mask to create, e.g. binary_rec, entropy, noise.
    :param options: Settings for the mask type.
    :return: Returns a mask of the specified type.
    """
    if options:
        return MaskType[mask_type].value(image, **options)

    return MaskType[mask_type].value(image)
