from torch import Tensor
from kornia.color import lab_to_rgb
from numpy import uint8


class RunningMeanStack(list):

    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def append(self, x):
        super().append(x)
        if len(self) > self.max_length:
            self.pop(0)

    @property
    def mean(self):
        return sum(self) / len(self)

    @property
    def head(self):
        return self[-1]

    @property
    def tail(self):
        return self[0]


def normalize_image(img):
    img = img.cpu().detach()

    # l_mean: float = 50, l_std: float = 29.59, ab_mean: float = 0, ab_std: float = 74.04
    img[0, 0] = img[0, 0] * 29.59 + 50
    img[0, 1] = img[0, 1] * 74.04
    img[0, 2] = img[0, 2] * 74.04

    # clip values
    img[0, 0] = img[0, 0].clamp(0, 100)
    img[0, 1] = img[0, 1].clamp(-128, 127)
    img[0, 2] = img[0, 2].clamp(-128, 127)

    img = lab_to_rgb(img)
    img = img.squeeze(0).permute(1, 2, 0).numpy()
    return img
