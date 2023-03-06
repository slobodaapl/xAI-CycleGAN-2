from torch import Tensor
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


def to_np(x: Tensor):
    return x.data.cpu().detach().numpy()

def normalize_image(img):
    img = img.squeeze()
    img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(uint8)
    return img
