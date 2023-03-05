from torch import Tensor


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
    return x.data.cpu().numpy()
