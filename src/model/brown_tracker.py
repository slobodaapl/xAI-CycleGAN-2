from setup.logging_utils import RunningMeanStackFast
from torch import Tensor


class HistoDataTracker:

    def __init__(self, size_max=500, margin=0.05):
        self.ratio = RunningMeanStackFast(size_max)
        self.margin = margin

        for i in range(2):
            for _ in range(250):
                self.ratio.append(i)

    def check_image(self, imgc: Tensor):
        img = imgc.squeeze(0)
        is_above_zero = ((img[1] + img[2]) / 2).mean() >= 0
        ratio_mean = self.ratio.mean

        if ratio_mean > 0.5 - self.margin and not is_above_zero:
            ratio_mean.append(0)
            return True
        elif ratio_mean < 0.5 + self.margin and is_above_zero:
            ratio_mean.append(1)
            return True

        return False
