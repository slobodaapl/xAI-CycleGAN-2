import random

from torch import cat, unsqueeze
from torch.autograd import Variable


class LambdaLR:  # Epoch decay, didn't end up using this

    def __init__(self, n_epochs, decay_start_epoch, offset=0):
        assert((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class ImagePool:  # Image pool for fake images
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_images = 0
            self.images = []

    def query(self, images): # Query the pool
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = unsqueeze(image, 0)
            if self.num_images < self.pool_size: # If pool is not full, add image to pool
                self.num_images = self.num_images + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: # Randomly replace image in pool
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(cat(return_images, 0)) # Return images as a tensor
        return return_images
