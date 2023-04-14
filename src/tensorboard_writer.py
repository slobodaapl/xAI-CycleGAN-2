from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

images = np.load('../data/tf_images.npy')
codes = np.load('../data/pca_results.npy')

writer = SummaryWriter(log_dir='../data/tensorboard')

codes = torch.Tensor(codes)
images = torch.Tensor(images).permute(0, 3, 1, 2)
images /= 255

writer.add_embedding(codes, label_img=images)

writer.close()

print('Done')
