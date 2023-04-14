import numpy as np

from model.model import Generator
from model.mask import get_mask_noise
from model.dataset import DefaultTransform
from setup.logging_utils import normalize_image
from vsiprocesssor.vsi_file import VSIFile
from cv2 import imwrite
import torch

generator = Generator(32, 8)
generator.load_state_dict(torch.load('../data/model_checkpoint_best.pth')['generator_he_to_p63_state_dict'])
generator.eval()
generator.to('cuda')

tf = DefaultTransform()

with torch.no_grad(), VSIFile('../data/raw/4_HE.vsi') as vsi:
    out_img = np.zeros((vsi.max_y_idx * vsi.target_size[0], vsi.max_x_idx * vsi.target_size[1], 3), dtype=np.uint8)
    #out_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    roi_y = vsi.target_size[0]
    roi_x = vsi.target_size[1]

    for roi in vsi:
        img = tf(roi).expand(1, -1, -1, -1).cuda()
        mask = get_mask_noise(img)
        fake = generator(img, mask)
        fake = normalize_image(fake)
        curr_y = (vsi.idx - 1) // vsi.max_x_idx
        curr_x = (vsi.idx - 1) % vsi.max_x_idx

        out_img[curr_y * roi_y:(curr_y + 1) * roi_y, curr_x * roi_x:(curr_x + 1) * roi_x, :] = fake

    imwrite('out.png', out_img)

print("Done!")
