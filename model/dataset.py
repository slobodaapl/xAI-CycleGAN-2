from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import os
import random


class DefaultTransform:

    def __init__(self, norm_dict=None):
        """
        Serves as the default transform for the dataset, which only
        includes transforms.ToTensor() and transforms.Normalize()

        :param norm_dict: dictionary with mean and std values for normalization
        """

        self.norm_dict = norm_dict
        self.transform = None

        if self.norm_dict is None:
            self.init()

    def __call__(self, img):
        return img

    def init(self):

        if self.norm_dict is None:
            self.norm_dict = {
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]
            }

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.norm_dict)
        ])


class DatasetFromFolder(data.Dataset):
    def __init__(
            self,
            image_dir: str,
            sub_folder: str,
            transform_norm_dict: dict = None,
            transform: DefaultTransform = None,
            resize: int = 256,
            crop_size: int = None,
            flip_h: bool = True,
            flip_v: bool = True
    ):

        """
        Dataset class for loading images from a folder to use while training

        :param image_dir: path to the folder containing the images, ideally with subfolders for train and test
        :param sub_folder: sub-folder to load images from, e.g. 'train/p63'
        :param transform: transform to apply to the images
        :param transform_norm_dict: dictionary with mean and std values for normalization
        :param resize: resize the images to the size of width and height specified by this argument
        :param crop_size: crop the images to the size of width and height specified by this argument
        :param flip_h: flip the images horizontally with a 50% chance
        :param flip_v: flip the images vertically with a 50% chance
        """

        super(DatasetFromFolder, self).__init__()

        self.input_path = os.path.join(image_dir, sub_folder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.resize = resize
        self.crop_size = crop_size
        self.flip_h = flip_h
        self.flip_v = flip_v

        if transform is None:
            self.transform = DefaultTransform(transform_norm_dict)

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn).convert('RGB')

        # preprocessing
        if self.resize:
            img = img.resize((self.resize, self.resize), Image.BILINEAR)

        if self.crop_size:
            x = random.randint(0, self.resize - self.crop_size + 1)
            y = random.randint(0, self.resize - self.crop_size + 1)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))

        if self.flip_h:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.flip_v:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_filenames)

    def get_random_image(self):
        return self.__getitem__(random.randint(0, len(self.image_filenames) - 1))
