import os
import numpy as np
# import matplotlib.pyplot as plt
# import glob
# from tqdm import tqdm
# from multiprocessing import Pool
# from functools import partial
from setup.settings_module import Settings
# from model.macenko import MacenkoNormalizer
from vsiprocesssor.vsi_file import VSIFile
from vsiprocesssor.vsi_entropy import vsi_has_sufficient_information
import cv2
from argparse import ArgumentParser

settings = Settings('settings.cfg')

# Directories for loading data and saving results
data_source = settings.data_source
data_dir = settings.data_root
train_he_dir = os.path.join(data_dir, settings.data_train_he)
train_p63_dir = os.path.join(data_dir, settings.data_train_p63)
test_he_dir = os.path.join(data_dir, settings.data_test_he)
test_p63_dir = os.path.join(data_dir, settings.data_test_p63)
# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(train_he_dir, exist_ok=True)
os.makedirs(train_p63_dir, exist_ok=True)
os.makedirs(test_he_dir, exist_ok=True)
os.makedirs(test_p63_dir, exist_ok=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--start_index', type=int, help='Start index of train_wsis')
    parser.add_argument('--end_index', type=int, help='End index of train_wsis')
    parser.add_argument('--stain', type=str, help='p63 or he')
    parser.add_argument('--type', type=str, help='Train or Test')
    args = parser.parse_args()

    vsi_images = list(map(lambda x: os.path.join(data_source, x),
                      filter(lambda x: not os.path.isdir(x) and x.endswith('.vsi'),
                             os.listdir(data_source))
                          )
                      )
    # Get Test file
    test_he = os.path.join(data_source, settings.test_he)
    test_p63 = os.path.join(data_source, settings.test_p63)

    # Remove test files from dataset
    to_remove = []
    for index in range(len(vsi_images)):
        file = vsi_images[index]
        if test_he in file or test_p63 in file:
            to_remove.append(file)
    # Remove test files from dataset
    for file in to_remove:
        vsi_images.remove(file)

    # Split into HE and p63 WSIs
    vsi_he_images = list(filter(lambda x: 'HE' in x, vsi_images))
    vsi_p63_images = list(filter(lambda x: 'p63' in x, vsi_images))

    np.random.seed(42)

    if len(vsi_he_images) < len(vsi_p63_images):
        select_by = len(vsi_he_images)
        train_by = len(vsi_p63_images)
    else:
        select_by = len(vsi_p63_images)
        train_by = len(vsi_he_images)

    valid_wsis = np.random.randint(0, select_by, int(select_by * 0.2))
    train_wsis = list(set(range(train_by)) - set(valid_wsis))

    if args.type == 'test':
        wsis = valid_wsis
        if args.stain == 'p63':
            stain_dir = test_p63_dir
            vsi_stain_images = vsi_p63_images
        if args.stain == 'he':
            stain_dir = test_he_dir
            vsi_stain_images = vsi_he_images
    if args.type == 'train':
        wsis = train_wsis
        if args.stain == 'p63':
            stain_dir = train_p63_dir
            vsi_stain_images = vsi_p63_images
        if args.stain == 'he':
            stain_dir = train_he_dir
            vsi_stain_images = vsi_he_images

    for index in wsis[args.start_index:args.end_index]:
        try:
            with VSIFile(vsi_stain_images[index]) as vsi:
                # counter = 0
                for (patch, x, y) in vsi:
                    if vsi_has_sufficient_information(patch):
                        cv2.imwrite(os.path.join(train_he_dir, f'{vsi.vsi_name}_{x}_{y}.png'), patch)
                        # counter += 1
                # if counter == 100:
                #     break
        except IndexError as e:
            print(e)
            print(f'Index: {index} not in {wsis} {len(wsis)=} for arguments: {args.start_index} : {args.end_index}')

    # normalization_target = np.load(f'F:/path')
    # rgb_img = normalization_target[:, :, :3]
    # # rgb_img = np.transpose(rgb_img, (2, 0, 1))
    # normalizer = MacenkoNormalizer()
    # normalizer.fit(rgb_img)

    # for mode in ['train', 'test']:
    #     file_list = glob.glob(f'F:/Halinkovic/conic/hovernet_patches/no_overlap/conic/{mode}/256x256_256x256'.npy')
    #     print(f'Processing mode: {mode}')
    #     norm_func = partial(normalize_image, normalizer=normalizer, mode=mode)

    #     with Pool(12) as p:
    #         list(tqdm(p.imap(norm_func, file_list), total=len(file_list)))

# if __name__ == '__main__':
#     # img = np.load('F:/Halinkovic/conic/hovernet_patches/conic/train/256x256_128x128/consep_2_005.npy')
#     # rgb_img = img[:, :, :3]
#     # rgb_img = rgb_img.reshape((3, 256, 256))
#     # stain_matrix = get_stain_matrix(rgb_img)
#     # print(stain_matrix)
#     #
#     # test = get_concentrations(rgb_img, stain_matrix)
#     # print(test)
#
#     # dpath_4_080.npy7
#     # consep_3_006.npy
#     # crag_19_060.npy
#     # glas_59_022.npy
#     # pannuke_9_008.npy
#
#     img = np.load('F:/Halinkovic/conic/hovernet_patches/conic/train/256x256_128x128/dpath_4_080.npy')
#     rgb_img = img[:, :, :3]
#
#     # rgb_img = rgb_img.reshape((3, 256, 256))
#     # rgb_img = np.transpose(rgb_img, (2, 0, 1))
#     print(rgb_img.shape)
#     normalizer = Normalizer()
#     normalizer.fit(rgb_img)
#     target = np.load('F:/Halinkovic/conic//hovernet_patches/conic/train/256x256_128x128/pannuke_9_008.npy')
#     target = target[:, :, :3]
#     out = normalizer.transform(target)
#     plt.imshow(rgb_img)
#     plt.show()
#     plt.imshow(target)
#     plt.show()
#     plt.imshow(out)
#     plt.show()
#     print()
