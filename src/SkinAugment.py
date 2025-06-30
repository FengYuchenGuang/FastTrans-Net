# MedEnhance

import albumentations as A
import torch
import math
import random
import os
import cv2
import shutil
import numpy as np
import argparse
from torchvision import transforms
import PIL
from PIL import Image


def npy2png(
        dim=(512, 512),get_dir='../Newdata/npyPic/',save_dir='../Newdata/npyPic/skinaugment/',dataset='isic2016'):
    get_dir = get_dir+dataset+'/'
    save_dir = save_dir+dataset+'/'
    process_name = ["Test", "Train", "Validation"]

    for mkdir in process_name:
        usesave_dir = save_dir
        image_dir_path = get_dir+'{}/Image'.format(mkdir)
        mask_dir_path = get_dir+'{}/Label'.format(mkdir)
        usesave_dir = usesave_dir+mkdir+'/'

        image_path_list = os.listdir(image_dir_path)
        mask_path_list = os.listdir(mask_dir_path)

        image_path_list = list(filter(lambda x: x[-3:] == 'npy', image_path_list))
        mask_path_list = list(filter(lambda x: x[-3:] == 'npy', mask_path_list))

        image_path_list.sort()
        mask_path_list.sort()

        print(len(image_path_list), len(mask_path_list))

        # ISBI Dataset
        for image_path, mask_path in zip(image_path_list, mask_path_list):

            print(image_path)

            assert os.path.basename(image_path) == os.path.basename(mask_path)
            _id = os.path.basename(image_path)[:-4]

            image_path = os.path.join(image_dir_path, image_path)
            mask_path = os.path.join(mask_dir_path, mask_path)

            #load
            data_image = np.load(image_path)
            data_mask = np.load(mask_path)

            data_image = data_image[:, :, ::-1]

            # npy to png
            image_new = Image.fromarray(np.uint8(data_image),mode='RGB')
            mask_new = Image.fromarray(np.uint8(data_mask),mode='L')

            # save
            save_dir_path = usesave_dir + '/Image/'
            os.makedirs(save_dir_path, exist_ok=True)
            image_new.save(os.path.join(save_dir_path, str(_id) + '.png'))

            save_dir_path = usesave_dir + '/Label/'
            os.makedirs(save_dir_path, exist_ok=True)
            mask_new.save(os.path.join(save_dir_path, str(_id) + '.png'))

def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num


def med_augment(data_path, folder_path, name, level, number_branch, mask_i=False, shield=False):
    if mask_i:
        image_path = f"{data_path}{name}/Image/"
        mask_path = f"{data_path}{name}/Label/"

        output_path = f"{folder_path}{name}/Image/"
        out_mask = f"{folder_path}{name}/Label/"
    else:
        image_path = data_path + name
        output_path = folder_path + name+'/'

    transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                 crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 2 * level), 'y': (0, 0)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 0), 'y': (0, 2 * level)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level)
    ])

    for j, file_name in enumerate(os.listdir(image_path)):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(image_path, file_name)
            file_n, file_s = file_name.split(".")[0], file_name.split(".")[1]
            image = cv2.imread(file_path)
            if mask_i: mask = cv2.imread(f"{mask_path}/{file_n}.{file_s}")
            # print('image = ', image.shape)
            # print('mask  = ', mask.shape)
            strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
            for i in range(number_branch):
                if number_branch != 4:
                    employ = random.choice(strategy)
                else:
                    index = random.randrange(len(strategy))
                    employ = strategy.pop(index)
                level, shape = random.sample(transform[:6], employ[0]), random.sample(transform[6:], employ[1])
                img_transform = A.Compose([*level, *shape])
                random.shuffle(img_transform.transforms)
                if not os.path.exists(output_path): os.makedirs(output_path)

                if mask_i:
                    if not os.path.exists(out_mask): os.makedirs(out_mask)
                    transformed = img_transform(image=image, mask=mask)
                    transformed_image, transformed_mask = transformed['image'], transformed['mask']

                    cv2.imwrite(f"{output_path}{file_n}_{i+1}.{file_s}", transformed_image)
                    cv2.imwrite(f"{out_mask}{file_n}_{i+1}_mask.{file_s}", transformed_mask)
                else:
                    transformed = img_transform(image=image)
                    transformed_image = transformed['image']
                    cv2.imwrite(f"{output_path}{file_n}_{i+1}.{file_s}", transformed_image)
                if not shield:

                    cv2.imwrite(f"{output_path}{file_n}_{number_branch+1}.{file_s}", image)
                    if mask_i: cv2.imwrite(f"{out_mask}{file_n}_{number_branch+1}_mask.{file_s}", mask)


def generate_datasets(train_type, dataset, seed, level, number_branch):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    if train_type == "classification":
        print('Executing data augmentation for image classification...')

        data_path = f"../Newdata/ISIC2016/"
        folder_path = f"../Newdata/SA_ISIC2016/"

        os.makedirs(f"{folder_path}", exist_ok=True)
        name = 'Train'
        med_augment(data_path,folder_path, name, level, number_branch)
    else:
        print('Executing data augmentation for image segmentation...')
        data_path = f"../Newdata/npyPic/skinaugment/{dataset}/"
        folder_path = f"../Newdata/npyPic/skinaugment/segmentation/{dataset}/"

        os.makedirs(f"{folder_path}", exist_ok=True)

        # folder_list = ["Test", "Train", "Validation"]
        folder_list = ["Train"]
        for i in range(len(folder_list)):
            name = folder_list[i]
            med_augment(data_path, folder_path, name, level, number_branch, mask_i=True)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_argument_group()
    group.add_argument('--dataset', default='isic2016') # isic2016 isic2017 isic2018
    group.add_argument('--train_type', choices=['classification', 'segmentation'], default='classification')
    group.add_argument('--level', help='Augmentation level', default=5, type=int, metavar='INT')
    group.add_argument('--number_branch', help='Number of branch', default=4, type=int, metavar='INT')
    group.add_argument('--seed', help='Seed', default=8, type=int, metavar='INT')
    args = parser.parse_args()
    generate_datasets(**vars(args))


if __name__ == '__main__':

    main()
