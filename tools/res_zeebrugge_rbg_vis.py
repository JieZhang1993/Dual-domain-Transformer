import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomResizedCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

Image.MAX_IMAGE_PIXELS = None

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0])  # label 1
LowVeg = np.array([255, 255, 0])  # label 2
Tree = np.array([0, 255, 0])  # label 3
Car = np.array([0, 255, 255])  # label 4
Clutter = np.array([0, 0, 255])  # label 5
Boundary = np.array([0, 0, 0])  # label 6
num_classes = 7


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="../data/zeebrugge/train_images")
    parser.add_argument("--mask-dir", default="../data/zeebrugge/online_res")
    parser.add_argument("--output-img-dir", default="../data/zeebrugge/train/images_512")
    parser.add_argument("--output-mask-dir", default="../data/zeebrugge/onlineGtRgb")
    parser.add_argument("--eroded", type=bool, default=False)
    parser.add_argument("--gt", type=bool, default=False)  # output RGB mask
    parser.add_argument("--rgb-image", type=bool, default=False)  # use Potsdam RGB format images
    parser.add_argument("--mode", type=str, default='val')
    parser.add_argument("--val-scale", type=float, default=1.0)  # ignore
    parser.add_argument("--split-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    return parser.parse_args()


def label2rgb(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # 定义颜色映射表
    # label_to_color = {
    #     0: [255, 255, 255],
    #     1: [0, 0, 255],
    #     2: [0, 255, 255],
    #     3: [0, 255, 0],
    #     4: [255, 255, 0],
    #     5: [0, 0, 255],
    #     6: [255, 0, 255],
    #     7: [0, 0, 129],
    #     8: [144, 144, 144],
    # }
    # 定义颜色映射表
    label_to_color = {
        # 1: [255, 255, 255],
        # 2: [0, 0, 255],  # [r, g, b] building
        # 3: [0, 255, 255],
        # 4: [0, 255, 0],
        # 5: [255, 255, 0], #yellow car
        # 6: [0, 0, 255],
        # 7: [255, 0, 255],# boat
        # 8: [0, 0, 129],

        1: [255, 255, 255], # white, Impervious Surface
        2: [0, 0, 129],   # [r, g, b] dark blue, Water
        3: [255, 0, 0],   # red, Clutter
        4: [0, 255, 255], # light blue, Low Vegetation
        5: [0, 0, 255],   # blue, Building
        6: [0, 255, 0],  # green, Tree
        7: [255, 0, 255],  # purple, Boat
        8: [255, 255, 0],  # yellow, Car
    }
    # 映射标签到颜色
    for label, color in label_to_color.items():
        rgb[mask == label] = color
    return rgb


def patch_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, rgb_image,
     mode, val_scale, split_size, stride) = inp
    # print(img)
    npf = cv2.imread(mask_path, 0)
    name_gray = os.path.join(masks_output_dir, f"{os.path.basename(mask_path)[:-4]}_gray.jpg")
    cv2.imwrite(name_gray, (npf - 1) * 32)

    rgb = label2rgb(npf)
    name = os.path.join(masks_output_dir, f"{os.path.basename(mask_path)[:-3]}jpg")
    cv2.imwrite(name, rgb[..., ::-1])



if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    eroded = args.eroded
    gt = args.gt
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    img_paths_raw = glob.glob(os.path.join(imgs_dir, "*.tif"))
    # img_paths = [p[:-9] for p in img_paths_raw]
    img_paths = img_paths_raw
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.tif"))

    if eroded:
        mask_paths = [(p[:-21]) for p in mask_paths_raw]
    else:
        # mask_paths = [p[:-10] for p in mask_paths_raw]
        mask_paths = mask_paths_raw

    img_paths.sort()
    mask_paths.sort()
    # print(img_paths[:10])
    # print(mask_paths[:10])
    rgb_image = None
    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        if gt:
            os.makedirs(masks_output_dir + '/origin')

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, rgb_image,
            mode, val_scale, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    for x in inp:
        patch_format(x)
