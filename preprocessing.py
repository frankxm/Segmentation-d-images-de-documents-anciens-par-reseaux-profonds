# -*- coding: utf-8 -*-

"""
    The preprocessing module
    ======================

    Use it to preprocess the images.
"""

import os
import shutil

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from doc_functions import rgb_to_gray_array, rgb_to_gray_value
import random
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json

from shapely.geometry import Polygon, MultiPolygon
import statistics
import math
from collections import Counter

def compute_class_pixel_ratios(dataset, num_classes,startindex,length,typeimage):
    # 初始化一个数组，用来存放每个类别的像素总数
    class_pixel_counts = torch.zeros(num_classes)
    total_pixels = 0

    # 确保 length 不超过数据集长度
    if length is None or length + startindex > len(dataset):
        length = len(dataset) - startindex

    for i in tqdm(range(startindex, startindex + length), desc="Calculate pixel ratio of the class in data",
                  total=length):
        # 获取指定索引的样本
        sample = dataset[i]
        if sample['type'] is not typeimage:
            continue
        labels = sample['mask']
        # 将标签转换为整数类型 (torch.int64)
        labels = torch.from_numpy(labels).long()
        # 使用 bincount 统计当前标签图像中每个类别的像素数 minlength=num_classes 确保即使某些类没有像素，它们的计数仍然返回为 0。
        pixel_counts = torch.bincount(labels.view(-1), minlength=num_classes)

        # 累加每个类别的像素数
        class_pixel_counts += pixel_counts

        # 计算总像素数
        total_pixels += labels.numel()

    # 计算每个类别的像素占比
    class_pixel_ratios = class_pixel_counts / total_pixels

    return class_pixel_ratios
class TrainingDataset(Dataset):

    def __init__(
            self, augment_all: list, colors: list,transform: list = None,augmentations_transformation: list = None, augmentations_pixel: list = None,forbid=False
    ):
        self.images = [sample["image"] for sample in augment_all]
        self.masks = [sample["mask"] for sample in augment_all]
        self.type_list=[sample["type"] for sample in augment_all]
        self.masks_binary= [sample["mask_binary"] for sample in augment_all]
        self.colors = colors
        self.transform = transform
        self.augmentations_transformation = augmentations_transformation if augmentations_transformation else []
        self.augmentations_pixel = augmentations_pixel if augmentations_pixel else []
        self.forbid=forbid

    def __len__(self) -> int:

        return len(self.images)

    def __getitem__(self, idx: int) -> dict:

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.masks[idx]
        label_binary=self.masks_binary[idx]
        type=self.type_list[idx]
        sample = {"image": image, "mask": label, "mask_binary":label_binary,"size": image.shape[0:2],"type":type}


        if not self.forbid and self.augmentations_transformation and self.augmentations_pixel:
            # 增强的概率为0.5
            if random.random() < 0.5:
                # 两类增强方式概率0.5
                if random.random() < 0.5:
                    aug = random.choice(self.augmentations_transformation)
                    image, label ,label_binary= aug(image, label,label_binary)
                    logging.info(f'operation {aug.__name__} for current image,label')
                else:
                    aug = random.choice(self.augmentations_pixel)
                    image = aug(image)
                    logging.info(f'operation {aug.__name__} for current image')
                # plt.close()
                # plt.figure(figsize=(15, 8))
                # plt.subplot(3,2,1)
                # plt.imshow(sample["image"])
                # plt.subplot(3,2,2)
                # plt.imshow(gray_array_to_rgb(sample["mask"]))
                # plt.subplot(3,2,3)
                # plt.imshow(image)
                # plt.subplot(3,2,4)
                # plt.imshow(gray_array_to_rgb(label))
                # plt.subplot(3, 2, 5)
                # plt.imshow(sample["mask_binary"], cmap='gray')
                # plt.subplot(3, 2, 6)
                # plt.imshow(label_binary,cmap='gray')
                # plt.savefig(f'./Augmentation/{aug.__name__}_{idx}.png')

                sample["image"] = image
                sample["mask"] = label
                sample["mask_binary"] = label_binary

                # all_images = [image for image in self.images]
                # mean, std = compute_mean_std(all_images, batch_size=100)  # 使用分批次计算均值和标准差
                # means.append(mean)
                # stds.append(std)


        #将灰度值改为类别值
        unique_values, counts = np.unique(label, return_counts=True)
        value_counts = dict(zip(unique_values, counts))
        new_label = np.ones_like(sample["mask"])*4
        graycolors = [rgb_to_gray_value(value) for index, value in enumerate(self.colors)]
        for gray_index, gray_color in enumerate(graycolors):
            new_label[label == gray_color] = gray_index
        sample["mask"] = new_label

        if self.transform:
            sample = self.transform(sample)

        return sample




class PredictionDataset(Dataset):
    def __init__(self, data, transform: list = None):

        self.images = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.images[idx]

        sample = {
            "image": img["image"],
            "name": img["name"].name,
            "dataset": img["dataset"],
            "size": img["size"],
            "position":img["position"]
        }

        if self.transform:
            sample = self.transform(sample)
        return sample




class Rescale:
    """
    The Rescale class is used to rescale the image of a sample into a
    given size.
    """

    def __init__(self, output_size: int):
        """
        Constructor of the Rescale class.
        :param output_size: The desired new size.
        """
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        """
        Rescale the sample image into the model input size.
        :param sample: The sample to rescale.
        :return sample: The rescaled sample.
        """
        old_size = sample["image"].shape[:2]
        # Compute the new sizes.
        ratio = float(self.output_size) / max(old_size)
        new_size = [int(x * ratio) for x in old_size]

        # Resize the image.
        if max(old_size) != self.output_size:
            image = cv2.resize(sample["image"], (new_size[1], new_size[0]))
            sample["image"] = image

        # Resize the label. MUST BE AVOIDED.
        if "mask" in sample.keys():
            if max(sample["mask"].shape[:2]) != self.output_size:
                mask = cv2.resize(sample["mask"], (new_size[1], new_size[0]))
                sample["mask"] = mask
        return sample


class Pad:
    """
    The Pad class is used to pad the image of a sample to make it divisible by 8.
    保持图像的大致宽高比并将其尺寸调整为8的倍数，通过适当的padding来实现，这是为了确保在处理过程中不会显著改变图像的原始比例，同时又能满足计算要求
    """

    def __init__(self):
        """
        Constructor of the Pad class.
        """
        pass

    def __call__(self, sample: dict) -> dict:
        """
        Pad the sample image with zeros.
        :param sample: The sample to pad.
        :return sample: The padded sample.
        """
        # Compute the padding parameters.
        delta_w = 0
        delta_h = 0
        if sample["image"].shape[0] % 8 != 0:
            delta_h = (
                    int(8 * np.ceil(sample["image"].shape[0] / 8))
                    - sample["image"].shape[0]
            )
        if sample["image"].shape[1] % 8 != 0:
            delta_w = (
                    int(8 * np.ceil(sample["image"].shape[1] / 8))
                    - sample["image"].shape[1]
            )

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Add padding to have same size images.
        image = cv2.copyMakeBorder(
            sample["image"],
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        sample["image"] = image
        sample["padding"] = {"top": top, "left": left,"bottom":bottom,"right":right}
        return sample


class Normalize:
    """
    The Normalize class is used to normalize the image of a sample.
    The mean value and standard deviation must be first computed on the
    training dataset.
    """

    def __init__(self, mean: list, std: list):
        """
        Constructor of the Normalize class.
        :param mean: The mean values (one for each channel) of the images
                     pixels of the training dataset.
        :param std: The standard deviations (one for each channel) of the
                    images pixels of the training dataset.
        """
        assert isinstance(mean, list)
        assert isinstance(std, list)
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        # 在归一化前不能先设定image类型为uin8，因为在计算过程中用的是浮点计算，而最后又被转成uint8，这样就会使得数值会被截取在0-255之间。从而影响后续逆归一化图像的显示
        image = np.zeros(sample["image"].shape,dtype=np.float64)
        for channel in range(sample["image"].shape[2]):
            image[:, :, channel] = (
                                           np.float64(sample["image"][:, :, channel]) - self.mean[channel]
                                   ) / self.std[channel]

        sample["image"] = image
        return sample


class ToTensor:
    """
    The ToTensor class is used convert ndarrays into Tensors.
    """

    def __call__(self, sample: dict) -> dict:
        """
        Transform the sample image and label into Tensors.
        :param sample: The initial sample.
        :return sample: The sample made of Tensors.
        """
        sample["image"] = torch.from_numpy(sample["image"].transpose((2, 0, 1)))
        if "mask" in sample.keys():
            sample["mask"] = torch.from_numpy(sample["mask"])
        return sample


# 将图像的四个顶点随机移动，从而改变图像的透视效果
def random_perspective_transform(image, mask,mask_binary):
    height, width = image.shape[:2]
    src_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])
    # 在大小的正百分之十和负百分之十之间变换
    max_shift = min(height, width) // 10
    delta = random.randint(-max_shift, max_shift)
    dst_points = np.float32([
        [delta, delta],
        [width - 1 - delta, delta],
        [width - 1 - delta, height - 1 - delta],
        [delta, height - 1 - delta]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    image = cv2.warpPerspective(image, matrix, (width, height))
    mask = cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    mask[mask == 0] = 127
    mask_binary = cv2.warpPerspective(mask_binary, matrix, (width, height), flags=cv2.INTER_NEAREST)
    mask_binary[mask_binary == 0] = 255
    return image, mask,mask_binary


# 对图像进行随机弹性变形，使图像看起来像被拉伸或挤压

def random_elastic_transform(image, mask,mask_binary, alpha=20, sigma=3):
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]  # 只取图像的高度和宽度
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # 对每个通道分别进行弹性变形
    transformed_image = np.zeros_like(image)
    for c in range(image.shape[2]):
        transformed_image[..., c] = map_coordinates(image[..., c], indices, order=1, mode='reflect').reshape(shape)
    transformed_mask = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
    transformed_mask_binary = map_coordinates(mask_binary, indices, order=1, mode='reflect').reshape(shape)

    return transformed_image, transformed_mask,transformed_mask_binary




# 使用一个结构化元素来腐蚀图像中的前景对象，使它们变得更小
def random_ouverture(image, mask,mask_binary):
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    return image, mask,mask_binary



# 随机旋转图像和掩码，使它们倾斜一定的角度。
def random_rotate(image, mask,mask_binary):
    angle = random.uniform(-10, 10)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, rot_mat, mask.shape[1::-1], flags=cv2.INTER_NEAREST)
    mask[mask == 0] = 127
    mask_binary = cv2.warpAffine(mask_binary, rot_mat, mask_binary.shape[1::-1], flags=cv2.INTER_NEAREST)
    mask_binary[mask_binary == 0] = 255
    return image, mask,mask_binary


# 随机水平翻转图像和掩码
def random_flip(image, mask,mask_binary):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        mask_binary = cv2.flip(mask_binary, 1)
    return image, mask,mask_binary


# 平移图像和掩码
def random_shift(image, mask, mask_binary,max_shift=30):
    rows, cols = image.shape[:2]
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    shift_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, shift_matrix, (cols, rows))
    mask = cv2.warpAffine(mask, shift_matrix, (cols, rows))
    mask[mask == 0] = 127
    mask_binary = cv2.warpAffine(mask_binary, shift_matrix, (cols, rows))
    mask_binary[mask_binary == 0] = 127
    return image, mask,mask_binary

# 随机调整图像的亮度、对比度和颜色饱和度。
def random_color_jitter(image):
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    return np.array(image)


# 对图像应用高斯模糊，使图像变得模糊
# 高斯模糊只是针对图像的局部细节进行模糊化，不会显著改变全局的亮度或对比度，因此不会明显影响均值和方差。
def random_gaussian_blur(image):
    image = Image.fromarray(image)
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))
    return np.array(image)


# 向图像添加高斯噪声，使图像看起来更嘈杂。
# 当 var 较大时，图像可能会变得过于嘈杂，从而改变其整体的均值和方差
def random_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    image = image + gauss
    return np.clip(image, 0, 255).astype(np.uint8)


# 增强图像的锐度，使图像中的边缘变得更加清晰
# 锐化主要会增强边缘细节，但不会显著改变图像的全局亮度、对比度等特性。因此，对图像均值和方差的影响较小。
def random_sharpen(image):
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(random.uniform(1.5, 3))
    return np.array(image)

# 图像对比度增强  对比度调整的范围缩小至 0.8 - 1.2，这样增强操作不会对数据的方差产生剧烈变化。
def random_contrast(image):
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    return np.array(image)


def gray_array_to_rgb(mask):
    color_map = {
        76: [255, 0, 0],  # Blue
        149: [0, 255, 0],  # Green
        29: [0, 0, 255],  # Red
        225: [255, 255, 0],  # Cyan
        105: [255, 0, 255],  # Magenta
        178: [0, 255, 255],  # Yellow
        127: [128, 128, 128]  # Gray
    }
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for gray_value, rgb_color in color_map.items():
        rgb_image[mask == gray_value] = rgb_color

    return rgb_image


def combine_class(mask):
    # rgb
    colors = [
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [128, 128, 128]
    ]
    graycolormap = {
        0: 29,  # Blue
        1: 149,  # Green
        2: 76,  # Red
        3: 178,  # Cyan
        4: 105,  # Magenta
        5: 225,  # Yellow
        6: 127  # Gray
    }
    graycolors = [rgb_to_gray_value(value) for index, value in enumerate(colors)]
    # 算差值，并找出差值最小的索引（防止mask中灰度值有偏差的情况）
    differences = np.abs(mask[:, :, None] - graycolors)
    new_label = np.argmin(differences, axis=2)
    # np.argmin 的返回值为索引，索引的默认数据类型是 int64需转换
    new_label = new_label.astype(np.uint8)
    new_label[(new_label == 3) | (new_label == 4) | (new_label == 5)] = 2
    for gray_index, gray_color in graycolormap.items():
        new_label[new_label == gray_index] = gray_color
    return new_label



# 背景类不要占过多，白色背景部分不要过多
def is_valid_crop(cropped_img,cropped_mask,min_value_percentage):
    num_pixels = cropped_mask.size
    num_127_pixels = np.sum(cropped_mask == 127)
    value_percentage = num_127_pixels / num_pixels
    flag1=value_percentage <= min_value_percentage
    return flag1
# 黑色像素要多
def random_crop(image, mask, crop_size, num_crops, min_value_percentage,i):
    img_height, img_width = image.shape[:2]
    crop_width, crop_height = crop_size
    mask_combined = combine_class(mask)

    augmented_data = []
    crops = []
    while len(augmented_data) < num_crops:
        candidates = [(random.randint(0, img_width - crop_width), random.randint(0, img_height - crop_height)) for _ in
                      range(num_crops * 5)]

        for x, y in candidates:
            cropped_img = image[y:y + crop_height, x:x + crop_width]
            cropped_mask = mask_combined[y:y + crop_height, x:x + crop_width]

            if is_valid_crop(cropped_img,cropped_mask,min_value_percentage):
                crops.append((cropped_img, cropped_mask))
                if len(crops) >= num_crops:
                    break
        if len(crops) >= num_crops:
            break

    for count, (cropped_img, cropped_mask) in enumerate(crops):
        cropped_mask_rgb = gray_array_to_rgb(cropped_mask)
        augmented_data.append({"image": cropped_img, "mask": cropped_mask, "type": 'real'})

    #     plt.close()
    #     plt.figure(figsize=(15, 8))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(cropped_img)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(cropped_mask_rgb)
    #     plt.savefig(f'./Augmentation/random_crop_index{i}_{count}.png')
    logging.info(f'create {len(augmented_data)} random cropped images')

    return augmented_data
def shift_crop(image, mask, crop_size,shift_value, i):
    img_height, img_width = image.shape[:2]
    crop_width, crop_height = crop_size
    shift_right,shift_down=shift_value
    mask_combined = combine_class(mask)

    augmented_data = []
    count = 0

    for y in range(shift_down, img_height, crop_height):
        for x in range(shift_right, img_width, crop_width):
            x_end = x + crop_width
            y_end = y + crop_height

            if x_end > img_width or y_end > img_height:
                pad_x = max(0, x_end - img_width)
                pad_y = max(0, y_end - img_height)

                cropped_img = image[y:min(y_end, img_height), x:min(x_end, img_width)]
                cropped_mask = mask_combined[y:min(y_end, img_height), x:min(x_end, img_width)]
                mask_binary_all = otsu_binary(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
                mask_binary = mask_binary_all[y:min(y_end, img_height), x:min(x_end, img_width)]
                # cropped_img 是一个 3D 图像数组，通常形状为 (height, width, channels)。
                # ((0, pad_y), (0, pad_x), (0, 0)) 表示：
                # 对第一个维度（高度 height）：在上边填充 0 个像素，在下边填充 pad_y 个像素。
                # 对第二个维度（宽度 width）：在左边填充 0 个像素，在右边填充 pad_x 个像素。
                # 对第三个维度（通道 channels，例如 RGB 通道）：在两端都不填充（填充 0 个像素）。
                # mode='constant'：表示使用常数值进行填充。
                # constant_values=0：表示填充值为 0，即零填充
                cropped_img = np.pad(cropped_img, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=0)
                cropped_mask = np.pad(cropped_mask, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=4)
                mask_binary = np.pad(mask_binary, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=255)
            else:

                cropped_img = image[y:y + crop_height, x:x + crop_width]
                cropped_mask = mask_combined[y:y + crop_height, x:x + crop_width]
                mask_binary_all = otsu_binary(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
                mask_binary = mask_binary_all[y:y + crop_height, x:x + crop_width]

            cropped_mask_rgb = gray_array_to_rgb(cropped_mask)
            # 将二值图像的所有像素设置为白色（255）如果所有值相同
            if np.min(mask_binary) == np.max(mask_binary):
                # 设置整个数组为白色
                mask_binary[:] = 255

            augmented_data.append({"image": cropped_img, "mask": cropped_mask, "mask_binary": mask_binary, "type": 'real'})
            plt.close()
            plt.figure(figsize=(15, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(cropped_img)
            plt.subplot(1, 3, 2)
            plt.imshow(cropped_mask_rgb)
            plt.subplot(1, 3, 3)
            plt.imshow(mask_binary,cmap='gray')
            plt.savefig(f'./Augmentation/shift_crop_right_index{i}_{count}.png') if shift_right>shift_down else plt.savefig(f'./Augmentation/shift_crop_down_index{i}_{count}.png')
            count += 1
    logging.info(f'create {len(augmented_data)} shift_right cropped images') if shift_right>shift_down else logging.info(f'create {len(augmented_data)} shift_down cropped images')
    return augmented_data
def standard_crop(image, mask, crop_size,i):
    img_height, img_width = image.shape[:2]
    crop_width, crop_height = crop_size
    mask_combined = combine_class(mask)

    augmented_data = []
    count=0
    for y in range(0, img_height, crop_height):
        for x in range(0, img_width, crop_width):

            if x + crop_width > img_width:
                x = img_width - crop_width
            if y + crop_height > img_height:
                y = img_height - crop_height

            cropped_img = image[y:y + crop_height, x:x + crop_width]
            cropped_mask = mask_combined[y:y + crop_height, x:x + crop_width]
            cropped_mask_rgb = gray_array_to_rgb(cropped_mask)
            mask_binary_all=otsu_binary( cv2.cvtColor(image,cv2.COLOR_RGB2GRAY))
            mask_binary=mask_binary_all[y:y + crop_height, x:x + crop_width]
            if np.min(mask_binary) == np.max(mask_binary):
                # 设置整个数组为白色
                mask_binary[:] = 255
            augmented_data.append({"image": cropped_img, "mask": cropped_mask, "mask_binary":mask_binary,"type": 'real'})
            # plt.close()
            # plt.figure(figsize=(15, 8))
            # plt.subplot(1, 3, 1)
            # plt.imshow(cropped_img)
            # plt.subplot(1, 3, 2)
            # plt.imshow(cropped_mask_rgb)
            # plt.subplot(1, 3, 3)
            # plt.imshow(mask_binary,cmap='gray')
            # plt.savefig(f'./Augmentation/standard_crop_index{i}_{count}.png')

            count+=1


    logging.info(f'create {len(augmented_data)} standard cropped images')

    return augmented_data


def otsu_binary(img):
    _, img_binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(img_binary_otsu, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_binary_otsu)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask_binary = np.where(mask == 255, img_binary_otsu, 255)
    return mask_binary

def readImageMask(image_folder,json_folder,bgrdir,output_size):
    image_mask_bgr_element_dict={}

    for filename in os.listdir(image_folder):
        polygons_dict = {}
        polygons_data = []
        all_elements = []
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            width, height = image.shape[1], image.shape[0]
            if image is None:
                continue
            base_name, _ = os.path.splitext(filename)
            bgrpath = os.path.join(bgrdir, 'background_' + base_name + '.jpg')
            background = cv2.imread(bgrpath)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            bgr_resized = cv2.resize(background, (output_size, output_size))
            json_path = os.path.join(json_folder, f"{base_name}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as f:
                annotations = json.load(f)

            for shape in annotations['shapes']:

                element = (image_path, shape)
                all_elements.append(element)

                polygon_points = np.array(shape['points'], np.int32)
                polygon = Polygon(polygon_points)

                label = shape['label'].lower()

                if label not in polygons_dict:
                    polygons_dict[label] = []

                polygons_dict[label].append((polygon, base_name, polygon_points))
                polygons_data.append((label, polygon, base_name, polygon_points))

            # 定义颜色映射，假设有六个类和一个背景 BGR
            color_map = {
                'texte': [255, 0, 0],  # Blue
                'figure': [0, 255, 0],  # Green
                'math': [0, 0, 255],  # Red
                'mathstructuree': [255, 255, 0],  # Cyan
                'textemath': [255, 0, 255],  # Magenta
                'mathbarree': [0, 255, 255],  # Yellow
                'background': [128, 128, 128]  # Gray
            }
            mask_structures = np.zeros((height, width, 3), dtype=np.uint8)
            mask_structures[:, :] = color_map['background']
            intersection_data = []


            for i in range(0, len(polygons_data) - 1):
                for j in range(i + 1, len(polygons_data)):
                    currentpolygon = polygons_data[i][1]
                    currentlabel = polygons_data[i][0]
                    currentpolypoints = polygons_data[i][3]
                    otherpolygon = polygons_data[j][1]
                    otherlabel = polygons_data[j][0]
                    otherpolypoints = polygons_data[j][3]
                    cv2.fillPoly(mask_structures, [currentpolypoints], color_map[currentlabel])
                    cv2.fillPoly(mask_structures, [otherpolypoints], color_map[otherlabel])
                    # cv2.polylines(image, [currentpolypoints], isClosed=True, color=color_map[currentlabel], thickness=5)
                    # cv2.polylines(image, [otherpolypoints], isClosed=True, color=color_map[otherlabel], thickness=5)
                    if currentpolygon.intersects(otherpolygon):
                        intersection = currentpolygon.intersection(otherpolygon)
                        area_intersection = intersection.area
                        area_polygon = currentpolygon.area
                        area_other_polygon = otherpolygon.area
                        proportion_polygon = area_intersection / area_polygon
                        proportion_other = area_intersection / area_other_polygon
                        if isinstance(intersection, MultiPolygon):
                            for poly in intersection.geoms:
                                if isinstance(poly, Polygon):
                                    intersection_points = np.array(poly.exterior.coords, dtype=np.int32)
                        elif isinstance(intersection, Polygon):
                            intersection_points = np.array(intersection.exterior.coords, dtype=np.int32)
                        if proportion_polygon > proportion_other:
                            intersection_data.append((currentlabel, intersection_points))

                        else:
                            intersection_data.append((otherlabel, intersection_points))


            for label, polypoints in intersection_data:
                cv2.fillPoly(mask_structures, [polypoints], color_map[label])
                # cv2.polylines(image, [polypoints], isClosed=True, color=color_map[label], thickness=5)
            mask_structures = cv2.cvtColor(mask_structures, cv2.COLOR_BGR2RGB)
            # combine = cv2.addWeighted(image, 0.7, mask_structures, 0.3, 0)

            current_elements = select_elements(all_elements)
            image_mask_bgr_element_dict[base_name]=(image,mask_structures,bgr_resized,current_elements)
    return image_mask_bgr_element_dict



def compute_mean_std(image_list, batch_size=100):
    num_images = len(image_list)
    means = []
    stds = []

    for i in range(0, num_images, batch_size):
        batch = image_list[i:i + batch_size]
        batch_images = np.stack([np.array(img, dtype=np.float32) for img in batch])

        mean = np.mean(batch_images, axis=(0, 1, 2))
        std = np.std(batch_images, axis=(0, 1, 2))

        means.append(mean)
        stds.append(std)

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std


# 验证集裁剪版本
def apply_augmentations_and_compute_stats(imagedir, maskdir, jsondir, bgrdir, output_size, set, generated_images,start_ratio):
    image_mask_bgr_elements_dict = readImageMask(str(imagedir), str(jsondir), bgrdir, output_size)

    means = []
    stds = []
    augment_all = []
    if os.path.exists('./Augmentation'):
        shutil.rmtree('./Augmentation')
    os.makedirs('./Augmentation')

    imagenamelist = [os.path.splitext(i)[0] for i in os.listdir(str(imagedir))]
    # 对验证集先固定位置裁剪再随机裁剪
    if set == "val":
        for i, name in enumerate(tqdm(imagenamelist, desc="Calcule mean aug des images validation")):
            sample = image_mask_bgr_elements_dict[name]
            image = sample[0]
            label = sample[1]
            label = rgb_to_gray_array(label)

            standard_samples = standard_crop(image, label, (output_size, output_size), i)
            shift_samples_right = shift_crop(image, label, (output_size, output_size), (int(output_size / 2), 0), i)
            shift_samples_down = shift_crop(image, label, (output_size, output_size), (0, int(output_size / 2)), i)
            augment_all.extend(standard_samples + shift_samples_right + shift_samples_down)
            # 计算每个批次的均值和标准差
            # all_images = [data["image"] for data in (standard_samples + shift_samples_right + shift_samples_down)]
            # mean, std = compute_mean_std(all_images, batch_size=100)  # 使用分批次计算均值和标准差
            # means.append(mean)
            # stds.append(std)

        return augment_all
    # 对训练集采用多种增强手段  标准裁剪+随机增强+合成图像
    else:
        for i, name in enumerate(tqdm(imagenamelist, desc="Augmente Images originales")):
            sample = image_mask_bgr_elements_dict[name]
            image = sample[0]
            label = sample[1]
            label = rgb_to_gray_array(label)

            # 执行标准裁剪 右平移裁剪 下平移裁剪
            standard_samples = standard_crop(image, label, (output_size, output_size), i)
            shift_samples_right = shift_crop(image, label, (output_size, output_size), (int(output_size / 2), 0), i)
            shift_samples_down = shift_crop(image, label, (output_size, output_size), (0, int(output_size / 2)), i)
            augment_all.extend(standard_samples + shift_samples_right + shift_samples_down)

            # 计算每个批次的均值和标准差
            all_images = [data["image"] for data in (standard_samples + shift_samples_right + shift_samples_down)]
            # 使用分批次计算均值和标准差
            mean, std = compute_mean_std(all_images, batch_size=100)
            means.append(mean)
            stds.append(std)

        if generated_images:
            num_synthetique = int(len(augment_all) * (1 - start_ratio))
            augment_syn = []

            for index in tqdm(range(num_synthetique), desc="Augmente Images synthetiques"):
                # 对训练集的所有图像循环粘贴
                i = index % len(imagenamelist)
                name = imagenamelist[i]
                # 背景缺块过多
                if name == 'img_LH_35_8_1_6v':
                    continue
                augmented_samples = place_elements_without_collision(image_mask_bgr_elements_dict[name], index)
                augment_all.append(augmented_samples)
                augment_syn.append(augmented_samples)

            all_images = [data["image"] for data in augment_syn]
            mean, std = compute_mean_std(all_images, batch_size=100)  # 使用分批次计算均值和标准差
            means.append(mean)
            stds.append(std)

        mean_final = np.mean(means, axis=0)
        std_final = np.mean(stds, axis=0)
        logging.info(" Mean after augmentation in {}: {}".format(set, np.uint8(mean_final)))
        logging.info(" Std after augmentation in {}: {}".format(set, np.uint8(std_final)))

        return augment_all, np.uint8(mean_final), np.uint8(std_final)



def select_elements(all_elements):
    random.shuffle(all_elements)
    current_element={}
    math_elements = []
    texte_elements = []
    figure_elements=[]
    for img_path, annotation in all_elements:
        label = annotation['label'].lower()
        if 'texte' ==label :
            texte_elements.append((img_path, annotation))
        elif 'math' in label :
            math_elements.append((img_path, annotation))
        elif 'figure' in label :
            figure_elements.append((img_path, annotation))
        if len(texte_elements) + len(math_elements) + len(figure_elements)==len(all_elements):
            break
    current_element['texte']=texte_elements
    current_element['math']=math_elements
    current_element['figure']=figure_elements
    return current_element

def get_element_math_figure(elements,area_current,background,image,remaining_width,remaining_height):
    elements_dict={'math':[],'figure':[]}
    namelist=['math','figure']
    for name in namelist:
        for img_path, element in elements[name]:
            points = np.array(element['points'], dtype=np.int32)
            rect = cv2.boundingRect(points)
            x, y, w, h = rect
            area = w*h
            # 防止起始点超出图像,防止过大的多边形图像缩小后内容不清晰，防止选取的多边形过宽过高，超出背景
            if x < image.shape[1] and y < image.shape[0] and area<area_current*1.5 and w<background.shape[1] and h<background.shape[0] :
                if w >= remaining_width or h >= remaining_height:
                    if w > h:
                        scale = remaining_width / w
                    else:
                        scale = remaining_height / h
                else:
                    scale = 1

                # 根据比例缩放多边形尺寸
                newwidth = math.ceil(w * scale)
                newheight = math.ceil(h * scale)

                if newheight < remaining_height and newwidth<remaining_width:
                    elements_dict[name].append((element,newwidth,newheight))

    return elements_dict

def get_element_texte(image, mask, width, height):

    blue_mask = np.all(mask == [0, 0, 255], axis=-1)  # 找到符合条件的区域
    # 获取蓝色区域的坐标
    blue_coords = np.column_stack(np.where(blue_mask))


    # 如果没有蓝色区域，返回
    if blue_coords.size == 0:
        return None, True,None,None

    # 随机选择一个蓝色区域中的点,replace=false不重复
    rand_index = np.random.choice(blue_coords.shape[0],replace=False)
    rand_point = blue_coords[rand_index]
    rand_y,rand_x=rand_point

    # 尝试将该点分别作为左上角、右上角、右下角、左下角，尝试取给定大小的矩形
    potential_rectangles = [
        (rand_x, rand_y, rand_x + width, rand_y + height),  # 作为左上角
        (rand_x - width, rand_y, rand_x, rand_y + height),  # 作为右上角
        (rand_x - width, rand_y - height, rand_x, rand_y),  # 作为右下角
        (rand_x, rand_y - height, rand_x + width, rand_y)  # 作为左下角
    ]

    # 遍历所有可能的矩形
    for rect in potential_rectangles:
        x1, y1, x2, y2 = rect

        # 判断矩形是否超出图像边界
        if x1 < 0 or y1 < 0 or x2 > mask.shape[1] or y2 > mask.shape[0]:
            continue

        # 检查矩形区域是否全部为蓝色
        selected_mask = blue_mask[y1:y2, x1:x2]
        if np.all(selected_mask):
            points = [
                # 左上角
                [x1, y1],
                # 左下角
                [x1, y2],
                # 右下角
                [x2, y2],
                # 右上角
                [x2, y1]
            ]

            selected_section = image[y1:y2, x1:x2]

            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(selected_section)
            # plt.subplot(1, 3, 2)
            # plt.imshow(mask[y1:y2,x1:x2])
            # plt.subplot(1,3,3)
            # plt.imshow(mask)
            # plt.show()

            # 返回找到的区域和中心点
            return {'points': points}, False,x2-x1,y2-y1

    # 如果没有找到符合要求的区域，返回空
    return None, True,None,None


def paste_area(elements,image,mask,background_origin,num_rows, num_cols, adjusted_grid_height,horizontal_gap, vertical_gap,index):

    background=background_origin.copy()
    adjusted_grid_width=adjusted_grid_height

    maskbackground=np.zeros_like(background)
    maskbackground[:] = (128, 128, 128)
    mask_binary= np.ones(background.shape[:2], dtype=np.uint8)*255
    augmented_data={'image':'','mask':'','mask_binary':'','type':''}
    count2=0
    count3=0
    name_index=0
    namelist = ['texte', 'math', 'figure']
    namelist=[name for name in namelist if len(elements[name])>0]
    num_name=len(namelist)
    for row in range(num_rows):
        for col in range(num_cols):

            remaining_width = math.ceil(adjusted_grid_width)
            remaining_height = math.ceil(adjusted_grid_height)
            x_position = math.ceil(col * (adjusted_grid_width + horizontal_gap) + horizontal_gap)
            y_position = math.ceil(row * (adjusted_grid_height + vertical_gap) + vertical_gap)

            while remaining_height > 30:
                idx=name_index%num_name
                name = namelist[idx]
                area_current = remaining_width * remaining_height
                elements_dict=get_element_math_figure(elements,area_current,background,image,remaining_width,remaining_height)
                if name=='texte':
                    element,nonfound,newwidth,newheight=get_element_texte(image,mask,remaining_width,remaining_height)
                    if nonfound:
                        count3 += 1
                        if count3 == 20:
                            count3 = 0
                            break
                        else:
                            name_index+=1
                            continue

                else:
                    if len(elements_dict[name]) > 0:
                        # 随机不重复选取
                        element_selected = random.sample(elements_dict[name], 1)[0]
                        element,newwidth,newheight=element_selected
                    else:
                        list1=['math','figure']
                        remaining_name = list(set(list1) - {name})[0]
                        if len(elements_dict[remaining_name]) > 0:
                            # 随机不重复选取
                            element_selected = random.sample(elements_dict[remaining_name], 1)[0]
                            element, newwidth, newheight = element_selected
                        else:
                            count2 += 1
                            if count2 == 20:
                                count2 = 0
                                break
                            else:
                                name_index += 1
                                continue


                polygon_points = np.array(element['points'], np.int32)
                rect = cv2.boundingRect(polygon_points)
                x, y, w, h = rect

                # 将Polygon的点偏移到外接矩形的局部坐标
                polygon_points_shifted = polygon_points - [x, y]
                # 创建一个与外接矩形大小一致的空白掩膜
                polygon_mask = np.zeros((h, w), dtype=np.uint8)
                # 在掩膜中绘制Polygon
                cv2.fillPoly(polygon_mask, [polygon_points_shifted], 255)
                # 创建一个新的三通道掩码，大小为外接矩形的大小，初始颜色为 (128, 128, 128)
                new_masque = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
                # 在 Polygon 区域复制颜色
                new_masque[polygon_mask == 255] = mask[y:y + h, x:x + w][polygon_mask == 255]

                # 将 128, 128, 128 排除，找到最多的颜色,表示主要的类 .any(axis=-1)：沿着最后一个维度（即颜色通道）进行检查，如果任何一个通道的值不为 128，则返回 True。这意味着这个像素不是灰色。
                # non_gray_pixels一个二维数组，其中每个元素是一个表示颜色的三维数组 [R, G, B]
                non_gray_pixels = new_masque[(new_masque != [128, 128, 128]).any(axis=-1)]
                # 统计每种颜色的频率 map(tuple, non_gray_pixels) 将每个 RGB 值转换为一个元组（tuple），因为 Counter 需要不可变的对象（如元组）来统计频次。
                # 统计每种颜色出现的频次，并返回一个字典，其中键是颜色 (R, G, B)，值是该颜色出现的次数。
                color_counts = Counter(map(tuple, non_gray_pixels))
                # 找到最多的颜色 most_common(1)表示只返回最常见的颜色。
                # color_counts.most_common(1) 返回一个列表，其中每个元素是一个 (color, count) 的元组。[0][0] 提取第一个元组的颜色部分。
                most_common_color = color_counts.most_common(1)[0][0]
                # 将所有非最多颜色的像素设置为 (128, 128, 128),因为非最多的颜色代表着标注时判断交集归属时，判断给了对方
                mask_to_modify = (new_masque != most_common_color).any(axis=-1)  # 找到非最多颜色的像素
                new_masque[mask_to_modify] = [128, 128, 128]  # 将这些像素设置为灰色

                cropped_element = image[y:y + h, x:x + w]
                # 数组进行切片操作时（如 background_origin[:h, :w]），它返回的是原数组的一个视图。视图是指向同一块内存的，不会创建新的数据。所以得用copy创建独立副本
                new_element=background_origin[:h,:w].copy()
                new_element[polygon_mask==255]=cropped_element[polygon_mask==255]

                binary_image_all=otsu_binary(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
                binary_image_crop=binary_image_all[y:y+h,x:x+w]
                new_binary_image=np.ones((h,w),dtype=np.uint8)*255
                new_binary_image[polygon_mask==255]=binary_image_crop[polygon_mask==255]


                y_position_end = y_position + newheight
                x_position_end = x_position + newwidth
                # 取值在右边界和下边界的情况下
                if y_position_end > background.shape[0]:
                    y_position_end = background.shape[0]
                    height_exceed = y_position_end - y_position
                    scale = height_exceed / newheight
                    width_scale = int(newwidth * scale)
                    cropped_element = cv2.resize(new_element, (width_scale, height_exceed),interpolation=cv2.INTER_NEAREST)
                    cropped_mask = cv2.resize(new_masque, (width_scale, height_exceed),interpolation=cv2.INTER_NEAREST)
                    cropped_binary_img = cv2.resize(new_binary_image, (width_scale, height_exceed),interpolation=cv2.INTER_NEAREST)
                    x_position_end = x_position + width_scale
                    newheight = height_exceed
                elif x_position_end > background.shape[1]:
                    x_position_end = background.shape[1]
                    width_exceed = x_position_end - x_position
                    scale = width_exceed / newwidth
                    height_scale = int(newheight * scale)
                    cropped_element = cv2.resize(new_element, (width_exceed, height_scale),interpolation=cv2.INTER_NEAREST)
                    cropped_mask = cv2.resize(new_masque, (width_exceed, height_scale),interpolation=cv2.INTER_NEAREST)
                    cropped_binary_img = cv2.resize(new_binary_image, (width_exceed, height_scale),interpolation=cv2.INTER_NEAREST)
                    y_position_end = y_position + height_scale
                    newheight=height_scale
                else:
                    # cv2.INTER_LINEAR（默认）：双线性插值，适用于一般的缩放操作，在缩小时会使图像有一定的模糊。
                    # cv2.INTER_NEAREST：邻近插值，适用于需要保持图像锐利度、不需要模糊处理的场景，尤其是在缩小时保持清晰边缘。
                    cropped_element = cv2.resize(new_element, (newwidth, newheight),interpolation=cv2.INTER_NEAREST)
                    cropped_mask = cv2.resize(new_masque, (newwidth, newheight),interpolation=cv2.INTER_NEAREST)
                    cropped_binary_img=cv2.resize(new_binary_image, (newwidth, newheight),interpolation=cv2.INTER_NEAREST)
                # # 将图像转为灰度图并使用otsu方法二值化 仅粘贴二值图像中黑色（即原图像中的黑色像素）
                gray_element = cv2.cvtColor(cropped_element, cv2.COLOR_BGR2GRAY)
                binary_image=otsu_binary(gray_element)
                black_pixel_mask = binary_image == 0

                background[y_position:y_position_end, x_position:x_position_end][cropped_binary_img == 0] = cropped_element[cropped_binary_img == 0]
                mask_binary[y_position:y_position_end, x_position:x_position_end] = cropped_binary_img

                # 区域masque
                maskbackground[y_position:y_position_end, x_position:x_position_end] = cropped_mask
                # 内容masque
                # maskbackground[y_position:y_position_end, x_position:x_position_end][black_pixel_mask] = \
                #     cropped_mask[black_pixel_mask]


                # plt.figure(figsize=(20,15))
                # plt.subplot(3, 2, 1)
                # plt.imshow(cropped_element)
                # plt.subplot(3, 2, 2)
                # plt.imshow(cropped_binary_img,cmap='gray')
                # plt.subplot(3, 2, 3)
                # plt.imshow(background)
                # plt.subplot(3,2,4)
                # plt.imshow(mask_binary,cmap='gray')
                # plt.subplot(3, 2, 5)
                # plt.imshow(cropped_mask)
                # plt.subplot(3, 2, 6)
                # plt.imshow(maskbackground)
                # plt.show()
                # 更新位置和剩余高度
                y_position += newheight
                remaining_height -= newheight
                name_index+=1

    maskbackground_gray = rgb_to_gray_array(maskbackground)
    maskbackground_combined = combine_class(maskbackground_gray)
    kernel = np.ones((3, 3), np.uint8)
    background_open = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel)


    plt.close()
    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.subplot(1, 3, 1)
    plt.imshow(background_open)
    plt.subplot(1, 3, 2)
    plt.imshow(gray_array_to_rgb(maskbackground_combined))
    plt.subplot(1, 3, 3)
    plt.imshow(mask_binary, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    plt.savefig(f'./Augmentation/synthetic_index{index}open.png',bbox_inches='tight')

    augmented_data["image"] = background_open
    augmented_data["mask"] = maskbackground_combined
    augmented_data["mask_binary"]=mask_binary
    augmented_data["type"] = 'synthetic'


    return augmented_data

def calculate_area(elements,name):
    annotations=elements[name]
    area_list=[]
    for sample in annotations:
        polygon_points = np.array(sample[1]['points'], np.int32)
        polygon = Polygon(polygon_points)
        area_list.append(polygon.area)
    return statistics.median(area_list)*4 if len(area_list) >0 else 100000

def set_grid(standard_area,background_height,background_width):
    # 计算每个格子的宽度和高度
    grid_width = math.sqrt(standard_area)
    grid_height = grid_width
    # 每行能够容纳的格子数
    num_cols = int(background_width // grid_width)
    # 计算多余的水平间隙
    total_width_occupied = num_cols * grid_width
    horizontal_gap = (background_width - total_width_occupied) / (num_cols + 1)
    # 如果水平间隙为负，则调整策略，压缩格子尺寸以适应宽度
    if horizontal_gap < 0:
        horizontal_gap = 0
        grid_width = background_width / num_cols
        grid_height = grid_width
    # 计算整个区域内的格子数
    total_area = background_width * background_height
    num_grids = total_area // standard_area
    # 总行数
    num_rows = int(num_grids // num_cols)
    # 计算多余的垂直间隙
    total_height_occupied = num_rows * grid_height
    vertical_gap = (background_height - total_height_occupied) / (num_rows + 1)
    # 如果垂直间隙为负，则调整策略，压缩格子尺寸以适应高度
    if vertical_gap < 0:
        vertical_gap = 0
        grid_height = background_height / num_rows
        grid_width = grid_height  # 确保格子保持方形

    return num_rows,num_cols,grid_width,horizontal_gap, vertical_gap


def place_elements_without_collision(image_mask_bgr_elements_dict,index):

    background=image_mask_bgr_elements_dict[2]
    background_height,background_width=background.shape[0],background.shape[1]
    elements=image_mask_bgr_elements_dict[3]
    image=image_mask_bgr_elements_dict[0]
    mask=image_mask_bgr_elements_dict[1]
    standard_area=calculate_area(elements,'math')
    num_rows, num_cols, grid_height,horizontal_gap, vertical_gap=set_grid(standard_area,background_height,background_width)
    augmented_data=paste_area(elements, image, mask, background, num_rows, num_cols,grid_height, horizontal_gap, vertical_gap,index)

    return augmented_data

def show_augmented_images(image, mask, augmentation_functions):
    for i, aug in enumerate(augmentation_functions):

        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        augname = str(aug.__name__)
        if 'elastic' in aug.__name__:
            augmented_image, augmented_mask = aug(image, mask, alpha=34, sigma=4)
        elif 'random_color_jitter' == aug.__name__ or 'random_gaussian_blur' == aug.__name__ or 'random_gaussian_noise' == aug.__name__ or 'random_sharpen' == aug.__name__ or 'random_contrast' == aug.__name__:
            augmented_image = aug(image)
        else:
            augmented_image, augmented_mask = aug(image, mask)
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 1].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'Augmented Image {augname}')
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Original Mask')
        axes[1, 1].imshow(augmented_mask, cmap='gray')
        axes[1, 1].set_title(f'Augmented Mask {augname}')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    image1 = cv2.imread(
        r"C:\Users\86139\Downloads\Leibniz-Manuscript-Detection-Segmentation-main\AnnotedImage\img_LH_35_5_6_11v.jpg")
    # 灰度计算的和直接灰度读取的会有部分偏差
    mask1 = cv2.imread(
        r"C:\Users\86139\Downloads\Leibniz-Manuscript-Detection-Segmentation-main\DataAugmentation\Labelmap\img_LH_35_5_6_11v_mask.png",
        cv2.IMREAD_GRAYSCALE)
    label = cv2.cvtColor(mask1, cv2.COLOR_BGR2RGB)
    mask2 = rgb_to_gray_array(label)
    a = mask1 == mask2
    # rgb
    color_map = {
        (0, 0, 255): 0,  # texte
        (0, 255, 0): 1,  # figure
        (255, 0, 0): 2,  # math
        (0, 255, 255): 3,  # mathstructuree
        (255, 0, 255): 4,  # textemath
        (255, 255, 0): 5,  # mathbarree
        (128, 128, 128): 6  # background
    }
    # rgb:(0, 0, 255) gray:29
    # rgb:(0, 255, 0) gray:149
    # rgb:(255, 0, 0) gray:76
    # rgb:(0, 255, 255) gray:178
    # rgb:(255, 0, 255) gray:105
    # rgb:(255, 255, 0) gray:225
    # rgb:(128, 128, 128) gray:127

    for index, value in enumerate(color_map):
        color = rgb_to_gray_value(value)
        print(f'rgb:{value} gray:{color}')

    augmentations = [
        random_perspective_transform, random_elastic_transform,
         random_ouverture, random_color_jitter,
        random_gaussian_blur, random_gaussian_noise, random_sharpen,
        random_rotate, random_flip, random_shift, random_contrast
    ]
    old_size = image1.shape[:2]
    output_size = 768
    ratio = float(output_size) / max(old_size)
    new_size = [int(x * ratio) for x in old_size]
    if max(old_size) != output_size:
        image_resized = cv2.resize(image1, (new_size[1], new_size[0]))
    if max(mask2.shape[:2]) != output_size:
        mask_resized = cv2.resize(mask2, (new_size[1], new_size[0]))
    show_augmented_images(image_resized, mask_resized, augmentations)
