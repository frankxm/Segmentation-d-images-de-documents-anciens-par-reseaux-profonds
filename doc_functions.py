# -*- coding: utf-8 -*-

"""
    The utils module
    ======================

    Generic functions used during all the steps.
"""

import copy
import math
import random

import numpy as np
import torch
import logging
# Useful functions.


def rgb_to_gray_value(rgb: tuple) -> int:
    """
    Compute the gray value of a RGB tuple.
    :param rgb: The RGB value to transform.
    :return: The corresponding gray value.
    """
    try:
        return int(rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
    except TypeError:
        return int(int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114)


def rgb_to_gray_array(rgb: np.ndarray) -> np.ndarray:
    """
    Compute the gray array (NxM) of a RGB array (NxMx3).
    :param rgb: The RGB array to transform.
    :return: The corresponding gray array.
    """
    gray_array = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return np.uint8(gray_array)


def create_buckets(images_sizes, bin_size):
    """
    Group images into same size buckets.
    :param images_sizes: The sizes of the images.
    :param bin_size: The step between two buckets.
    :return bucket: The images indices grouped by size.
    """

    max_size = max([image_size for image_size in images_sizes.values()])
    min_size = min([image_size for image_size in images_sizes.values()])
    # binsize为每个桶的尺寸范围，先创建空桶，每个桶的最大尺寸作为键
    bucket = {}
    current = min_size + bin_size - 1
    while current < max_size:
        bucket[current] = []
        current += bin_size
    bucket[max_size] = []
    # 遍历图像尺寸分配到特定桶
    for index, value in images_sizes.items():
        # 计算当前尺寸所属的桶的区域，计算上限
        dict_index = (((value - min_size) // bin_size) + 1) * bin_size + min_size - 1
        bucket[min(dict_index, max_size)].append(index)
    # 删除空桶，只保留有图像的桶
    bucket = {
        dict_index: values for dict_index, values in bucket.items() if len(values) > 0
    }
    return bucket
# # 同分布情况
# class Sampler(torch.utils.data.Sampler):
#     def __init__(self, data, bin_size, batch_size,start_ratio,end_ratio,no_of_epochs,israndom):
#         self.bin_size = bin_size
#         self.batch_size = batch_size
#         self.data_sizes = [(sample["image"].shape[0],sample["image"].shape[1]) for sample in data]
#         self.start_ratio = start_ratio
#         self.end_ratio = end_ratio
#         self.num_epochs = no_of_epochs/2
#         self.current_epoch = 0
#         self.israndom=israndom
#
#
#         self.real_indices = [i for i, sample in enumerate(data) if sample["type"] == "real"]
#         self.synthetic_indices = [i for i, sample in enumerate(data) if sample["type"] == "synthetic"]
#         if not self.israndom:
#             num_total = len(self.real_indices + self.synthetic_indices)
#             self.valid_indices=random.sample(self.real_indices, int(num_total / 4)) + random.sample(self.synthetic_indices, int(num_total / 4))
#         # 区分水平和竖直图像
#         self.vertical = {
#             index: sample[1]
#             for index, sample in enumerate(self.data_sizes)
#             if sample[0] > sample[1]
#         }
#         self.horizontal = {
#             index: sample[0]
#             for index, sample in enumerate(self.data_sizes)
#             if sample[0] <= sample[1]
#         }
#         # 创建竖直图像桶和水平图像桶
#         self.buckets = [
#             create_buckets(self.vertical, self.bin_size)
#             if len(self.vertical) > 0
#             else {},
#             create_buckets(self.horizontal, self.bin_size)
#             if len(self.horizontal) > 0
#             else {},
#         ]
#
#     def __len__(self):
#         return len(self.vertical) + len(self.horizontal)
#
#     def __iter__(self):
#         print(f"{self.current_epoch}")
#         buckets = copy.deepcopy(self.buckets)
#         # 打乱每种桶中每个键的图像索引
#         for index, bucket in enumerate(buckets):
#             for key in bucket.keys():
#                 random.shuffle(buckets[index][key])
#         # 对于训练集，两类图像的比例变化
#         if self.israndom:
#             # 一开始大部分为合成图，小部分为真实图，随着轮次增加真实图占全部，因此合成图减为0.合成图总数为n，真实图总数为n，总的训练数也为n
#             real_ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (self.current_epoch / self.num_epochs)
#             if self.current_epoch > self.num_epochs:
#               real_ratio = self.end_ratio
#             num_real = int(real_ratio * len(self.real_indices))
#             num_synthetic = len(self.synthetic_indices) - num_real
#             real_indices_sample = random.sample(self.real_indices, num_real)
#             synthetic_indices_sample = random.sample(self.synthetic_indices, num_synthetic)
#             mixed_indices = real_indices_sample + synthetic_indices_sample
#             random.shuffle(mixed_indices)
#             logging.info(f"real images:{num_real}  synthetic images:{num_synthetic} total images:{len(mixed_indices)}")
#         else:
#             mixed_indices = self.valid_indices
#             random.shuffle(mixed_indices)
#             logging.info(f"images in valid:{len(mixed_indices)} ")
#
#
#         # 按批次分组，根据每个桶的每个键逆序遍历，依次加入到final_indices数组中。每当达到batchsize时，批次增加索引增加。最后在打乱所有批次。
#         if self.batch_size is not None:
#             final_indices = []
#             index_current = -1
#             for bucket in buckets:
#                 current_batch_size = self.batch_size
#                 for key in sorted(bucket.keys(), reverse=True):
#                     for index in bucket[key]:
#                         if index in mixed_indices:
#                             if current_batch_size + 1 > self.batch_size:
#                                 current_batch_size = 0
#                                 final_indices.append([])
#                                 index_current += 1
#                             current_batch_size += 1
#                             final_indices[index_current].append(index)
#             random.shuffle(final_indices)
#
#         self.current_epoch+=1
#         return iter(final_indices)
class Sampler(torch.utils.data.Sampler):
    def __init__(self, data, bin_size, batch_size,start_ratio,end_ratio,no_of_epochs,generated_images,israndom):
        self.bin_size = bin_size
        self.batch_size = batch_size
        self.data_sizes = [(sample["image"].shape[0],sample["image"].shape[1]) for sample in data]
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.num_epochs = no_of_epochs
        self.current_epoch = 0
        self.israndom=israndom


        self.real_indices = [i for i, sample in enumerate(data) if sample["type"] == "real"]
        self.synthetic_indices = [i for i, sample in enumerate(data) if sample["type"] == "synthetic"]
        self.generated_images=generated_images

        # 区分水平和竖直图像
        self.vertical = {
            index: sample[1]
            for index, sample in enumerate(self.data_sizes)
            if sample[0] > sample[1]
        }
        self.horizontal = {
            index: sample[0]
            for index, sample in enumerate(self.data_sizes)
            if sample[0] <= sample[1]
        }
        # 创建竖直图像桶和水平图像桶
        self.buckets = [
            create_buckets(self.vertical, self.bin_size)
            if len(self.vertical) > 0
            else {},
            create_buckets(self.horizontal, self.bin_size)
            if len(self.horizontal) > 0
            else {},
        ]

    def __len__(self):
        return len(self.vertical) + len(self.horizontal)

    def __iter__(self):
        print(f"{self.current_epoch}")
        buckets = copy.deepcopy(self.buckets)
        # 打乱每种桶中每个键的图像索引
        for index, bucket in enumerate(buckets):
            for key in bucket.keys():
                random.shuffle(buckets[index][key])
        if self.generated_images:
            # 对于训练集，两类图像的比例变化
            if self.israndom:
                # 一开始大部分为合成图，小部分为真实图，随着轮次增加真实图占全部，因此合成图减为0.合成图总数为n，真实图总数为n，总的训练数也为n
                # 训练到期望的一半轮次时保证真实图比例最高
                real_ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (self.current_epoch / self.num_epochs)
                if self.current_epoch>self.num_epochs:
                    real_ratio=self.end_ratio
                num_real = math.ceil(real_ratio * len(self.real_indices))
                num_synthetic = len(self.real_indices) - num_real
                real_indices_sample = random.sample(self.real_indices, num_real)
                synthetic_indices_sample = random.sample(self.synthetic_indices, num_synthetic)
                mixed_indices = real_indices_sample + synthetic_indices_sample
                random.shuffle(mixed_indices)
                logging.info(f"real images:{num_real}  synthetic images:{num_synthetic} total images:{len(mixed_indices)}")
            # 对于验证集,只用原图
            else:
                mixed_indices = self.real_indices
                random.shuffle(mixed_indices)
                logging.info(f"real images in valid:{len(self.real_indices)} ")

        else:
            mixed_indices = self.real_indices
            random.shuffle(mixed_indices)
            logging.info(f"real images in train:{len(self.real_indices)} ") if self.israndom else logging.info(f"real images in valid:{len(self.real_indices)} ")


        # 按批次分组，根据每个桶的每个键逆序遍历，依次加入到final_indices数组中。每当达到batchsize时，批次增加索引增加。最后在打乱所有批次。
        if self.batch_size is not None:
            final_indices = []
            index_current = -1
            for bucket in buckets:
                current_batch_size = self.batch_size
                for key in sorted(bucket.keys(), reverse=True):
                    for index in bucket[key]:
                        if index in mixed_indices:
                            if current_batch_size + 1 > self.batch_size:
                                current_batch_size = 0
                                final_indices.append([])
                                index_current += 1
                            current_batch_size += 1
                            final_indices[index_current].append(index)
            random.shuffle(final_indices)

        self.current_epoch+=1
        return iter(final_indices)


def pad_images_masks(
    images, masks,masks_binary, image_padding_value, mask_padding_value
):
    """
    Pad images and masks to create batchs.
    :param images: The batch images to pad.
    :param masks: The batch masks to pad.
    :param image_padding_value: The value used to pad the images.
    :param mask_padding_value: The value used to pad the masks.
    :return padded_images: An array containing the batch padded images.
    :return padded_masks: An array containing the batch padded masks.
    """
    heights = [element.shape[0] for element in images]
    widths = [element.shape[1] for element in images]
    max_height = max(heights)
    max_width = max(widths)

    # Make the tensor shape be divisible by 8.
    if max_height % 8 != 0:
        max_height = int(8 * np.ceil(max_height / 8))
    if max_width % 8 != 0:
        max_width = int(8 * np.ceil(max_width / 8))
    # 创建一个批次，维度为batchsize  height width 3
    padded_images = (
        np.ones((len(images), max_height, max_width, images[0].shape[2]))
        * image_padding_value
    )
    padded_masks = np.ones((len(masks), max_height, max_width)) * mask_padding_value
    padded_masks_binary = np.ones((len(masks_binary), max_height, max_width)) * 255
    for index, (image, mask,mask_binary) in enumerate(zip(images, masks,masks_binary)):
        delta_h = max_height - image.shape[0]
        delta_w = max_width - image.shape[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_images[
            index,
            top : padded_images.shape[1] - bottom,
            left : padded_images.shape[2] - right,
            :,
        ] = image
        padded_masks[
            index,
            top : padded_masks.shape[1] - bottom,
            left : padded_masks.shape[2] - right,
        ] = mask
        padded_masks_binary[
        index,
        top: padded_masks_binary.shape[1] - bottom,
        left: padded_masks_binary.shape[2] - right,
        ] = mask_binary

    return padded_images, padded_masks,padded_masks_binary

# DataLoader 将获取到的样本数据传递给 collate_fn 函数。collate_fn 函数定义了如何将这些样本数据组合成一个批次（batch）
class DLACollateFunction:
    def __init__(self):
        self.image_padding_token = 0
        self.mask_padding_token = 4

    def __call__(self, batch):
        image = [item["image"] for item in batch]
        mask = [item["mask"] for item in batch]
        mask_binary = [item["mask_binary"] for item in batch]
        pad_image, pad_mask ,pad_mask_binary= pad_images_masks(
            image, mask,mask_binary, self.image_padding_token, self.mask_padding_token
        )
        return {
            "image": torch.tensor(pad_image).permute(0, 3, 1, 2),
            "mask": torch.tensor(pad_mask),
            "mask_binary": torch.tensor(pad_mask_binary),
        }
