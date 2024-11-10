# -*- coding: utf-8 -*-

"""
    The normalization params module
    ======================

    Use it to get the mean and standard deviation of the training set.
"""

import logging

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from preprocessing import PredictionDataset, Rescale, ToTensor


def run(
    data_paths: dict,
    img_size: int,
    num_workers: int ,
    norm_params:dict
):

    dataset = PredictionDataset(
        data_paths["test"]["image"],
        transform=transforms.Compose([Rescale(img_size), ToTensor()]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # Compute mean and std.
    mean = []
    std = []
    for data in tqdm(loader, desc="Computing parameters (prog)"):
        image = data["image"].numpy()
        # 要跨所有样本图像的所有宽高来求各通道的均值方差。
        # 具体为当计算某个通道的均值就是所有图像在该通道下的像素值总和除以所有图像在该通道下的像素总数量
        mean.append(np.mean(image, axis=(0, 2, 3)))
        std.append(np.std(image, axis=(0, 2, 3)))
    # n个样本，每个样本3个通道下各有均值，此时在批次上做均值，获得一个通道的全局均值
    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)

    logging.info("Mean in test: {}".format(np.uint8(mean)))
    logging.info("Std in test: {}".format(np.uint8(std)))
    norm_params['test']['mean']=np.uint8(mean)
    norm_params['test']['std']=np.uint8(std)
    return norm_params

