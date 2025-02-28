# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import model

from evaluate import run as evaluate

from normalization_params import run as normalization_params
from predict import run as predict
from training import run as train
from doc_functions import DLACollateFunction, Sampler
from preprocessing import (
    Normalize,
    Pad,
    PredictionDataset,
    Rescale,
    ToTensor,
    TrainingDataset,
    apply_augmentations_and_compute_stats,
    random_perspective_transform, random_elastic_transform, random_ouverture, random_rotate, random_flip,
    random_color_jitter, random_gaussian_blur, random_gaussian_noise, random_sharpen, random_contrast,
    compute_class_pixel_ratios, compute_mean_std)
from training_utils import Diceloss,CustomCrossEntropyLoss
import json

logger = logging.getLogger(__name__)


def get_mean_std(log_path: Path, mean_name: str, std_name: str) -> dict:

    mean_path = log_path / mean_name
    if not mean_path.exists():
        raise Exception(f"No file found at {mean_path}")

    std_path = log_path / std_name
    if not std_path.exists():
        raise Exception(f"No file found at {std_path}")

    with mean_path.open() as f:
        mean = f.read().splitlines()
        mean = [int(value) for value in mean]

    with std_path.open() as f:
        std = f.read().splitlines()
        std = [int(value) for value in std]

    return {"mean": mean, "std": std}


def training_loaders(
    exp_data_paths: dict,
    classes_colors: list,
    img_size: int,
    bin_size: int,
    batch_size: int,
    start_ratio:float,
    end_ratio:float,
    no_of_epochs:int,
    num_workers: int ,
    norm_params:dict,
    bgrdir,
    generated_images
) -> dict:

    loaders = {}

    t = tqdm(["train", "val"])
    t.set_description("Loading data")
    for set, images, masks,jsons in zip(
        t,[exp_data_paths["train"]["image"], exp_data_paths["val"]["image"]],[exp_data_paths["train"]["mask"], exp_data_paths["val"]["mask"]],[exp_data_paths["train"]["json"], exp_data_paths["val"]["json"]]):
        augment_all, mean_aug, std_aug = apply_augmentations_and_compute_stats(images, masks,jsons,bgrdir,img_size,set,generated_images,start_ratio)
        norm_params[set]['mean']=mean_aug
        norm_params[set]['std']=std_aug

        dataset = TrainingDataset(
            augment_all,
            classes_colors,
            transform=transforms.Compose([
                Normalize(mean_aug.tolist(), std_aug.tolist())
            ]),
            augmentations_transformation=[random_perspective_transform, random_elastic_transform, random_rotate, random_flip ,random_ouverture] if set=='train' else None,
            augmentations_pixel=[random_color_jitter,random_gaussian_blur,random_gaussian_noise,random_sharpen,random_contrast] if set=='train' else None
        )

        dataset.forbid = True
        class_pixel_ratios = compute_class_pixel_ratios(dataset, 5,0,int(len(dataset)),'real')
        logging.info(f"Real images: Pixel ratio of each class in {set}: {class_pixel_ratios}")
        class_pixel_ratios = compute_class_pixel_ratios(dataset, 5, 0, int(len(dataset)), 'synthetic')
        logging.info(f"Generated images: Pixel ratio of each class in {set}: {class_pixel_ratios}")
        dataset.forbid=False
        loaders[set] = DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_sampler=Sampler(
                augment_all,
                bin_size=bin_size,
                batch_size=batch_size,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                no_of_epochs=no_of_epochs,
                generated_images=generated_images,
                israndom=True if set=="train" else False
            ),
            collate_fn=DLACollateFunction()
        )
        if generated_images:
            logging.info(f"{set}: Found {len(dataset)} images {len(dataset)/2} real images {len(dataset)/2} synthetic images") if set=='train' else logging.info(f"{set}: Found {len(dataset)} images ")
        else:
            logging.info(f"{set}: Found {len(dataset)} images ")
    return loaders,norm_params


def is_small_patch(img, var_threshold, mad_threshold):
    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算灰度方差
    variance = np.var(gray_img)

    # 计算平均绝对偏差
    mean_gray = np.mean(gray_img)
    mad = np.mean(np.abs(gray_img - mean_gray))

    # 判断是否低于阈值
    if variance < var_threshold or mad < mad_threshold:
        return True  # 认为是过小或无效区域
    else:
        return False  # 认为是有效区域
def standard_crop_padding(image, img_name,dataset_name,crop_size, i):
    img_height, img_width = image.shape[:2]
    crop_width, crop_height = crop_size

    count=0
    cropped_img_list=[]
    # # 计算最小区域的信息量
    # gray_img = cv2.cvtColor(min_element, cv2.COLOR_RGB2GRAY)
    # # 计算灰度方差
    # variance = np.var(gray_img)
    # # 计算平均绝对偏差
    # mean_gray = np.mean(gray_img)
    # mad = np.mean(np.abs(gray_img - mean_gray))

    for y in range(0, img_height , crop_width):
        for x in range(0, img_width , crop_height):

            if x + crop_width > img_width and y + crop_height > img_height:
                padded_img = np.zeros((crop_height, crop_width, image.shape[2]), dtype=image.dtype)
                padded_img[0:img_height-y, 0:img_width - x] = image[y:img_height, x:img_width]
                cropped_img = padded_img
                crop_coords = (x, y, img_width,img_height)
                str_crop_coords = f'{x}_{y}_{img_width}_{img_height}'
                ratio_area=(img_width-x)*(img_height-y)/(crop_width*crop_height)
                # if is_small_patch(cropped_img,variance,mad):
                #     print('x y exceed', x, y, x + crop_width, y + crop_height,'invalid')
                #     continue

                # plt.figure(1)
                # plt.subplot(2, 1, 1)
                # plt.imshow(cropped_img)
                # plt.subplot(2, 1, 2)
                # plt.imshow(image)
                # plt.show()
            elif x + crop_width > img_width:
                diff=x+crop_width-img_width
                padded_img = np.zeros((crop_height, crop_width, image.shape[2]), dtype=image.dtype)
                padded_img[0:crop_height, 0:img_width-x] = image[y:y + crop_height, x:img_width]
                cropped_img = padded_img
                crop_coords = (x, y, img_width, y+crop_height)
                str_crop_coords = f'{x}_{y}_{img_width}_{ y+crop_height}'
                ratio_area = (img_width - x) * (crop_height) / (crop_width * crop_height)
                # if is_small_patch(cropped_img,variance,mad):
                #     print('x exceed', x, y, x + crop_width, y + crop_height,'invalid')
                #     continue

                # plt.figure(1)
                # plt.subplot(2, 1, 1)
                # plt.imshow(cropped_img)
                # plt.subplot(2, 1, 2)
                # plt.imshow(image)
                # plt.show()

            elif y + crop_height > img_height:
                diff=y+crop_height-img_height
                padded_img = np.zeros((crop_height, crop_width, image.shape[2]), dtype=image.dtype)
                padded_img[0:img_height-y, 0:crop_width] = image[y:img_height, x:x+crop_width]
                cropped_img = padded_img
                crop_coords = (x, y, x+crop_width, img_height)
                str_crop_coords = f'{x}_{y}_{x+crop_width}_{img_height}'
                ratio_area = (crop_width) * (img_height - y) / (crop_width * crop_height)
                # if is_small_patch(cropped_img,variance,mad):
                #     print('x exceed', x, y, x + crop_width, y + crop_height,'invalid')
                #     continue

                # plt.figure(1)
                # plt.subplot(2, 1, 1)
                # plt.imshow(cropped_img)
                # plt.subplot(2, 1, 2)
                # plt.imshow(image)
                # plt.show()

            else:
                cropped_img = image[y:y + crop_height, x:x + crop_width]
                crop_coords = (x, y, x + crop_width, y + crop_height)
                str_crop_coords=f'{x}_{y}_{x+crop_width}_{y+crop_height}'

            cropped_img_list.append({"image":cropped_img,"name":img_name,"dataset":dataset_name,"size":image.shape[:2],"position":str_crop_coords})
            count+=1


    logging.info(f"Image {img_name.name} Create {count} sub images")

    return cropped_img_list,count
def find_min_value(folder,json_name,img_size):
    base_name=os.path.basename(json_name)
    json_name = os.path.join(folder, base_name.replace('jpg', 'json'))
    all_elements=[]
    with open(os.path.join(folder, json_name), 'r') as f:
        annotations = json.load(f)
    for annotation in annotations['shapes']:
        element = (os.path.join(folder, json_name.replace('.json', '.jpg')), annotation)
        all_elements.append(element)
    elements_with_areas = []
    for img_path, element in all_elements:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = np.array(element['points'], dtype=np.int32)
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        cropped_element = image[y:y + h, x:x + w]
        area = cropped_element.shape[0] * cropped_element.shape[1]
        elements_with_areas.append((cropped_element, area, cropped_element.shape[1], cropped_element.shape[0]))
        # plt.figure(1)
        # plt.imshow(cropped_element)
        # plt.title(os.path.basename(img_path).split('.')[0])
        # plt.show()
    sorted_elements = sorted(elements_with_areas, key=lambda x: x[1])
    min_element=sorted_elements[0]
    min_area=min_element[1]
    # plt.figure(1)
    # plt.imshow(min_element[0])
    # plt.show()
    return min_element[0]

def apply_crop(images_dir,output_size):
    images = [
        (images_dir.parent.parent.name, images_dir / element)
        for element in os.listdir(images_dir)
    ]
    cropped_list=[]
    overlap_list=[]
    count_list=[]
    means=[]
    stds=[]
    for i, name in enumerate(tqdm(images, desc="Split images for test")):

        img_name = images[i][1]
        # min_element=find_min_value('../AnnotedImage', str(img_name),output_size)
        dataset_name=images[i][0]
        image = cv2.imread(str(img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cropped_samples,overlap_pairs,count = standard_crop_overlap(image,img_name,dataset_name,(output_size, output_size), 0,i)
        cropped_samples, count = standard_crop_padding(image, img_name, dataset_name,
                                                                      (output_size, output_size), i)
        cropped_list.extend(cropped_samples)
        count_list.append(count)
    all_images = [sample["image"] for sample in cropped_list]
    mean, std = compute_mean_std(all_images, batch_size=10)  # 使用分批次计算均值和标准差
    means.append(mean)
    stds.append(std)
    mean_final = np.mean(means, axis=0)
    std_final = np.mean(stds, axis=0)
    # all_images =[sample["image"] for sample in cropped_list]
    # mean=np.mean(all_images, axis=(0, 1, 2))
    # std=np.std(all_images, axis=(0, 1, 2))
    return cropped_list,np.uint8(mean_final), np.uint8(std_final),count_list

def prediction_loaders(
         exp_data_paths: dict, img_size: int, num_workers: int, steps
) -> dict:
    loaders = {}
    for set, images in zip(
            ["train", "val", "test"],
            [
                exp_data_paths["train"]["image"],
                exp_data_paths["val"]["image"],
                exp_data_paths["test"]["image"],
            ],
    ):
        if set != "test" and not set in steps:
            loaders[set + "_loader"] = {}
            continue
        # 有问题！！！！ 测试集的标准化参数必须用训练集的！！！
        cropped_list,new_mean,new_std,count_list=apply_crop(images,img_size)

        #  训练阶段不用显示转换为tensor是因为有collate_fn（DLACollateFunction）
        dataset = PredictionDataset(
            cropped_list,
            transform=transforms.Compose(
                [
                    Normalize(new_mean.tolist(),new_std.tolist()),
                    ToTensor(),
                ]
            ),
        )
        loaders[set + "_loader"] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders,count_list,cropped_list
def training_initialization(
    training: str,
    classes_names: list,
    use_amp: bool,
    learning_rate: float,
    same_classes:bool,
    loss:str,
) -> dict:

    no_of_classes = len(classes_names)

    net = model.load_network(no_of_classes, use_amp)
    net.apply(model.weights_init)

    if training is None:
        tr_params = {
            "net": net,
            "criterion": CustomCrossEntropyLoss(no_of_classes),
            "optimizer": Adam(net.parameters(), lr=learning_rate),
            "saved_epoch": 0,
            "best_loss": 10e5,
            "scaler": GradScaler(enabled=use_amp),
            "use_amp": use_amp,
        }
    else:
        # Restore model to resume training.
        checkpoint, net, optimizer, scaler = model.restore_model(
            net,
            Adam(net.parameters(), lr=learning_rate),
            GradScaler(enabled=use_amp),
            str(training),
            same_classes,
        )
        tr_params = {
            "net": net,
            "criterion": CustomCrossEntropyLoss(no_of_classes),
            "optimizer": optimizer,
            "saved_epoch": checkpoint["epoch"],
            "best_loss": checkpoint["best_loss"]
            if loss == "best"
            else 10e5,
            "scaler": scaler,
            "use_amp": use_amp,
        }
    return tr_params


def prediction_initialization(
    model_path: str, classes_names: list, log_path: str
) -> dict:

    no_of_classes = len(classes_names)
    net = model.load_network(no_of_classes, False)
    _, net, _, _ = model.restore_model(net, None, None,model_path)
    return net


def run(config: dict, num_workers: int = 0):
    assert len(config["steps"]) > 0, "No step to run"
    run_experiment(config=config, num_workers=num_workers)


def run_experiment(config: dict, num_workers: int ):

    assert len(config["steps"]) > 0, "No step to run"
    norm_params={"train": {},"val": {},"test": {}}

    if "train" in config["steps"]:
        loaders,norm_params = training_loaders(
            exp_data_paths=config["data_paths"],
            classes_colors=config["classes_colors"],
            img_size=config["img_size"],
            bin_size=config["bin_size"],
            batch_size=config["batch_size"],
            start_ratio=config["start_ratio"],
            end_ratio=config["end_ratio"],
            no_of_epochs=config["no_of_epochs"],
            num_workers=num_workers,
            norm_params=norm_params,
            bgrdir=config['bgrdir'],
            generated_images=config['generated_images']

        )

        savepath = str(config['log_path'] / config["norm_params"])+'.txt'
        with open(savepath, "w") as file:
            for key,value in norm_params.items():
                file.write(f"set:{key}:" + "\n")
                for k,v in value.items():
                    file.write(f"{k}:"+str(v) + "\n")

        tr_params = training_initialization(
            config["model_path"],
            config["classes_names"],
            config["use_amp"],
            config["learning_rate"],
            config["same_classes"],
            config["loss"],
        )
        train(
            config["model_path"],
            config["log_path"],
            config["tb_path"],
            config["no_of_epochs"],
            norm_params,
            config["classes_names"],
            loaders,
            tr_params,
            config["batch_size"],
            config["desired_batchsize"],
            config["learning_rate"],
        )


    if "prediction" in config["steps"]:
        img_dir = getimg(config["data_paths"]['test']['image'])

        loaders,count_list,cropped_list = prediction_loaders(
            config["data_paths"], config["img_size"],num_workers,config["steps"]
        )
        net = prediction_initialization(
            str(config["model_path"]), config["classes_names"], config["log_path"]
        )
        predict(
            config["prediction_path"],
            config["log_path"],
            config["img_size"],
            config["classes_colors"],
            config["classes_names"],
            config["save_image"],
            config["min_cc"],
            loaders,
            net,
            img_dir,
            count_list,
            cropped_list
        )


    if "evaluation" in config["steps"]:
        for set in config["data_paths"].keys():
            if set =='train' or set=='val':
                continue
            jsondir=config["data_paths"][set]["json"]
            logpath=str(config["log_path"])
            predir=os.path.join(logpath,'prediction',set)
            if len(os.listdir(predir)) == 0:
                continue
            if os.path.isdir(jsondir):
                evaluate(
                    config["log_path"],
                    config["classes_names"],
                    set,
                    config["data_paths"][set]["json"],
                    config["data_paths"][set]["image"],
                    str(jsondir.parent.parent.name),
                    config["prediction_path"],
                    config["evaluation_path"]
                )
            else:
                logging.info(f"{jsondir} folder not found.")
def getimg(path):
    imgdir={}
    for p in os.listdir(str(path)):
        img=cv2.imread(os.path.join(path,p))
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgdir[p]=img_rgb
    return imgdir