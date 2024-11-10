# -*- coding: utf-8 -*-

"""
    The predict module
    ======================

    Use it to predict some images from a trained network.
"""

import logging
import os
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import prediction as pr_utils
import json




def run(
    prediction_path: str,
    log_path: str,
    img_size: int,
    colors: list,
    classes_names: list,
    save_image: list,
    min_cc: int,
    loaders: dict,
    net,
    img_dir,
    count_list,
    cropped_list
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Run prediction.
    net.eval()

    logging.info("Starting predicting")
    starting_time = time.time()
    index_img=0
    num_subimg=count_list[index_img]
    num_img=0
    flag=False
    all_polygons = {class_name: [] for class_name in classes_names[:-1]}
    with torch.no_grad():
        for index, (set, loader) in enumerate(zip(["train", "val", "test"], loaders.values())):

            seen_datasets = []
            # Create folders to save the predictions.
            os.makedirs(os.path.join(log_path, prediction_path, set), exist_ok=True)

            for i, data in enumerate(tqdm(loader, desc="Prediction (prog) " + set), 0):

                if data["dataset"][0] not in seen_datasets:
                    os.makedirs(
                        os.path.join(
                            log_path, prediction_path, set, data["dataset"][0]
                        ),
                        exist_ok=True,
                    )
                    seen_datasets.append(data["dataset"][0])

                logging.info(f"inference of image:{data['name'][0]} 's subimages, images index:{num_img}")
                num_img += 1

                output = net(data["image"].to(device).float())
                input_size = [element.numpy()[0] for element in data["size"][:2]]

                assert output.shape[0] == 1
                # 获得了每个预测的轮廓区域已经对应的置信度分数（区域内的预测概率值总和除以区域全1总和）
                polygons = pr_utils.get_predicted_polygons(
                    output[0].cpu().numpy(), min_cc, classes_names
                )


                polygons = pr_utils.get_polygons_points(
                    polygons, img_size
                )
                adjusted_polygons=pr_utils.adjust_polygons(polygons,data["position"][0],input_size)

                for class_name in classes_names[:-1]:
                    all_polygons[class_name].extend(adjusted_polygons[class_name])

                img_parent = os.path.join(log_path, prediction_path, set, data["dataset"][0],
                                          data["name"][0].split('.')[0])
                if not os.path.exists(img_parent):
                    os.makedirs(img_parent)
                polygons["img_size"] = [int(element) for element in input_size]
                adjusted_polygons["img_size"] = [int(element) for element in input_size]
                all_polygons["img_size"]=[int(element) for element in input_size]
                pr_utils.save_prediction(
                    polygons,
                    os.path.join(
                        img_parent,
                        data["position"][0]+'.jpg'
                    ),
                )


                if set in save_image:
                    pr_utils.save_prediction_image(
                        polygons,
                        colors,
                        (img_size,img_size),
                        os.path.join(
                            img_parent,
                            data["position"][0] + '.jpg'
                        ),
                        os.path.join(
                            img_parent,
                            str(data["position"][0]) + '_combined.jpg'
                        ),
                        cropped_list[i]
                    )
                    masque_combined,image_formask,flag=pr_utils.save_prediction_image_combined(
                        adjusted_polygons,
                        colors,
                        input_size,
                        os.path.join(log_path,prediction_path, set,data["dataset"][0], data["name"][0]),
                        os.path.join(log_path,prediction_path,set,data["dataset"][0],data["name"][0].split('.')[0]+'_combined.jpg'),img_dir[data["name"][0]],masque_combined if flag else None,image_formask if flag else None
                    )
                if num_img==num_subimg:
                    logging.info(f"inference of image:{data['name'][0]} finish, {num_subimg} sub images in total")
                    index_img+=1
                    if index_img<len(count_list):
                        num_subimg=count_list[index_img]
                    num_img=0
                    flag=False
                    pr_utils.save_prediction(all_polygons,os.path.join(log_path, prediction_path, set, data["dataset"][0],data["name"][0]))
                    all_polygons = {class_name: [] for class_name in classes_names[:-1]}


    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished predicting in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )



# def run(
#     prediction_path: str,
#     log_path: str,
#     img_size: int,
#     colors: list,
#     classes_names: list,
#     save_image: list,
#     min_cc: int,
#     loaders: dict,
#     net,
#     img_dir
#
# ):
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # Run prediction.
#     net.eval()
#
#     logging.info("Starting predicting")
#     starting_time = time.time()
#     num_img=0
#     with torch.no_grad():
#         for index, (set, loader) in enumerate(zip(["train", "val", "test"], loaders.values())):
#
#             seen_datasets = []
#
#             os.makedirs(os.path.join(log_path, prediction_path, set), exist_ok=True)
#
#             for i, data in enumerate(tqdm(loader, desc="Prediction (prog) " + set), 0):
#
#                 if data["dataset"][0] not in seen_datasets:
#                     os.makedirs(
#                         os.path.join(
#                             log_path, prediction_path, set, data["dataset"][0]
#                         ),
#                         exist_ok=True,
#                     )
#                     seen_datasets.append(data["dataset"][0])
#
#
#
#                 output = net(data["image"].to(device).float())
#                 input_size = [element.numpy()[0] for element in data["size"][:2]]
#
#                 assert output.shape[0] == 1
#                 # 获得了每个预测的轮廓区域已经对应的置信度分数（区域内的预测概率值总和除以区域全1总和）
#                 polygons = pr_utils.get_predicted_polygons(
#                     output[0].cpu().numpy(), min_cc, classes_names
#                 )
#
#                 polygons = pr_utils.resize_polygons(
#                     polygons, input_size, img_size, data["padding"]
#                 )
#
#
#
#                 polygons["img_size"] = [int(element) for element in input_size]
#                 pr_utils.save_prediction(
#                     polygons,
#                     os.path.join(
#                         log_path,
#                         prediction_path,
#                         set,
#                         data["dataset"][0],
#                         data["name"][0],
#                     ),
#                 )
#                 if set in save_image:
#                     pr_utils.save_prediction_image(
#                         polygons,
#                         colors,
#                         input_size,
#                         os.path.join(
#                             log_path,
#                             prediction_path,
#                             set,
#                             data["dataset"][0],
#                             data["name"][0]
#                         ),
#                         os.path.join(
#                             log_path,
#                             prediction_path,
#                             set,
#                             data["dataset"][0],
#                             data["name"][0].split('.')[0]+'_combined.jpg'
#                         ),
#                         img_dir[data["name"][0]]
#                     )
#
#
#     end = time.gmtime(time.time() - starting_time)
#     logging.info(
#         "Finished predicting in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
#     )
