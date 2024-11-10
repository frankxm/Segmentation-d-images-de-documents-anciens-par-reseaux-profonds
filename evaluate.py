# -*- coding: utf-8 -*-

"""
    The evaluation module
    ======================

    Use it to evaluation a trained network.
"""

import logging
import os
import time
from pathlib import Path


import numpy as np
from tqdm import tqdm

import evaluation as ev_utils
import object_metrics as o_metrics
import pixel_metrics as p_metrics
import cv2
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
def watch_labels(pred_regions,image,ground_classes_names):
    color_map = {
        'texte': [255, 0, 0],  # Blue
        'figure': [0, 255, 0],  # Green
        'math': [0, 0, 255],  # Red
        'mathstructuree': [255, 255, 0],  # Cyan
        'textemath': [255, 0, 255],  # Magenta
        'mathbarree': [0, 255, 255],  # Yellow
        'background': [128, 128, 128]  # Gray
    }
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # for index, channel in enumerate(ground_classes_names[:-1], 1):
    #     for shape in pred_regions['shapes']:
    #         if channel == shape['label'].lower():
    #             poly=Polygon(shape["points"]).buffer(0)
    #             polypoints = np.array(poly.exterior.coords, dtype=np.int32)
    #             cv2.fillPoly(image, [polypoints], color_map[channel])
    #             cv2.polylines(image, [polypoints], isClosed=True, color=color_map[channel], thickness=5)
    #     plt.figure(1)
    #     plt.imshow(image)
    #     plt.show()

    for index, channel in enumerate(ground_classes_names[:3], 1):
        for pred in pred_regions[channel]:
            polypoints = np.array(pred['polygon'], dtype=np.int32)
            cv2.fillPoly(image, [polypoints], color_map[channel])
            cv2.polylines(image, [polypoints], isClosed=True, color=color_map[channel], thickness=5)
        plt.figure(1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

def run(
    log_path: str,
    classes_names: list,
    set: str,
    data_paths: dict,
    image_path: dict,
    dataset: str,
    prediction_path: Path,
    evaluation_path: Path,
):

    # Run evaluation.
    logging.info("Starting evaluation: " + dataset)
    starting_time = time.time()

    label_dir = data_paths
    image_dir = image_path
    ground_classes_names = ["texte", "figure", "math", "mathstructuree", "textemath", "mathbarree","math_total", "background"]
    pixel_metrics = {
        channel: {methode+metric: [] for methode in ['black_','gray_'] for metric in ["iou", "dice","precision", "recall", "fscore"]}
        for channel in ground_classes_names[:-1]
    }
    object_metrics_black = {
        channel: {metric: {} for metric in ["precision", "recall", "fscore", "AP"]}
        for channel in ground_classes_names[:-1]
    }
    object_metrics_gray = {
        channel: {metric: {} for metric in ["precision", "recall", "fscore", "AP"]}
        for channel in ground_classes_names[:-1]
    }
    rank_scores_black = {
        channel: {
            iou: {rank: {"True": 0, "Total": 0} for rank in range(95, -5, -5)}
            for iou in range(50, 100, 5)
        }
        for channel in ground_classes_names[:-1]
    }
    rank_scores_gray={
        channel: {
            iou: {rank: {"True": 0, "Total": 0} for rank in range(95, -5, -5)}
            for iou in range(50, 100, 5)
        }
        for channel in ground_classes_names[:-1]
    }
    num_classes = len(ground_classes_names[:-1])
    global_confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    global_confusion_matrix_combined= np.zeros((4, 4), dtype=int)

    number_of_gt = {channel: 0 for channel in ground_classes_names[:-1]}
    for img_name in tqdm(os.listdir(label_dir), desc="Evaluation (prog) " + set):
        gt_regions = ev_utils.read_json(os.path.join(label_dir, img_name))

        gt_image,imagergb,imagegray=ev_utils.read_binimage(os.path.join(image_dir,os.path.splitext(img_name)[0]+'.jpg'))
        pred_regions = ev_utils.read_json(
            os.path.join(log_path, prediction_path, set, dataset, img_name)
        )

        # watch_labels(pred_regions, imagergb, ground_classes_names)
        gt_polys,mask_gt = ev_utils.get_gt_polygons(gt_regions, ground_classes_names,imagergb.shape)
        gt_polys=ev_utils.combine_math(gt_polys)
        size_gt_region=[gt_regions['imageHeight'],gt_regions['imageWidth']]
        if size_gt_region != pred_regions["img_size"]:
            pred_regions = ev_utils.resize_polygons(
                pred_regions, gt_regions["img_size"], pred_regions["img_size"]
            )
        pred_polys,mask_pred = ev_utils.get_pred_polygons(pred_regions, classes_names,imagergb.shape)

        global_confusion_matrix,global_confusion_matrix_combined=ev_utils.compute_confusion_matrix_pixel(mask_gt,mask_pred,global_confusion_matrix,global_confusion_matrix_combined)

        pixel_metrics = p_metrics.compute_metrics(
            gt_polys, pred_polys, ground_classes_names[:-1], pixel_metrics,gt_image,imagergb,imagegray
        )

        image_rank_scores_black,image_rank_scores_gray = o_metrics.compute_rank_scores(
            gt_polys, pred_polys, ground_classes_names[:-1],gt_image,imagegray
        )
        rank_scores_black = o_metrics.update_rank_scores(
            rank_scores_black, image_rank_scores_black, ground_classes_names[:-1]
        )
        rank_scores_gray = o_metrics.update_rank_scores(
            rank_scores_gray, image_rank_scores_gray, ground_classes_names[:-1]
        )
        number_of_gt = {
            channel: number_of_gt[channel] + len(gt_polys[channel])
            for channel in ground_classes_names[:-1]
        }

    object_metrics_black,weighted_confusion_matrix_black,weighted_confusion_matrix_combined_black = o_metrics.get_mean_results(
        rank_scores_black, number_of_gt, ground_classes_names[:-1], object_metrics_black
    )
    object_metrics_gray,weighted_confusion_matrix_gray,weighted_confusion_matrix_combined_gray = o_metrics.get_mean_results(
        rank_scores_gray, number_of_gt, ground_classes_names[:-1], object_metrics_gray
    )

    print(set)
    for channel in ground_classes_names[:-1]:
        print(channel)
        pixeltype=['black_','gray_']
        for type in pixeltype:
            print(type)
            print(f"{type}IOU       = ", np.round(np.mean(pixel_metrics[channel][type+"iou"]), 4))
            print(f"{type}Dice      = ", np.round(np.mean(pixel_metrics[channel][type + "dice"]), 4))
            print(f"{type}Precision = ", np.round(np.mean(pixel_metrics[channel][type+"precision"]), 4))
            print(f"{type}Recall    = ", np.round(np.mean(pixel_metrics[channel][type+"recall"]), 4))
            print(f"{type}F-score   = ", np.round(np.mean(pixel_metrics[channel][type+"fscore"]), 4))

        aps_black = object_metrics_black[channel]["AP"]
        print('Black:')
        print("AP [IOU=0.50] = ", np.round(aps_black[50], 4))
        print("AP [IOU=0.75] = ", np.round(aps_black[75], 4))
        print("AP [IOU=0.95] = ", np.round(aps_black[95], 4))
        print("AP [0.5,0.95] = ", np.round(np.mean(list(aps_black.values())), 4))

        aps_gray = object_metrics_gray[channel]["AP"]
        print('Gray:')
        print("AP [IOU=0.50] = ", np.round(aps_gray[50], 4))
        print("AP [IOU=0.75] = ", np.round(aps_gray[75], 4))
        print("AP [IOU=0.95] = ", np.round(aps_gray[95], 4))
        print("AP [0.5,0.95] = ", np.round(np.mean(list(aps_gray.values())), 4))
        print("\n")

    os.makedirs(os.path.join(log_path, evaluation_path, set), exist_ok=True)


    ev_utils.plot_confusion_matrix(global_confusion_matrix,
                                   ["texte", "figure", "math", "mathstructuree", "textemath", "mathbarree",
                                    "background"], os.path.join(log_path, evaluation_path, set),'pixel')


    ev_utils.plot_confusion_matrix(global_confusion_matrix_combined,
                                   ["texte", "figure", "math_total","background"], os.path.join(log_path, evaluation_path, set),'pixel_combined')



    ev_utils.plot_pixel_metrics(pixel_metrics, ground_classes_names[:-1],os.path.join(log_path, evaluation_path, set))


    ev_utils.save_graphical_results(object_metrics_black, ground_classes_names[:-1],os.path.join(log_path, evaluation_path, set),'black',weighted_confusion_matrix_black,weighted_confusion_matrix_combined_black)
    ev_utils.save_graphical_results(object_metrics_gray, ground_classes_names[:-1],os.path.join(log_path, evaluation_path, set),'gray',weighted_confusion_matrix_gray,weighted_confusion_matrix_combined_gray)
    ev_utils.save_results(
        pixel_metrics,
        object_metrics_black,
        object_metrics_gray,
        ground_classes_names[:-1],
        os.path.join(log_path, evaluation_path, set),
        dataset,
    )

    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished evaluating in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )


