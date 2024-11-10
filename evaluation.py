# -*- coding: utf-8 -*-

"""
    The evaluation utils module
    ======================

    Use it to during the evaluation stage.
"""

import json
import os
import warnings

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from shapely import MultiPolygon
from shapely.geometry import Polygon
import seaborn as sns
import copy
from scipy.interpolate import interp1d
# 忽略所有的 UserWarning 警告
warnings.filterwarnings("ignore", category=UserWarning)

def combine_math(polys):
    math_list=["math","mathstructuree","textemath","mathbarree"]
    for channel in math_list:
        polys["math_total"].extend(polys[channel])
    return polys


def resize_polygons(polygons: dict, gt_size: tuple, pred_size: tuple) -> dict:

    # Compute resizing ratio
    ratio = [gt / pred for gt, pred in zip(gt_size, pred_size)]

    for channel in polygons.keys():
        if channel == "img_size":
            continue
        for index, polygon in enumerate(polygons[channel]):
            x_points = [int(element[1] * ratio[0]) for element in polygon["polygon"]]
            y_points = [int(element[0] * ratio[1]) for element in polygon["polygon"]]

            x_points = [
                int(element) if element < gt_size[0] else int(gt_size[0])
                for element in x_points
            ]
            y_points = [
                int(element) if element < gt_size[1] else int(gt_size[1])
                for element in y_points
            ]
            x_points = [int(element) if element > 0 else 0 for element in x_points]
            y_points = [int(element) if element > 0 else 0 for element in y_points]

            assert max(x_points) <= gt_size[0]
            assert min(x_points) >= 0
            assert max(y_points) <= gt_size[1]
            assert min(y_points) >= 0
            polygons[channel][index]["polygon"] = list(zip(y_points, x_points))
    return polygons


def read_binimage(filename):
    imagebgr = cv2.imread(filename)
    image_rgb=cv2.cvtColor(imagebgr,cv2.COLOR_BGR2RGB)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # # 自适应阈值化
    # binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15,
    #                                      9)
    # # 形态学闭运算定义一个3x3的卷积核
    # kernel = np.ones((3, 3), np.uint8)
    #
    # # 闭运算（先膨胀后腐蚀）膨胀的点为白色,为背景。关心的为黑色
    # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    _, img_binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学操作去除黑色边缘,扩张白色区域，减少黑色区域，方便后续获取文本页面边缘
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(img_binary_otsu, kernel, iterations=1)
    # 创建掩码以去除无用的黑色区域
    # cv2.RETR_EXTERNAL: 只检索最外层轮廓。这个模式会忽略内部轮廓，只保留最外面的轮廓。适用于只需要找出图像中最外面的边界（例如页面的边界）而不关心内部内容时使用。
    # cv2.CHAIN_APPROX_SIMPLE 只保留轮廓的端点，减少了轮廓点的数量。这种方法能减少存储和处理轮廓所需的内存，但可能会丢失一些轮廓细节
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_otsu = np.zeros_like(img_binary_otsu)
    for contour in contours:
        # cv2.drawContours: 绘制轮廓，将掩码的轮廓区域填充为白色（255）
        cv2.drawContours(mask_otsu, [contour], -1, 255, thickness=cv2.FILLED)
    # 将黑色区域转换为白色背景 掩码为白色（255）的区域保留原图像的像素，其他区域设置为白色（255）
    binary_image = np.where(mask_otsu == 255, img_binary_otsu, 255)
    return binary_image,image_rgb,image


def read_json(filename: str) -> dict:
    with open(filename, "r") as file:
        return json.load(file)


def get_pred_polygons(regions: dict, classes: list,image_shape) -> dict:

    color_map = {
        'texte': [0, 0, 255],  # Blue
        'figure': [0, 255, 0],  # Green
        'math': [255, 0, 0],  # Red
        'mathstructuree': [0, 255, 255],  # Cyan
        'textemath': [255, 0, 255],  # Magenta
        'mathbarree': [255, 255, 0],  # Yellow
        'background': [128, 128, 128]  # Gray
    }
    polys = {}
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[:, :] = color_map['background']
    for index, channel in enumerate(classes[:-1], 1):
        if channel in regions.keys():
            polys[channel] = []
            for polygon in regions[channel]:
                # 获取置信度和多边形
                confidence = polygon["confidence"]
                poly = Polygon(polygon["polygon"]).buffer(0)
                polys[channel].append((confidence, poly))
                # 将多边形点转化为int32数组以用于fillPoly
                contour = np.array(polygon["polygon"], dtype=np.int32)
                # 绘制多边形到掩码上
                cv2.fillPoly(mask, [contour], color_map[channel])
    return polys, mask


def get_gt_polygons(regions: dict, classes: list,image_shape) -> dict:
    color_map = {
        'texte': [0, 0, 255],  # Blue
        'figure': [0, 255, 0],  # Green
        'math': [255, 0, 0],  # Red
        'mathstructuree': [0, 255, 255],  # Cyan
        'textemath': [255, 0, 255],  # Magenta
        'mathbarree': [255, 255, 0],  # Yellow
        'background': [128, 128, 128]  # Gray
    }
    polys = {}
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[:, :] = color_map['background']
    # buffer(0) 可以消除或减少由于输入数据质量低而导致的几何错误
    for index, channel in enumerate(classes[:-1], 1):
        polys[channel] = []
        for shape in regions['shapes']:
            if channel == shape['label'].lower():
                # 生成多边形对象，并消除可能的几何错误
                poly = Polygon(shape["points"]).buffer(0)
                polys[channel].append((1, poly))
                # 将多边形点转化为int32数组以用于fillPoly
                contour = np.array(shape["points"], dtype=np.int32)
                # 绘制多边形到掩码上
                cv2.fillPoly(mask, [contour], color_map[channel])

    return polys,mask


# Save the metrics.


def save_results(
        pixel_results: dict, object_results_black: dict,object_results_gray:dict, classes: list, path: str, dataset: str
):
    json_dict = {channel: {} for channel in classes}

    for channel in classes:
        json_dict[channel]["black_iou"] = np.round(np.mean(pixel_results[channel]["black_iou"]), 4)
        json_dict[channel]["black_dice"] = np.round(np.mean(pixel_results[channel]["black_dice"]), 4)
        json_dict[channel]["black_precision"] = np.round(np.mean(pixel_results[channel]["black_precision"]), 4 )
        json_dict[channel]["black_recall"] = np.round( np.mean(pixel_results[channel]["black_recall"]), 4)
        json_dict[channel]["black_fscore"] = np.round(np.mean(pixel_results[channel]["black_fscore"]), 4)
        json_dict[channel]["gray_iou"] = np.round(np.mean(pixel_results[channel]["gray_iou"]), 4)
        json_dict[channel]["gray_dice"] = np.round(np.mean(pixel_results[channel]["gray_dice"]), 4)
        json_dict[channel]["gray_precision"] = np.round(np.mean(pixel_results[channel]["gray_precision"]), 4)
        json_dict[channel]["gray_recall"] = np.round(np.mean(pixel_results[channel]["gray_recall"]), 4)
        json_dict[channel]["gray_fscore"] = np.round(np.mean(pixel_results[channel]["gray_fscore"]), 4)


        aps = object_results_black[channel]["AP"]
        json_dict[channel]["black_AP@[.5]"] = np.round(aps[50], 4)
        json_dict[channel]["black_AP@[.75]"] = np.round(aps[75], 4)
        json_dict[channel]["black_AP@[.95]"] = np.round(aps[95], 4)
        json_dict[channel]["black_AP@[.5,.95]"] = np.round(np.mean(list(aps.values())), 4)

        aps2 = object_results_gray[channel]["AP"]
        precision2=object_results_gray[channel]["precision"]
        recall2=object_results_gray[channel]["recall"]
        fscore2=object_results_gray[channel]["fscore"]
        json_dict[channel]["gray_AP@[.5]"] = np.round(aps2[50], 4)
        json_dict[channel]["gray_precision_average@[.5]"] = np.round(np.mean(list(precision2[50].values())), 4)
        json_dict[channel]["gray_recall_average@[.5]"] =np.round(np.mean(list(recall2[50].values())), 4)
        json_dict[channel]["gray_fscore_average@[.5]"] =np.round(np.mean(list(fscore2[50].values())), 4)

        json_dict[channel]["gray_AP@[.75]"] = np.round(aps2[75], 4)
        json_dict[channel]["gray_precision_average@[.75]"] = np.round(np.mean(list(precision2[75].values())), 4)
        json_dict[channel]["gray_recall_average@[.75]"] = np.round(np.mean(list(recall2[75].values())), 4)
        json_dict[channel]["gray_fscore_average@[.75]"] = np.round(np.mean(list(fscore2[75].values())), 4)

        json_dict[channel]["gray_AP@[.95]"] = np.round(aps2[95], 4)
        json_dict[channel]["gray_precision_average@[.95]"] = np.round(np.mean(list(precision2[95].values())), 4)
        json_dict[channel]["gray_recall_average@[.95]"] = np.round(np.mean(list(recall2[95].values())), 4)
        json_dict[channel]["gray_fscore_average@[.95]"] = np.round(np.mean(list(fscore2[95].values())), 4)


        json_dict[channel]["gray_AP@[.5,.95]"] = np.round(np.mean(list(aps2.values())), 4)


    with open(os.path.join(path, dataset + "_results.json"), "w") as json_file:
        json.dump(json_dict, json_file, indent=4)


def save_graphical_results(results, classes, path,niveaustr,confusion_matrix,confusion_matrix_combined):

    plot_precision_recall_curve(results, classes, path,niveaustr)
    plot_rank_score(results, classes, "Precision", path,niveaustr)
    plot_rank_score(results, classes, "Recall", path,niveaustr)
    plot_rank_score(results, classes, "F-score", path,niveaustr)
    plot_confusion_matrix(confusion_matrix,["texte", "figure", "math", "mathstructuree", "textemath", "mathbarree"],path,'_object_'+niveaustr)
    plot_confusion_matrix(confusion_matrix_combined, ["texte", "figure", "math_total"], path, '_object_combined' + niveaustr)


def generate_figure(params: dict, rotation: bool = None):

    plt.close()
    fig = plt.figure(figsize=params["size"])
    axis = fig.add_subplot(111)
    axis.set_xlabel(params["xlabel"])
    axis.set_ylabel(params["ylabel"])

    axis.set_xticklabels(params["xticks"])
    axis.set_yticklabels(
        params["yticks"],  rotation=rotation, va="center"
    )
    plt.title(params["title"],  fontsize=16, pad=20)
    return fig, axis


def plot_rank_score(scores: dict, classes: list, metric: str, path: str,niveaustr):

    params = {
        "size": (12, 8),
        "title": metric + " vs. confidence score for various IoU thresholds",
        "xlabel": "Confidence score",
        "ylabel": metric,
        "xticks": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        "yticks": [0, 0.2, 0.4, 0.6, 0.8, 1],
    }
    colors = plt.cm.RdPu(np.linspace(0.2, 1, 10))
    for channel in classes:
        # 使用 axis.grid 设置灰色网格线，alpha=0.2 表示网格线透明度较低。
        # 使用 axis.xaxis.set_major_locator(ticker.MultipleLocator(5)) 控制 X 轴的刻度间隔，每隔 5 个单位显示一个刻度。
        _, axis = generate_figure(params)
        axis.grid(color="grey", alpha=0.2)
        axis.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for index, iou in enumerate(range(50, 100, 5)):
            if metric == "Precision":
                score = list(scores[channel]["precision"][iou].values())
                rank = list(scores[channel]["precision"][iou].keys())
            if metric == "Recall":
                score = list(scores[channel]["recall"][iou].values())
                rank = list(scores[channel]["recall"][iou].keys())
            if metric == "F-score":
                score = list(scores[channel]["fscore"][iou].values())
                rank = list(scores[channel]["fscore"][iou].keys())
            axis.plot(
                rank,
                score,
                label="{:.2f}".format(iou / 100),
                alpha=1,
                color=colors[index],
                linewidth=2,
            )
            axis.scatter(
                rank,
                score,
                color=colors[index],
                facecolors="none",
                linewidth=1,
                marker="o",
            )
        axis.set_xlim([49, 96])
        axis.set_ylim([0, 1])
        plt.legend( loc="lower left")
        plt.savefig(
            os.path.join(path, metric + "_" + channel + ".jpg"), bbox_inches="tight"
        )


def plot_precision_recall_curve(object_metrics: dict, classes: list, path: str,niveaustr):
    params = {
        "size": (12, 8),
        "title": f"Precision-recall({niveaustr}) curve for various IoU thresholds",
        "xlabel": "Recall",
        "ylabel": "Precision",
        "xticks": [0, 0.2, 0.4, 0.6, 0.8, 1],
        "yticks": [0, 0.2, 0.4, 0.6, 0.8, 1],
    }
    # 使用 matplotlib 的色彩映射 RdPu（红紫色调）生成一系列颜色，给每个 IoU 阈值的曲线分配不同的颜色。通过 np.linspace(0.2, 1, 10) 生成 10 个从浅到深的颜色，用于不同的 IoU 阈值。
    colors = plt.cm.RdPu(np.linspace(0.2, 1, 10))
    for channel in classes:
        params['title']=str(channel)+'_'+params['title']
        _, axis = generate_figure(params)
        # 使用 axis.grid 设置灰色网格线，alpha=0.2 使得网格线透明度较低。
        axis.grid(color="grey", alpha=0.2)
        for index, iou in enumerate(range(50, 100, 5)):
            current_pr = list(object_metrics[channel]["precision"][iou].values())
            current_rec = list(object_metrics[channel]["recall"][iou].values())
            # 处理重复的 recall 值，计算平均 precision
            recall_precision_map = {}
            for rec, pr in zip(current_rec, current_pr):
                if rec in recall_precision_map:
                    recall_precision_map[rec].append(pr)
                else:
                    recall_precision_map[rec] = [pr]
            # 创建唯一的 recall 列表和对应的平均 precision 列表
            unique_recalls = []
            average_precisions = []

            for rec in sorted(recall_precision_map.keys()):
                unique_recalls.append(rec)
                average_precisions.append(np.mean(recall_precision_map[rec]))

            # 检查数据点数量
            if len(unique_recalls) >= 4:
                # 插值以平滑曲线
                interp_func = interp1d(unique_recalls, average_precisions, kind='cubic', fill_value="extrapolate")
                new_recalls = np.linspace(0, 1, 100)  # 在 [0, 1] 范围内生成100个点
                new_precisions = interp_func(new_recalls)
            else:
                # 数据点不足，不进行插值
                new_recalls = np.array(unique_recalls)
                new_precisions = np.array(average_precisions)



            axis.plot(
                new_recalls,
                new_precisions,
                label="{:.2f}".format(iou / 100),
                alpha=1,
                color=colors[index],
                linewidth=2,
            )
            # 使用 axis.scatter 在曲线上添加散点图，使得 Precision 和 Recall 的数据点在图上更加明显。
            # facecolors="none" 表示散点的填充色为空心，marker="o" 是圆形标记，linewidth=1 表示散点的边框线条宽度。
            axis.scatter(
                unique_recalls,
                average_precisions,
                color=colors[index],
                facecolors="none",
                linewidth=1,
                marker="o",
            )
        #  将 X 轴（Recall）和 Y 轴（Precision） 的范围设置为 [0, 1]，确保图像的显示范围始终在合理区间。
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1])
        plt.legend( loc="lower right")
        # bbox_inches="tight" 表示保存图像时紧贴内容边缘，避免过多空白区域。
        plt.savefig(
            os.path.join(path, "Precision-recall_" + channel + f"({niveaustr}).jpg"),
            bbox_inches="tight",
        )
        params["title"]=f"Precision-recall({niveaustr}) curve for various IoU thresholds"


def plot_pixel_metrics(pixel_metrics, ground_classes_names,path):

    metrics = ["iou", "dice", "precision", "recall", "fscore"]
    pixel_types = ['black_', 'gray_']
    bar_width = 0.35
    index = np.arange(len(metrics))

    for channel in ground_classes_names:
        plt.figure(figsize=(10, 6))

        for i, pixel_type in enumerate(pixel_types):
            # 对n张图的指标做平均
            values = [np.round(np.mean(pixel_metrics[channel][pixel_type + metric]),4) for metric in metrics]

            plt.bar(index + i * bar_width, values, bar_width, label=pixel_type[:-1])

        plt.title(f"Pixel-level Metrics for {channel}")
        plt.xlabel("Metrics")
        plt.ylabel("Metric Value")
        plt.xticks(index + bar_width / 2, metrics)
        plt.legend()
        plt.grid(True, axis='y')


        plt.savefig(os.path.join(path, f"{channel}_pixel_metrics.jpg"))
        plt.close()


def compute_confusion_matrix_pixel(mask_gt,mask_pred,global_confusion_matrix,global_confusion_matrix_combined):

    color_map = {
        (0, 0,255): 0,  # texte
        (0, 255, 0): 1,  # figure
        (255,0, 0): 2,  # math
        (0, 255, 255): 3,  # mathstructuree
        (255, 0, 255): 4,  # textemath
        (255, 255, 0): 5,  # mathbarree
        (128, 128, 128): 6  # background
    }
    color_map2 = {
        (0, 0,255): 0,  # texte
        (0, 255, 0): 1,  # figure
        (255,0, 0): 2,  # math
        (0, 255, 255): 2,  # mathstructuree
        (255, 0, 255): 2,  # textemath
        (255, 255, 0): 2,  # mathbarree
        (128, 128, 128): 3  # background
    }

    label_mask = np.ones(mask_gt.shape[:2], dtype=np.uint8) * 255

    for color, label in color_map.items():
        # 创建一个布尔数组，标记掩码图中与当前颜色相同的位置
        match = np.all(mask_gt == color, axis=-1)
        # print(f"Color: {color}, Label: {label}, Matched Pixels: {np.sum(match)}")
        # 将这些位置对应的标签值赋给二通道掩码图
        label_mask[match] = label


    label_mask_gt_combined = np.ones(mask_gt.shape[:2], dtype=np.uint8) * 255
    for color, label in color_map2.items():
        # 创建一个布尔数组，标记掩码图中与当前颜色相同的位置
        match = np.all(mask_gt == color, axis=-1)
        # print(f"Color: {color}, Label: {label}, Matched Pixels: {np.sum(match)}")
        # 将这些位置对应的标签值赋给二通道掩码图
        label_mask_gt_combined[match] = label

    label_mask_pred = np.ones(mask_pred.shape[:2], dtype=np.uint8) * 255
    for color, label in color_map.items():
        # 创建一个布尔数组，标记掩码图中与当前颜色相同的位置
        match = np.all(mask_pred == color, axis=-1)
        # print(f"Color: {color}, Label: {label}, Matched Pixels: {np.sum(match)}")
        # 将这些位置对应的标签值赋给二通道掩码图
        label_mask_pred[match] = label

    label_mask_pred_combined = np.ones(mask_pred.shape[:2], dtype=np.uint8) * 255
    for color, label in color_map2.items():
        # 创建一个布尔数组，标记掩码图中与当前颜色相同的位置
        match = np.all(mask_pred == color, axis=-1)
        # print(f"Color: {color}, Label: {label}, Matched Pixels: {np.sum(match)}")
        # 将这些位置对应的标签值赋给二通道掩码图
        label_mask_pred_combined[match] = label



    # unique_true, counts_true = np.unique(label_mask, return_counts=True)
    # unique_pred, counts_pred = np.unique(label_mask_pred, return_counts=True)
    # 通过向量化计算混淆矩阵
    true_classes = label_mask.flatten()
    pred_classes = label_mask_pred.flatten()

    # 特殊情况处理
    pred_classes[(true_classes > 2) & (true_classes<6) & (pred_classes == 2)] = true_classes[(true_classes > 2) & (true_classes<6) & (pred_classes == 2)]

    # 计算混淆矩阵
    for true_class, pred_class in zip(true_classes, pred_classes):
        global_confusion_matrix[true_class][pred_class] += 1

    true_classes2=label_mask_gt_combined.flatten()
    pred_classes2=label_mask_pred_combined.flatten()
    for true_class2, pred_class2 in zip(true_classes2, pred_classes2):
        global_confusion_matrix_combined[true_class2][pred_class2] += 1


    return global_confusion_matrix,global_confusion_matrix_combined


def plot_confusion_matrix(confusion_matrix, classes,path,str=''):
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    # 避免除零错误
    row_sums[row_sums == 0] = 1
    confusion_matrix_pixel_normalized = confusion_matrix / row_sums
    plt.close()
    plt.figure(figsize=(20, 15))

    # 使用Seaborn的heatmap绘制混淆矩阵
    sns.heatmap(confusion_matrix_pixel_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes,annot_kws={"size": 16})

    # 设置图的标签

    plt.title(f"Normalized Confusion Matrix {str}", fontsize=20)
    plt.xlabel("Predicted Class", fontsize=16)
    plt.ylabel("True Class", fontsize=16)
    plt.xticks(rotation=0, fontsize=14)  # 调整x轴标签字体大小
    plt.yticks(rotation=0, fontsize=14)

    # 调整图形布局，避免标签被遮挡
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"confusion_matrix_{str}.jpg"),bbox_inches='tight')