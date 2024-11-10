import math
import os
from PIL import Image
import json
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def split_image_vertically(image_path, output_folder):
    # Open the image
    img = Image.open(image_path)

    # Calculate the middle of the image
    width_middle = img.width // 2

    # Split the image in half
    g_img = img.crop((0, 0, width_middle, img.height))
    d_img = img.crop((width_middle, 0, img.width, img.height))

    # Create file names for the halves
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    g_file = os.path.join(output_folder, f"{base_name}_g.jpg")
    d_file = os.path.join(output_folder, f"{base_name}_d.jpg")

    # Save both halves
    g_img.save(g_file)
    d_img.save(d_file)

def split_image_vertically_folder(source_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the folder
    for file in os.listdir(source_folder):
        full_path = os.path.join(source_folder, file)

        # Check if it is a file and an image
        if os.path.isfile(full_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            split_image_vertically(full_path, output_folder)


def remove_bottom_image(image_path, pixels_to_remove, output_folder):
    with Image.open(image_path) as img:
        # Calculate the new dimensions
        width, height = img.size
        new_height = height - pixels_to_remove

        # Crop the image
        cropped_img = img.crop((0, 0, width, new_height))

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save the modified image
        filename = os.path.basename(image_path)
        cropped_img.save(os.path.join(output_folder, filename))

def remove_Bottom_image_folder(source_folder, output_folder, pixels_to_remove):
    for file in os.listdir(source_folder):
        full_path = os.path.join(source_folder, file)
        if os.path.isfile(full_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            remove_bottom_image(full_path, pixels_to_remove, output_folder)




def get_background(images_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(images_dir) if
                   os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]


    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)

        def random_fermeture(image):
            kernel = np.ones((7, 7), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=2)
            return image

        output_path = os.path.join(output_dir, f'background_{image_file}')
        # 获取图像的尺寸
        height, width, _ = image.shape
        # 确定中心裁剪的起始和结束坐标
        crop_size = 2048
        # 计算中心点
        center_x, center_y = width // 2, height // 2
        # 计算裁剪区域的起点和终点
        start_x = max(center_x - crop_size // 2, 0)  # 确保不会超出边界
        start_y = max(center_y - crop_size // 2, 0)
        end_x = min(start_x + crop_size, width)
        end_y = min(start_y + crop_size, height)
        cropped_background = image[start_y:end_y, start_x:end_x]
        background = random_fermeture(cropped_background)
        cv2.imwrite(output_path, background)
        print(f'Saved background for {image_file} to {output_path}')

def combine_loss_figure(path1,path2,path3=None,path4=None):

    # 读取数据
    df1 = pd.read_csv(path1)
    step1 = df1['Step'].values.tolist()
    loss1 = df1['Value'].values.tolist()

    df2 = pd.read_csv(path2)
    step2 = df2['Step'].values.tolist()
    loss2 = df2['Value'].values.tolist()

    # 检查是否有学习率文件传入
    if path3 or path4:
        df3 = pd.read_csv(path3)
        step3 = df3['Step'].values.tolist()
        loss3 = df3['Value'].values.tolist()
        df4 = pd.read_csv(path4)
        step4 = df4['Step'].values.tolist()
        loss4 = df4['Value'].values.tolist()
    else:
        step3, loss3 = None, None

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 绘制 loss 曲线 (左边y轴)
    # ax1.plot(step1, loss1, label='Recall (Contenu: Sans contenu=1:0.5)')
    # ax1.plot(step2, loss2, label='Recall (Contenu: Sans contenu=0.5:1)')
    # ax1.plot(step1, loss1, label='Precision (Contenu: Sans contenu=1:0.5)')
    # ax1.plot(step2, loss2, label='Precision (Contenu: Sans contenu=0.5:1)')
    ax1.plot(step1, loss1, label='F1 (Contenu: Sans contenu=1:0.5)')
    ax1.plot(step2, loss2, label='F1 (Contenu: Sans contenu=0.5:1)')
    # ax1.plot(step1, loss1, label='Overall ACC')
    # ax1.plot(step1, loss1, label='Training Loss')
    # ax1.plot(step1, loss1, label='Validation Loss (Contenu: Sans contenu=1:0.5)')
    # ax1.plot(step2, loss2, label='Validation Loss (Contenu: Sans contenu=0.5:1)')


    ax1.set_xlabel('Epoch', fontsize=14)
    # ax1.set_ylabel('Loss Value', fontsize=14, color='b')
    # ax1.set_ylabel('Recall Value', fontsize=14, color='b')
    # ax1.set_ylabel('Precision Value', fontsize=14, color='b')
    ax1.set_ylabel('F1 Value', fontsize=14, color='b')
    # ax1.set_ylabel('Overall ACC Value', fontsize=14, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 如果有 learning rate 数据，则创建第二个 y 轴
    if step3 and loss3:
        ax2 = ax1.twinx()  # 创建共享 x 轴的第二个 y 轴
        ax2.plot(step3, loss3, label='Learning Rate (Contenu: Sans contenu=1:0.5)',  linestyle='--')
        ax2.plot(step4, loss4, label='Learning Rate (Contenu: Sans contenu=0.5:1)', linestyle='--')
        ax2.set_ylabel('Learning Rate', fontsize=14, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='lower left', fontsize=12)

    # 设置 x 轴范围
    step_min = min(step1 + step2)
    step_max = max(step1 + step2)
    # step_min = min(step1)
    # step_max = max(step1)
    ax1.set_xticks(np.arange(step_min, step_max + 1, 10))  # x 轴每隔 10 个 step 标注一次
    # 将 x 轴的刻度标签旋转 45 度
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # 设置 y 轴范围
    loss_min = min(loss1 + loss2)
    loss_max = max(loss1 + loss2)
    # loss_min = min(loss1)
    # loss_max = max(loss1)
    ax1.set_yticks(np.arange(loss_min, loss_max + 0.05, 0.05))  # 设置左边 y 轴 (Loss) 的范围

    # 显示图形
    # plt.title('Loss and Content weight', fontsize=16)
    # plt.title('Recall and different content weight', fontsize=16)
    # plt.title('Precision and different content weight', fontsize=16)
    plt.title('F1 and different content weight', fontsize=16)
    # plt.title('Overall acc and Learning Rate over Epochs', fontsize=16)
    # plt.title('Different Validation Loss and different sampling ratio range', fontsize=16)
    plt.savefig('result.png',bbox_inches='tight')

# def combine_loss_figure(path1,path2,path3,path4,path5,path6):
#
#     # 读取数据
#     df1 = pd.read_csv(path1)
#     step1 = df1['Step'].values.tolist()
#     loss1 = df1['Value'].values.tolist()
#
#     df2 = pd.read_csv(path2)
#     step2 = df2['Step'].values.tolist()
#     loss2 = df2['Value'].values.tolist()
#
#     df3 = pd.read_csv(path3)
#     step3 = df3['Step'].values.tolist()
#     loss3 = df3['Value'].values.tolist()
#
#     # 检查是否有学习率文件传入
#     if path4:
#         df4 = pd.read_csv(path4)
#         step4 = df4['Step'].values.tolist()
#         loss4 = df4['Value'].values.tolist()
#         df5 = pd.read_csv(path5)
#         step5 = df5['Step'].values.tolist()
#         loss5 = df5['Value'].values.tolist()
#         df6 = pd.read_csv(path6)
#         step6 = df6['Step'].values.tolist()
#         loss6 = df6['Value'].values.tolist()
#
#
#
#     # 创建图形
#     fig, ax1 = plt.subplots(figsize=(12, 8))
#
#     # 绘制 loss 曲线 (左边y轴)
#     # ax1.plot(step1, loss1, label='Recall (lr initial=1e-2)')
#     # ax1.plot(step2, loss2, label='Recall (lr initial=1e-3)')
#     # ax1.plot(step3, loss3, label='Recall (lr initial=1e-4)')
#     # ax1.plot(step1, loss1, label='Precision (lr initial=1e-2)')
#     # ax1.plot(step2, loss2, label='Precision (lr initial=1e-3)')
#     # ax1.plot(step3, loss3, label='Precision (lr initial=1e-4)')
#     ax1.plot(step1, loss1, label='F1 (lr initial=1e-2)')
#     ax1.plot(step2, loss2, label='F1 (lr initial=1e-3)')
#     ax1.plot(step3, loss3, label='F1 (lr initial=1e-4)')
#     # ax1.plot(step1, loss1, label='Validation Loss (lr initial=1e-2)')
#     # ax1.plot(step2, loss2, label='Validation Loss (lr initial=1e-3)')
#     # ax1.plot(step3, loss3, label='Validation Loss (lr initial=1e-4)')
#
#
#     ax1.set_xlabel('Epoch', fontsize=14)
#     # ax1.set_ylabel('Loss Value', fontsize=14, color='b')
#     # ax1.set_ylabel('Recall Value', fontsize=14, color='b')
#     # ax1.set_ylabel('Precision Value', fontsize=14, color='b')
#     ax1.set_ylabel('F1 Value', fontsize=14, color='b')
#     # ax1.set_ylabel('Overall ACC Value', fontsize=14, color='b')
#     ax1.tick_params(axis='y', labelcolor='b')
#     ax1.legend(loc='upper left', fontsize=12)
#     ax1.grid(True, linestyle='--', alpha=0.7)
#
#     # 如果有 learning rate 数据，则创建第二个 y 轴
#     if step4 and loss4:
#         ax2 = ax1.twinx()  # 创建共享 x 轴的第二个 y 轴
#         ax2.plot(step4, loss4, label='Learning Rate 1e-2)', linestyle='--')
#         ax2.plot(step5, loss5, label='Learning Rate 1e-3', linestyle='--')
#         ax2.plot(step6, loss6, label='Learning Rate 1e-4', linestyle='--')
#         ax2.set_ylabel('Learning Rate', fontsize=14, color='r')
#         ax2.tick_params(axis='y', labelcolor='r')
#         ax2.legend(loc='lower left', fontsize=12)
#
#     # 设置 x 轴范围
#     step_min = min(step1 + step2+step3)
#     step_max = max(step1 + step2+step3)
#     # step_min = min(step1)
#     # step_max = max(step1)
#     ax1.set_xticks(np.arange(step_min, step_max + 1, 10))  # x 轴每隔 10 个 step 标注一次
#     # 将 x 轴的刻度标签旋转 45 度
#     plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
#
#     # 设置 y 轴范围
#     loss_min = min(loss1 + loss2+loss3)
#     loss_max = max(loss1 + loss2+loss3)
#     # loss_min = min(loss1)
#     # loss_max = max(loss1)
#     ax1.set_yticks(np.arange(loss_min, loss_max + 0.05, 0.05))  # 设置左边 y 轴 (Loss) 的范围
#
#     # 显示图形
#     # plt.title('validation Loss and different Learning Rate over Epochs', fontsize=16)
#     # plt.title('Different Recall and different lr over Epochs', fontsize=16)
#     # plt.title('Different Precision and different lr over Epochs', fontsize=16)
#     plt.title('Different F1 and different lr over Epochs', fontsize=16)
#     # plt.title('Overall acc and Learning Rate over Epochs', fontsize=16)
#     # plt.title('Validation Loss and Accumulation Validation Loss over Epochs', fontsize=16)
#     plt.savefig('result.png',bbox_inches='tight')
def collect_all_elements(jsons, folder):
    all_elements = []
    for json_name in jsons:
        with open(os.path.join(folder, json_name), 'r') as f:
            annotations = json.load(f)
        for annotation in annotations['shapes']:
            element = (os.path.join(folder, json_name.replace('.json', '.jpg')), annotation)
            all_elements.append(element)
    return all_elements
def watch_annotations(folder,imgname=None):
    files = os.listdir(folder)
    images = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if imgname is None:
        jsons = [f for f in files if f.endswith('.json')]
    else:
        jsons=[os.path.join(folder,imgname+'.jpg'.replace('jpg','json'))]
    all_elements = collect_all_elements(jsons, folder)
    selected_elements=[]
    for img_path, annotation in all_elements:
        label = annotation['label'].lower()
        if 'math' == label.lower():
            selected_elements.append((img_path, annotation))
        elif 'textemath' == label.lower():
            selected_elements.append((img_path, annotation))
        elif 'mathstructuree' == label.lower():
            selected_elements.append((img_path, annotation))
        elif 'mathbarree' == label.lower():
            selected_elements.append((img_path, annotation))

    elements_with_areas=[]
    areamin=100000000000000000000000000
    for img_path, element in selected_elements:
        image = cv2.imread(img_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = np.array(element['points'], dtype=np.int32)
        polygon_area = cv2.contourArea(points)
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        cropped_element = image[y:y+h, x:x+w]
        area = cropped_element.shape[0] * cropped_element.shape[1]
        if polygon_area<areamin:
            element_min=cropped_element
            areamin=polygon_area
            rect_area=area
            currentname=img_path
        # elements_with_areas.append((cropped_element, area,cropped_element.shape[1],cropped_element.shape[0]))

    # sorted_elements = sorted(elements_with_areas, key=lambda x: x[1])
    plt.figure(1)
    plt.imshow(element_min)
    plt.title(f'Minimum Polygon area in Math:{areamin}')
    plt.savefig('minelement.png')
    print('minarea:',areamin)

watch_annotations('../AnnotedImage')
# combine_loss_figure(r"C:\Users\86139\Downloads\newdatalossbszacc8syn(lossmodifiestableratio\f1.csv",r"C:\Users\86139\Downloads\newdatalossbszaccumulation8\F1.csv",r"C:\Users\86139\Downloads\newdatalossbszacc8syn(lossmodifiestableratio\lr.csv",r"C:\Users\86139\Downloads\newdatalossbszaccumulation8\LR.csv")
# get_background('../AnnotedImage', './Background')
pixels_to_remove = 315
# remove_Bottom_image_folder(r'C:\Users\86139\Desktop\images2', r'C:\Users\86139\Desktop\images2', pixels_to_remove)
# split_image_vertically_folder(r'C:\Users\86139\Desktop\images2', r'C:\Users\86139\Desktop\images2_traitement')