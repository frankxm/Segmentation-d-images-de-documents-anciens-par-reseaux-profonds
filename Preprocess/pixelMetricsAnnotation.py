
import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

import os
import json
import shutil

def countannotation(image_folder):
    total_text=0
    total_figure=0
    total_math=0
    total_mathstructuree=0
    total_textemath=0
    total_mathbarree = 0

    with open(os.path.join(image_folder, 'annotation_counts.txt'), 'w') as outfile:
        for file in os.listdir(image_folder):
            if file.endswith('.json'):
                json_path = os.path.join(image_folder, file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    annotations = data['shapes']
                text_count = 0
                figure_count = 0
                math_count = 0
                mathstructuree_count = 0
                textemath_count = 0
                mathbarree_count = 0
                for annotation in annotations:
                    label_lower = annotation['label'].lower()
                    if label_lower == 'texte':
                        text_count += 1
                    elif label_lower == 'figure':
                        figure_count += 1
                    elif label_lower == 'math':
                        math_count += 1
                    elif label_lower == 'mathstructuree':
                        mathstructuree_count += 1
                    elif label_lower == 'textemath':
                        textemath_count += 1
                    elif label_lower == 'mathbarree':
                        mathbarree_count += 1

                total_text += text_count
                total_figure += figure_count
                total_math += math_count
                total_mathstructuree += mathstructuree_count
                total_textemath += textemath_count
                total_mathbarree += mathbarree_count

                result_line = (
                    f"{file.replace('.json', '')}: text = {text_count}, figure = {figure_count}, "
                    f"math = {math_count}, mathstructuree = {mathstructuree_count}, "
                    f"textemath = {textemath_count}, mathbarree = {mathbarree_count}, math_total ={math_count+mathstructuree_count+textemath_count+mathbarree_count}\n"
                )
                print(result_line.strip())
                outfile.write(result_line)
        total_result = (
            f"Total texte annotations in dataset: {total_text}\n"
            f"Total figure annotations in dataset: {total_figure}\n"
            f"Total math annotations in dataset: {total_math}\n"
            f"Total mathstructuree annotations in dataset: {total_mathstructuree}\n"
            f"Total textemath annotations in dataset: {total_textemath}\n"
            f"Total mathbarree annotations in dataset: {total_mathbarree}\n"
            f"Total math_all annotations in dataset: {total_math+total_mathstructuree+total_textemath+total_mathbarree}\n"
        )
        print(total_result.strip())
        outfile.write(total_result)



def analyze(image_folder,labelmap_folder):
    results = []
    with open(os.path.join(image_folder, 'analysis_results.txt'), 'w') as outfile:

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue

                base_name, _ = os.path.splitext(filename)
                print(base_name)
                maskname = base_name + '_mask'
                mask_path = os.path.join(labelmap_folder, maskname + '.png')
                mask = cv2.imread(mask_path)
                if mask is None:
                    continue

                color_map = {
                    (255, 0, 0): 0,  # texte
                    (0, 255, 0): 1,  # figure
                    (0, 0, 255): 2,  # math
                    (255, 255, 0): 3,  # mathstructuree
                    (255, 0, 255): 4,  # textemath
                    (0, 255, 255): 5,  # mathbarree
                    (128, 128, 128): 6  # background
                }

                label_mask = np.ones(mask.shape[:2], dtype=np.uint8) * 255

                for color, label in color_map.items():
                    # 创建一个布尔数组，标记掩码图中与当前颜色相同的位置
                    match = np.all(mask == color, axis=-1)
                    print(f"Color: {color}, Label: {label}, Matched Pixels: {np.sum(match)}")
                    # 将这些位置对应的标签值赋给二通道掩码图
                    label_mask[match] = label


                text_pixels = np.sum(label_mask == 0)
                figure_pixels = np.sum(label_mask == 1)
                math_pixels = np.sum(label_mask == 2)
                mathstructuree_pixels = np.sum(label_mask == 3)
                textemath_pixels = np.sum(label_mask == 4)
                mathbarree_pixels = np.sum(label_mask == 5)
                background_pixels = np.sum(label_mask==6)
                total_pixels_except_background = text_pixels + figure_pixels + math_pixels + mathstructuree_pixels + textemath_pixels + mathbarree_pixels
                total_pixels = image.shape[0] * image.shape[1]
                math_total_pixels=math_pixels+mathstructuree_pixels+textemath_pixels+mathbarree_pixels

                results.append({
                    'image': filename,
                    'text_proportion':                             text_pixels / total_pixels,
                    'figure_proportion':                           figure_pixels / total_pixels,
                    'math_proportion':                             math_pixels / total_pixels,
                    'mathstructuree_proportion':                   mathstructuree_pixels / total_pixels,
                    'textemath_proportion':                        textemath_pixels / total_pixels,
                    'mathbarree_proportion':                       mathbarree_pixels / total_pixels,
                    'background_proportion':                       background_pixels / total_pixels,
                    'math_total_proportion':                       math_total_pixels / total_pixels,
                    'text_proportion_except_background':           text_pixels / total_pixels_except_background,
                    'figure_proportion_except_background':         figure_pixels / total_pixels_except_background,
                    'math_proportion_except_background':           math_pixels / total_pixels_except_background,
                    'mathstructuree_proportion_except_background': mathstructuree_pixels / total_pixels_except_background,
                    'textemath_proportion_except_background':      textemath_pixels / total_pixels_except_background,
                    'mathbarree_proportion_except_background':     mathbarree_pixels / total_pixels_except_background,
                    'math_total_proportion_except_background':      math_total_pixels / total_pixels_except_background
                })

        for result in results:
            outfile.write(f"Image: {result['image']}\n%")
            outfile.write(f"Text Proportion: {round(100 * result['text_proportion'], 2)}%\n")
            outfile.write(f"Figure Proportion: {round(100 * result['figure_proportion'], 2)}%\n")
            outfile.write(f"Math Proportion: {round(100 * result['math_proportion'], 2)}%\n")
            outfile.write(f"Math Structure Proportion: {round(100 * result['mathstructuree_proportion'], 2)}%\n")
            outfile.write(f"Text Math Proportion: {round(100 * result['textemath_proportion'], 2)}%\n")
            outfile.write(f"Math Barree Proportion: {round(100 * result['mathbarree_proportion'], 2)}%\n")
            outfile.write(f"Background Proportion: {round(100 * result['background_proportion'], 2)}%\n")
            outfile.write(f"Math total Proportion: {round(100 * result['math_total_proportion'], 2)}%\n")
            outfile.write(
                f"Text Proportion Except Background: {round(100 * result['text_proportion_except_background'], 2)}%\n")
            outfile.write(
                f"Figure Proportion Except Background: {round(100 * result['figure_proportion_except_background'], 2)}%\n")
            outfile.write(
                f"Math Proportion Except Background: {round(100 * result['math_proportion_except_background'], 2)}%\n")
            outfile.write(
                f"Math Structure Proportion Except Background: {round(100 * result['mathstructuree_proportion_except_background'], 2)}%\n")
            outfile.write(
                f"Text Math Proportion Except Background: {round(100 * result['textemath_proportion_except_background'], 2)}%\n")
            outfile.write(
                f"Math Barree Proportion Except Background: {round(100 * result['mathbarree_proportion_except_background'], 2)}%\n")
            outfile.write(
                f"Math Total Proportion Except Background: {round(100 * result['math_total_proportion_except_background'], 2)}%\n")
            outfile.write("\n")
    averages = {
        'average_text': np.mean([r['text_proportion'] for r in results]),
        'average_figure': np.mean([r['figure_proportion'] for r in results]),
        'average_math': np.mean([r['math_proportion'] for r in results]),
        'average_mathstructuree': np.mean([r['mathstructuree_proportion'] for r in results]),
        'average_textemath': np.mean([r['textemath_proportion'] for r in results]),
        'average_mathbarree': np.mean([r['mathbarree_proportion'] for r in results]),
        'average_background': np.mean([r['background_proportion'] for r in results]),
        'average_math_total': np.mean([r['math_total_proportion'] for r in results]),
        'average_text_except_background':  np.mean([r['text_proportion_except_background'] for r in results]),
        'average_figure_except_background': np.mean([r['figure_proportion_except_background'] for r in results]),
        'average_math_except_background':  np.mean([r['math_proportion_except_background'] for r in results]),
        'average_mathstructuree_except_background':  np.mean([r['mathstructuree_proportion_except_background'] for r in results]),
        'average_textemath_except_background': np.mean([r['textemath_proportion_except_background'] for r in results]),
        'average_mathbarree_except_background':  np.mean([r['mathbarree_proportion_except_background'] for r in results]),
        'average_math_total_except_background': np.mean([r['math_total_proportion_except_background'] for r in results])
    }
    with open(os.path.join(image_folder, 'average_results.txt'), 'w') as outfile:
        outfile.write("\nAverage Results:\n")
        outfile.write(f"Average Text Proportion: {round(100*averages['average_text'],2)}%\n")
        outfile.write(f"Average Figure Proportion: {round(100*averages['average_figure'],2)}%\n")
        outfile.write(f"Average Math Proportion: {round(100*averages['average_math'],2)}%\n")
        outfile.write(f"Average Math Structure Proportion: {round(100*averages['average_mathstructuree'],2)}%\n")
        outfile.write(f"Average Text Math Proportion: {round(100*averages['average_textemath'],2)}%\n")
        outfile.write(f"Average Math Barree Proportion: {round(100*averages['average_mathbarree'],2)}%\n")
        outfile.write(f"Average Math Total Proportion: {round(100 * averages['average_math_total'], 2)}%\n")
        outfile.write(f"Average Background Proportion: {round(100 * averages['average_background'], 2)}%\n")
        outfile.write(f"Average Text Proportion Except Background: {round(100*averages['average_text_except_background'],2)}%\n")
        outfile.write(f"Average Figure Proportion Except Background: {round(100*averages['average_figure_except_background'],2)}%\n")
        outfile.write(f"Average Math Proportion Except Background: {round(100*averages['average_math_except_background'],2)}%\n")
        outfile.write(f"Average Math Structure Proportion Except Background: {round(100*averages['average_mathstructuree_except_background'],2)}%\n")
        outfile.write(f"Average Text Math Proportion Except Background: {round(100*averages['average_textemath_except_background'],2)}%\n")
        outfile.write(f"Average Math Barree Proportion Except Background: {round(100*averages['average_mathbarree_except_background'],2)}%\n")
        outfile.write(f"Average Math Total Proportion Except Background: {round(100 * averages['average_math_total_except_background'], 2)}%\n")
        outfile.write("\n")



def split_dataset(image_folder,labelmap_folder ,data_folder,background_folder,train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # 读取分析结果
    results_file = os.path.join(image_folder, 'analysis_results.txt')
    results = []

    with open(results_file, 'r') as infile:
        lines = infile.readlines()
        for i in range(0, len(lines), 17):  # 每个图像有 17 行结果
            image_data = {}
            image_data['image'] = lines[i].split(': ')[1].strip()
            image_data['text_proportion'] = float(lines[i + 1].split(': ')[1].replace('%', '').strip()) / 100
            image_data['figure_proportion'] = float(lines[i + 2].split(': ')[1].replace('%', '').strip()) / 100
            image_data['math_proportion'] = float(lines[i + 8].split(': ')[1].replace('%', '').strip()) / 100
            image_data['background_proportion'] = float(lines[i + 7].split(': ')[1].replace('%', '').strip()) / 100
            results.append(image_data)

    # 计算总的类别比例
    total_text = np.mean([r['text_proportion'] for r in results])
    total_figure = np.mean([r['figure_proportion'] for r in results])
    total_math = np.mean([r['math_proportion'] for r in results])
    total_background = np.mean([r['background_proportion'] for r in results])


    target_distribution = {
        'text': 0.1,
        'figure': 0.2,
        'math': 0.6,
        'background': 0.1
    }

    # 初始化集合
    train_set, val_set, test_set = [], [], []
    train_size = int(len(results) * train_ratio)
    val_size = int(len(results) * val_ratio)
    test_size = int(len(results) * test_ratio)
    # 初始化类别比例统计
    train_distribution = {'text': 0, 'figure': 0, 'math': 0, 'background': 0}
    val_distribution = {'text': 0, 'figure': 0, 'math': 0, 'background': 0}
    test_distribution = {'text': 0, 'figure': 0, 'math': 0, 'background': 0}

    # 分配图像
    # 当我们将某个图像加入到一个子集时，该子集的类别比例会发生变化。我们希望这种变化不会让该子集的类别比例偏离目标比例太多。因此，对于每一张图像，我们会依次尝试将它加入训练集、验证集和测试集，并计算加入后的类别比例偏差，选择使得偏差最小的子集。
    for image_data in results:
        current_text = image_data['text_proportion']
        current_figure = image_data['figure_proportion']
        current_math = image_data['math_proportion']
        current_background = image_data['background_proportion']

        # 计算分配前后的差异
        def compute_difference(target, current, subset_distribution, ratio, subset_size, max_size):
            # 当子集已满时，不再考虑分配到该子集
            if subset_size >= max_size:
                return float('inf')
            return abs(
                (subset_distribution + current) / (subset_size + 1) - target * ratio)

        train_diff = compute_difference(target_distribution['text'], current_text, train_distribution['text'],train_ratio, len(train_set), train_size) + \
                     compute_difference(target_distribution['figure'], current_figure, train_distribution['figure'],train_ratio, len(train_set), train_size) + \
                     compute_difference(target_distribution['math'], current_math, train_distribution['math'],train_ratio, len(train_set), train_size) + \
                     compute_difference(target_distribution['background'], current_background,train_distribution['background'], train_ratio, len(train_set), train_size)

        val_diff = compute_difference(target_distribution['text'], current_text, val_distribution['text'], val_ratio, len(val_set), val_size) + \
                   compute_difference(target_distribution['figure'], current_figure, val_distribution['figure'],val_ratio, len(val_set), val_size) + \
                   compute_difference(target_distribution['math'], current_math, val_distribution['math'], val_ratio, len(val_set), val_size) + \
                   compute_difference(target_distribution['background'], current_background, val_distribution['background'], val_ratio, len(val_set), val_size)

        test_diff = compute_difference(target_distribution['text'], current_text, test_distribution['text'], test_ratio, len(test_set), test_size) + \
                    compute_difference(target_distribution['figure'], current_figure, test_distribution['figure'],test_ratio, len(test_set), test_size) + \
                    compute_difference(target_distribution['math'], current_math, test_distribution['math'], test_ratio,len(test_set), test_size) + \
                    compute_difference(target_distribution['background'], current_background, test_distribution['background'], test_ratio, len(test_set), test_size)

        # 将图像分配到差异最小的集合中
        if train_diff <= val_diff and train_diff <= test_diff:
            train_set.append(image_data['image'])
            train_distribution['text'] += current_text
            train_distribution['figure'] += current_figure
            train_distribution['math'] += current_math
            train_distribution['background'] += current_background
        elif val_diff <= train_diff and val_diff <= test_diff:
            val_set.append(image_data['image'])
            val_distribution['text'] += current_text
            val_distribution['figure'] += current_figure
            val_distribution['math'] += current_math
            val_distribution['background'] += current_background
        else:
            test_set.append(image_data['image'])
            test_distribution['text'] += current_text
            test_distribution['figure'] += current_figure
            test_distribution['math'] += current_math
            test_distribution['background'] += current_background

    # 打印划分结果
    print(f"Training set: {len(train_set)} images:{train_set}")
    print(f"Validation set: {len(val_set)} images:{val_set}")
    print(f"Test set: {len(test_set)} images:{test_set}")
    # 计算每个子集的类别比例
    train_proportions = calculate_subset_proportions(train_set)
    val_proportions = calculate_subset_proportions(val_set)
    test_proportions = calculate_subset_proportions(test_set)
    # 打印训练集、验证集和测试集的类别比例
    print_proportions(train_proportions, "Training Set",image_folder,train_set)
    print_proportions(val_proportions, "Validation Set",image_folder,val_set)
    print_proportions(test_proportions, "Test Set",image_folder,test_set)
    get_background(image_folder,background_folder)
    make_data_dir('training')
    deplace_data(train_set,image_folder,labelmap_folder,data_folder,'train')
    deplace_data(val_set,image_folder,labelmap_folder,data_folder,'val')
    deplace_data(test_set,image_folder,labelmap_folder,data_folder,'test')


    return train_set, val_set, test_set
def make_data_dir(datasetname):
    datadir = os.path.join('../Doc-ufcn/Data', datasetname)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
        listname=['train','val','test']
        listname2=['images','labels','labels_json']
        for name1 in listname:
            for name2 in listname2:
                os.makedirs(os.path.join(datadir,name1,name2))
def get_background(images_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    image_files = [f for f in os.listdir(images_dir) if
                   os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]


    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)

        def random_fermeture(image):
            kernel = np.ones((9, 9), np.uint8)
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
def deplace_data(set,image_folder,labelmap_folder,data_folder,dataname):
    print(f'distribution of {dataname}')

    for imgname in set:
        imgpath=os.path.join(image_folder,imgname)
        target_path=os.path.join(data_folder,dataname,'images',imgname)
        shutil.copy(imgpath, target_path)
        jsonname=imgname.replace('.jpg','.json')
        jsonpath=imgpath.replace('.jpg','.json')
        target_pathjson = os.path.join(data_folder, dataname,'labels_json', jsonname)
        shutil.copy(jsonpath,target_pathjson)
        labelname=imgname.split('.')[0]+'_mask.png'
        labelpath=os.path.join(labelmap_folder,labelname)
        target_pathlabel=os.path.join(data_folder,dataname,'labels',labelname)
        shutil.copy(labelpath,target_pathlabel)



def calculate_subset_proportions(subset):
    total_text_pixels = 0
    total_figure_pixels = 0
    total_math_pixels = 0
    total_mathstructuree_pixels = 0
    total_textemath_pixels = 0
    total_mathbarree_pixels = 0
    total_background_pixels = 0
    total_math_all_pixels=0
    total_pixels = 0
    color_map = {
        (255, 0, 0): 0,  # texte
        (0, 255, 0): 1,  # figure
        (0, 0, 255): 2,  # math
        (255, 255, 0): 3,  # mathstructuree
        (255, 0, 255): 4,  # textemath
        (0, 255, 255): 5,  # mathbarree
        (128, 128, 128): 6  # background
    }
    for imgname in subset:
        base_name, _ = os.path.splitext(imgname)
        print(base_name)
        maskname = base_name + '_mask'
        mask_path = os.path.join(labelmap_folder, maskname + '.png')
        mask = cv2.imread(mask_path)
        label_mask = np.ones(mask.shape[:2], dtype=np.uint8) * 255
        for color, label in color_map.items():
            # 创建一个布尔数组，标记掩码图中与当前颜色相同的位置
            match = np.all(mask == color, axis=-1)
            # print(f"Color: {color}, Label: {label}, Matched Pixels: {np.sum(match)}")
            # 将这些位置对应的标签值赋给二通道掩码图
            label_mask[match] = label
        total_text_pixels += np.sum(label_mask == 0)
        total_figure_pixels += np.sum(label_mask == 1)
        total_math_pixels += np.sum(label_mask == 2)
        total_mathstructuree_pixels += np.sum(label_mask == 3)
        total_textemath_pixels += np.sum(label_mask == 4)
        total_mathbarree_pixels += np.sum(label_mask == 5)
        total_background_pixels += np.sum(label_mask == 6)
        total_math_all_pixels+=total_math_pixels+total_textemath_pixels+total_mathstructuree_pixels+total_mathbarree_pixels
        total_pixels += label_mask.size

    return {
        'text_proportion': total_text_pixels / total_pixels,
        'figure_proportion': total_figure_pixels / total_pixels,
        'math_proportion': total_math_pixels / total_pixels,
        'mathstructuree_proportion': total_mathstructuree_pixels / total_pixels,
        'textemath_proportion': total_textemath_pixels / total_pixels,
        'mathbarree_proportion': total_mathbarree_pixels / total_pixels,
        'background_proportion': total_background_pixels / total_pixels,
        'math_total_proportion': (total_math_pixels + total_mathstructuree_pixels + total_textemath_pixels + total_mathbarree_pixels) / total_pixels
    }




def print_proportions(proportions, subset_name,image_folder,subset):
    # 打印划分结果
    print(f"{subset_name}: {len(subset)} images:{subset}")
    print(f"{subset_name} Proportions:")
    print(f"Text Proportion: {round(100 * proportions['text_proportion'], 2)}%")
    print(f"Figure Proportion: {round(100 * proportions['figure_proportion'], 2)}%")
    print(f"Math Proportion: {round(100 * proportions['math_proportion'], 2)}%")
    print(f"Math Structure Proportion: {round(100 * proportions['mathstructuree_proportion'], 2)}%")
    print(f"Text Math Proportion: {round(100 * proportions['textemath_proportion'], 2)}%")
    print(f"Math Barree Proportion: {round(100 * proportions['mathbarree_proportion'], 2)}%")
    print(f"Background Proportion: {round(100 * proportions['background_proportion'], 2)}%")
    print(f"Math Total Proportion: {round(100 * proportions['math_total_proportion'], 2)}%\n")
    with open(os.path.join(image_folder, f'{subset_name}.txt'), 'w') as outfile:
        outfile.write(f"Average Results of classe proportion in {subset_name}:\n")
        outfile.write(f"{subset_name}: {len(subset)} \nimages:{subset}\n")
        outfile.write(f"{subset_name} Proportions:\n")
        outfile.write(f"Text Proportion: {round(100 * proportions['text_proportion'], 2)}%\n")
        outfile.write(f"Figure Proportion: {round(100 * proportions['figure_proportion'], 2)}%\n")
        outfile.write(f"Math Proportion: {round(100 * proportions['math_proportion'], 2)}%\n")
        outfile.write(f"Math Structure Proportion: {round(100 * proportions['mathstructuree_proportion'], 2)}%\n")
        outfile.write(f"Text Math Proportion: {round(100 * proportions['textemath_proportion'], 2)}%\n")
        outfile.write(f"Math Barree Proportion: {round(100 * proportions['mathbarree_proportion'], 2)}%\n")
        outfile.write(f"Background Proportion: {round(100 * proportions['background_proportion'], 2)}%\n")
        outfile.write(f"Math Total Proportion: {round(100 * proportions['math_total_proportion'], 2)}%\n")
        outfile.write("\n")




def readImageMask(image_folder):

    for filename in os.listdir(image_folder):
        polygons_dict = {}
        polygons_data = []
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            width, height = image.shape[1], image.shape[0]

            if image is None:
                continue

            base_name, _ = os.path.splitext(filename)
            print('{}'.format(base_name), width, height)
            json_path = os.path.join(image_folder, f"{base_name}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as f:
                annotations = json.load(f)

            for shape in annotations['shapes']:
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
                    cv2.polylines(image, [currentpolypoints], isClosed=True, color=color_map[currentlabel], thickness=5)
                    cv2.fillPoly(mask_structures, [otherpolypoints], color_map[otherlabel])
                    cv2.polylines(image, [otherpolypoints], isClosed=True, color=color_map[otherlabel], thickness=5)
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
                cv2.polylines(image, [polypoints], isClosed=True, color=color_map[label], thickness=5)

            combine = cv2.addWeighted(image, 0.7, mask_structures, 0.3, 0)
            # cv2.namedWindow("imagemask", cv2.WINDOW_NORMAL)
            # cv2.imshow('imagemask', combine)
            # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            # cv2.imshow('image', image)
            # cv2.namedWindow("mask2", cv2.WINDOW_NORMAL)
            # cv2.imshow('mask2', mask_structures)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 保存掩码图像时，通常使用的是PNG格式，因为PNG格式支持无损压缩，并且可以保存透明通道。对于掩码图像来说，保存为PNG格式是比较常见和推荐的做法，因为它能够保持图像中每个像素的精确数值，不会改变像素的颜色值。
            # JPG格式则是有损压缩的格式，适合保存真彩色或灰度图像，但是不适合保存掩码图像，因为它会对图像进行压缩并且会引入一定的损失，可能导致掩码图像的像素值发生变化，特别是对边缘和细节部分的像素值影响比较明显。
            # 否则会影响到后续像素比例的计算
            output_image_path = os.path.join('./Labelmap', f"{base_name}_processed.png")
            cv2.imwrite(output_image_path, combine)

            mask_output_path = os.path.join('./Labelmap', f"{base_name}_mask.png")
            cv2.imwrite(mask_output_path, mask_structures)



def change_data(new_polygon,polygons_data,index,label,base_name):
    # 处理 new_polygon 的情况
    if isinstance(new_polygon, Polygon):
        polygons_data[index] = (label, new_polygon, base_name, np.array(new_polygon.exterior.coords, dtype=np.int32))
    elif isinstance(new_polygon, MultiPolygon):
        # 如果是 MultiPolygon，取所有多边形进行处理
        for geom in new_polygon.geoms:
            polygons_data.append((label, geom, base_name, np.array(geom.exterior.coords, dtype=np.int32)))
    elif isinstance(new_polygon, GeometryCollection):
        # 提取所有 Polygon 并处理
        for geom in new_polygon.geoms:
            if isinstance(geom, Polygon):
                polygons_data.append((label, geom, base_name, np.array(geom.exterior.coords, dtype=np.int32)))

def plot_histograms(images, num_images, num_figures):
    fig, axes = plt.subplots(num_figures, num_images // num_figures, figsize=(15, 8))
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax = axes[i // (num_images // num_figures), i % (num_images // num_figures)]
        ax.plot(hist, color='blue')
        ax.set_xlim([0, 256])
        ax.set_title(os.path.basename(image_path))
    plt.tight_layout()
    plt.show()


def binarisation_show(images):
    num_images = len(images)
    num_figures = (num_images + 9) // 10  # 计算需要的 figure 数量

    for fig_idx in range(num_figures):
        start_idx = fig_idx * 10
        end_idx = min((fig_idx + 1) * 10, num_images)

        fig, axes = plt.subplots(2, 5, figsize=(15, 8))

        for i, image_idx in enumerate(range(start_idx, end_idx)):
            row = i // 5
            col = i % 5

            image_path = images[image_idx]
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            ### 法1 adaptive+fermeture
            img_binary_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
            cv2.imwrite('adaptive.jpg',img_binary_adaptive)
            # axes[row, col].imshow(img_binary_adaptive, cmap='gray')
            kernel = np.ones((3, 3), np.uint8)
            # 闭运算（先膨胀后腐蚀）膨胀的点为白色,为背景。关心的为黑色,需要去除黑色噪声
            closing = cv2.morphologyEx(img_binary_adaptive, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite('closing.jpg',closing)


            ### 法2 otsu+retrait de zone noir
            # 应用 Otsu's 二值化方法 0: 阈值的初始值（被 Otsu 方法自动调整）。 255: 目标二值化的最大值（255 对应白色） cv2.THRESH_BINARY + cv2.THRESH_OTSU: 指定使用 Otsu 方法自动选择阈值
            _, img_binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite('otsu_image_direct.jpg', img_binary_otsu)
            # 形态学操作去除黑色边缘,扩张白色区域，减少黑色区域，方便后续获取文本页面边缘
            kernel = np.ones((5, 5), np.uint8)
            dilated_image = cv2.dilate(img_binary_otsu, kernel, iterations=1)
            cv2.imwrite('dilate_image.jpg', dilated_image)
            # 创建掩码以去除无用的黑色区域
            # cv2.RETR_EXTERNAL: 只检索最外层轮廓。这个模式会忽略内部轮廓，只保留最外面的轮廓。适用于只需要找出图像中最外面的边界（例如页面的边界）而不关心内部内容时使用。
            # cv2.CHAIN_APPROX_SIMPLE 只保留轮廓的端点，减少了轮廓点的数量。这种方法能减少存储和处理轮廓所需的内存，但可能会丢失一些轮廓细节
            contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(img_binary_otsu)
            for contour in contours:
                # cv2.drawContours: 绘制轮廓，将掩码的轮廓区域填充为白色（255）
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            cv2.imwrite('mask.jpg', mask)
            # 将黑色区域转换为白色背景 掩码为白色（255）的区域保留原图像的像素，其他区域设置为白色（255）
            result_image = np.where(mask == 255, img_binary_otsu, 255)
            cv2.imwrite('result.jpg', result_image)


            axes[row, col].imshow(closing, cmap='gray')
            # axes[row, col].imshow(result_image, cmap='gray')
            axes[row, col].set_title(os.path.basename(image_path))
            axes[row, col].axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'adaptive+fermeture{fig_idx}.png')
if __name__ == '__main__':
    image_folder = '../AnnotedImage'
    labelmap_folder='./Labelmap'
    data_folder='../Data/training'
    background_folder='../background'
    # readImageMask(image_folder)
    # countannotation(image_folder)
    # analyze(image_folder,labelmap_folder)
    # split_dataset(image_folder,labelmap_folder,data_folder,background_folder)

    # images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')][:20]
    # binarisation_show(images)


