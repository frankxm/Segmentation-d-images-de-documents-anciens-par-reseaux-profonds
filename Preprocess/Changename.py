import os
import pandas as pd
import re
import json
def changename(image_folder,excel_path):

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            name=filename.split('.')[0]
            json_path = os.path.join(image_folder, f"{name}.json")
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            df = pd.read_excel(excel_path,engine='openpyxl')
            row = df[df['Nom'] == name]
            if not row.empty:
                identifiant = row.iloc[0]['Identifiant']
                elements = re.split(r'[\s,]+', identifiant)
                newfilename=f'img_{elements[0]}_{elements[1]}_{elements[2]}_{elements[3]}_{elements[6]}.jpg'
                newjsonname=f'img_{elements[0]}_{elements[1]}_{elements[2]}_{elements[3]}_{elements[6]}.json'
                new_path = os.path.join(image_folder, newfilename)
                annotations['imagePath']=newfilename
                newjson_path=os.path.join(image_folder,newjsonname)
                os.rename(image_path, new_path)
                with open(newjson_path, 'w') as file:
                    json.dump(annotations, file, indent=4)


if __name__ == '__main__':
    image_folder='../AnnotedImage'
    excel_path='./liste des images.xlsx'
    changename(image_folder,excel_path)