import json
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os
import argparse
import torch
import sys

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

"""
json files from ilocr saved into visual_crnn_parseq/crnn_json_files/
the structure is 
    visual_crnn_parseq/
        crnn_json_files/
            langxxx.json
            lang2xxx.json 
            .
            .
            .
        outputs/
            langxxx/
                langxxx.json
                langxxx.txt
                images/
                    image_1.png
                    image_2.png
                    .
                    .
                    .
            lang2xxx/
                lang2xxx.json
                lang2xxx.txt
                images/
                    image_1.png
                    image_2.png
                    .
                    .
                    .
give the language input and the checkpoint and ensure the language json is in the crnn_json_files folder and the work is done
"""
def save_base64_image(base64_str, file_path):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image.save(file_path)

@torch.inference_mode()
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--language', help='language currently working on')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    #STEP 1: saving images and json data
    root = 'visual_crnn_parseq'
    crnn_json_files_folder_path = os.path.join(root, 'crnn_json_files')
    outputs_path = os.path.join(root, 'outputs')

    lang_folder_path = os.path.join(outputs_path, args.language)
    # if os.path.exists(lang_folder_path)
    os.makedirs(lang_folder_path, exist_ok=True)
    image_folder_path = os.path.join(lang_folder_path, 'images')
    os.makedirs(image_folder_path)
    output_json_file_path = os.path.join(lang_folder_path, f'{args.language}.json')
    output_text_file_path = os.path.join(lang_folder_path, f'{args.language}.txt')


    crnn_json_file_path = os.path.join(crnn_json_files_folder_path, f'{args.language}.json')        
    with open(crnn_json_file_path, 'r') as file:
        data = json.load(file)

    output_data = []

    # Extract the labels
    for index, item in enumerate(tqdm(data, desc=f'saving images and writing data')):
        image_filename = f'image_{index + 1}.png'
        image_file_path = os.path.join(image_folder_path, image_filename)
        save_base64_image(item['image'], image_file_path)

        output_data.append({
            'image_name': image_filename,
            'ground_truth': item['gt'],
            'crnn_pred_label': item['ocr']
        })

    with open(output_json_file_path, 'w') as file:
        json.dump(output_data, file, ensure_ascii = False, indent=4)

    print("saved the json data and images successfully")

    #STEP 2: loading the checkpoint and getting the parseq predicted labels 
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    if image_folder_path:
        # List all files in the image folder
        image_files = [[os.path.join(image_folder_path, f), f] for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]
    else:
        image_files = []
        print("No images in the image folder")
        return
    
    with open(output_json_file_path, 'r') as file:
        data = json.load(file)
    
    for fname, img_name in tqdm(image_files, desc='predicting labels...'):
        image = Image.open(fname).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)

        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        for j in range(len(data)):
            if data[j]['image_name'] == img_name:
                data[j]['parseq_pred_label']= pred[0]

    with open(output_json_file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print('Completed updating the json file with parseq results')

    #STEP 3: comparing crnn and parseq and saving the results in text file
    with open(output_text_file_path, 'w') as file:
        sys.stdout = file
        
        with open(output_json_file_path, 'r') as file_json:
            data = json.load(file_json)
        
        #lists required
        c_r_p_r = []
        c_w_p_r = []
        c_w_p_w = []
        for unit in data:
            if unit['ground_truth'] == unit['crnn_pred_label'] and unit['ground_truth'] == unit['parseq_pred_label']:
                c_r_p_r.append(unit['image_name'])
            elif unit['ground_truth'] != unit['crnn_pred_label'] and unit['ground_truth'] == unit['parseq_pred_label']:
                c_w_p_r.append(unit['image_name'])
            elif unit['ground_truth'] != unit['crnn_pred_label'] and unit['ground_truth'] != unit['parseq_pred_label']:
                c_w_p_w.append(unit['image_name'])

        print(f'crnn right parseq right: ')
        for i in range(10):
            print(c_r_p_r[i])


        print(f'crnn wrong parseq right: ')
        for i in range(10):
            print(c_w_p_r[i])


        print(f'crnn wrong parseq wrong: ')
        for i in range(10):
            print(c_w_p_w[i])
        
        sys.stdout = sys.__stdout__
    
    print('Saved comparision sucessfully')
    print('COMPLETED')

if __name__ == '__main__':
    main()

    
    





