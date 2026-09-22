import argparse
import os
from tqdm import tqdm 

import torch

from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

import json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--image_folder', help='Folder containing images to read')
    # parser.add_argument('--output-file', help='File to save image names, paths, and predicted labels')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    if args.image_folder:
        # List all files in the image folder
        image_files = [[os.path.join(args.image_folder, f), f] for f in os.listdir(args.image_folder) if os.path.isfile(os.path.join(args.image_folder, f))]
    else:
        image_files = []

    with open('visual_crnn_compare/telugu.json', 'r') as file:
        data = json.load(file)
    
    for i in range(5):
        print(f'image file names: {image_files[i][1]}')
    print(f'len of data: {len(data)}')
    i = 0
    for fname, img_name in tqdm(image_files, desc='predicting labels...'):
        # Check if the file is an image
        if any(fname.lower().endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
            # Load image and prepare for input
            image = Image.open(fname).convert('RGB')
            image = img_transform(image).unsqueeze(0).to(args.device)

            p = model(image).softmax(-1)
            pred, p = model.tokenizer.decode(p)
            # results.append((fname, pred[0]))
            for j in range(len(data)):
                if data[j]['image_name'] == img_name:
                    data[j]['parseq_pred_label']= pred[0]
                    if i < 5:
                        print(f"image_name: {img_name}  actual name : {data[j]['image_name']}")


            # elif i <5 an d data[i]['image_name'] == img_name:
            #     print(f"image_name: {img_name}  actual name : {data[i]['image_name']}")
            if i == 0:
                print(f'type of pred : {type(pred)}')
            i += 1
    
    with open('visual_crnn_compare/telugu.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print('updated sucessfully')


if __name__ == '__main__':
    main()
