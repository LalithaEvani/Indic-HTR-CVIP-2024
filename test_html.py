import argparse
import string
import sys
import os
import base64
from typing import List
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from io import BytesIO
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown) 
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)
    test_set = sorted(set(['hindi_handwritten']))

    results = []
    max_width = max(map(len, test_set))
    
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        for index, (imgs, labels) in enumerate(tqdm(dataloader, desc=f'{name:>{max_width}}')):
            with torch.no_grad():
                p = model(imgs.to(model.device)).softmax(-1)
                pred, _ = model.tokenizer.decode(p) # Iterate over predicted labels and probability distributions
                res = model.test_step((imgs.to(model.device), labels), -1)['output']
                confidence = res.confidence

            # Convert tensor to PIL Image
            img_pil = transforms.ToPILImage()(imgs[0].cpu()).convert("RGB")

            # Convert PIL Image to base64 string
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Append the result to the list
            results.append((img_str, labels[0], pred[0], confidence))

    # Generate HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>

    <h2>Results</h2>
    <table>
      <tr>
        <th>Image</th>
        <th>Actual Label</th>
        <th>Predicted Label</th>
        <th>Confidence</th>
      </tr>
    """

    for result in results:
        html_content += f"""
        <tr>
            <td><img src="data:image/png;base64,{result[0]}" alt="Image" width="100"></td>
            <td>{result[1]}</td>
            <td>{result[2]}</td>
            <td>{result[3]}</td>
        </tr>
        """

    html_content += """
    </table>

    </body>
    </html>
    """

    # Save HTML content to file
    with open('results_796_epoch_val_88.6213_test_88.54.html', 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    main()

