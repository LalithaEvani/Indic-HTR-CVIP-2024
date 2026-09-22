import json
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def save_base64_image(base64_str, filename):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image.save('visual_crnn_compare/images/'+filename)

# Assuming the JSON data is stored in a file called 'data.json'
with open('telugu.json', 'r') as file:
    data = json.load(file)

output_data = []

# Extract the labels
for index, item in enumerate(tqdm(data, desc='converting data')):
    image_filename = f'image_{index + 1}.png'
    save_base64_image(item['image'], image_filename)

    output_data.append({
        'image_name': image_filename,
        'ground_truth': item['gt'],
        'crnn_pred_label': item['ocr']
    })

with open('visual_crnn_compare/telugu.json', 'w') as file:
    json.dump(output_data, file, ensure_ascii = False, indent=4)

print("saved")
