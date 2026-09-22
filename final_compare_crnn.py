from tqdm import tqdm
import json

with open('visual_crnn_compare/telugu.json', 'r') as file:
    telugu_data = json.load(file)


#lists required
c_r_p_r = []
c_w_p_r = []
c_w_p_w = []
for unit in telugu_data:
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

