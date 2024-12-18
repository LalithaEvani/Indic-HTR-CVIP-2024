#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string
import sys
from dataclasses import dataclass
from typing import List
import lmdb

import torch
import os

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from nltk import edit_distance


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float
    wer : float
    cer : float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |      WER |      CER |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|---------:|---------:|'.format('----', w=w), file=file)
    # c = Result('Combined', 0, 0, 0, 0, 0, 0)
    for res in results:
        # c.num_samples += res.num_samples
        # c.accuracy += res.num_samples * res.accuracy
        # c.ned += res.num_samples * res.ned
        # c.confidence += res.num_samples * res.confidence
        # c.label_length += res.num_samples * res.label_length
        # c.wer += res.num_samples * res.wer
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} | {res.wer:>8.2f} | {res.cer:>8.2f}', file=file)
    # c.accuracy /= c.num_samples
    # c.ned /= c.num_samples
    # c.confidence /= c.num_samples
    # c.label_length /= c.num_samples
    # c.wer /= c.num_samples
    # print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|----------|'.format('----', w=w), file=file)
    # print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
    #       f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} | {res.wer:>8.2f} | ', file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
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

    # test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    # if args.new:
    #     test_set += SceneTextDataModule.TEST_NEW
    # test_set = SceneTextDataModule.TEST_NEW
    test_set = sorted(set(['old','new']))

    test_output_path = 'test_output'
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)
        print(f'test_output folder created successfully')
    test_data_path = os.path.join(test_output_path,'test_data')
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)
        print(f'test_data folder created successfully')

    for name in test_set:
        image_data_path = os.path.join(test_data_path, name)
        if not os.path.exists(image_data_path):
            os.makedirs(image_data_path)
            print(f'{name} folder created successfully')
        test_path = os.path.join(args.data_root, 'test', name)
        env_test= lmdb.open(test_path, readonly=True)
        txn_test = env_test.begin()
        cursor_test = txn_test.cursor()
        for key, value in tqdm(cursor_test, desc='saving images'):
            if key.startswith(b'image-'):
                # print(f'image key in test code {key}')
                test_index = key.decode().split('-')[1]
                image_file_name = f'img_{test_index}.png'
                image_path = os.path.join(image_data_path, image_file_name)
                with open(image_path, 'wb') as image_file:
                    image_file.write(value)

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        with open(f'{test_output_path}/{name}_output_text_file.txt', 'w') as f_txt:
            correct = 0
            total = 0
            ned = 0
            confidence = 0
            label_length = 0
            cer = 0
            f_txt.write(f"|{'IMAGE NAME':>10} | {'CONFIDENCE':>10} | {'CORRECT':>7} | {'CER':>8} | {'GROUND TRUTH - PREDICTED'}\n")
            for imgs, imgs_transformed, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
                # f_txt.write(f'Number of samples: {len(labels)} \n')
                # f_txt.write('****_______________________________________________________________****')
                # # print('inside dataloader ')
                for index, img_transformed, label in zip(imgs,imgs_transformed, labels):
                    # print(f'shape of image: {img_transformed.shape}')
                    img_transformed = torch.unsqueeze(img_transformed, dim=0).to(args.device)
                    # print(f'after unsqueeze: {img_transformed.shape}')
                    probs = model(img_transformed).softmax(-1)
                    # print(f'output shape: {model.tokenizer.decode(probs)}')
                    pred, prob = model.tokenizer.decode(probs)
                    # print(f'type of pred: {type(pred[0])}')
                    # print(f'type of prob: {type(prob[0])}')
                    confidence_single = prob[0].prod().item()
                    confidence += confidence_single
                    pred = model.charset_adapter(pred[0])
                # Follow ICDAR 2019 definition of N.E.D.
                    ned += edit_distance(pred, label) / max(len(pred), len(label))
                    cer_one = edit_distance(pred,label)/len(label)
                    cer += cer_one
                    correct_bool = False
                    if pred == label:
                        correct += 1
                        correct_bool = True


                    total += 1
                    label_length += len(pred)
                    f_txt.write(f'|{index:>10} | {confidence_single:>10.2f} | {correct_bool:>7} | {cer_one:>8.2f}| {label} - {pred} \n')
                    # f_txt.write(f'image path: {image_path} \n ground truth: {label} \n predicted: {pred} \n confidence: {confidence_single} \n correct: {correct_bool} \n cer: {cer_one}')
                    # f_txt.write('_______________________________________________________________')

            f_txt.write('********************************************************************\n')
            f_txt.write('********************************************************************\n')
            accuracy = 100 * correct / total
            cer = 100 * cer / total
            mean_ned = 100 * (1 - ned / total)
            mean_conf = 100 * confidence / total
            mean_label_length = label_length / total
            wer = 100 - accuracy
            f_txt.write(f'WER : {wer} \n CER : {cer} \n NED : {mean_ned} \n Confidence : {mean_conf} \n Label length: {mean_label_length} \n Accuracy: {accuracy}')
            f_txt.write('\n~~~~~~~~~~~~~~~~END~~~~~~~~~~~~~~~')
            results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length, wer, cer)

    result_groups = {
        'Benchmark (Subset)': ['old','new']
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()