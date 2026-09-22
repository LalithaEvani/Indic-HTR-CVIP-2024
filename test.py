#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Modifications Copyright 2024 Evani Lalitha, CVIT, IIIT Hyderabad
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
import sys
from dataclasses import dataclass
from typing import List

import torch

from tqdm import tqdm
from nltk import edit_distance

from indichtr.data.module import IndicHTRDataModule
from indichtr.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    cer: float
    confidence: float
    label_length: float
    wer : float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'))
    print('| {:<{w}} | # samples | Accuracy |     CER | Confidence | Label Length |      WER |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|---------:|'.format('----', w=w), file=file)
    for res in results:
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.cer:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} | {res.wer:>8.2f} |', file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Path to a trained model checkpoint (.ckpt)")
    parser.add_argument('--data_root', required=True, help='Path to <lang>/datasets (see Datasets.md)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test_set', nargs='+', default=['IIIT-INDIC-HW-WORDS'],
                        help="Subdirectory name(s) under <data_root>/test/ to evaluate on "
                             "(e.g. the paper's Table 2 uses the full IIIT-INDIC-HW-WORDS test set; "
                             "pass 'inv_lmdb oov_lmdb' for the in-vocab/out-of-vocab split analysis)")
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown) 
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = IndicHTRDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)

    test_set = sorted(set(args.test_set))

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        cer = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res_dict = model.test_step((imgs.to(model.device), labels), -1)
            res = res_dict['output']
            #print('res: ', res_dict)
            total += res.num_samples
            correct += res.correct
            # Per Section 4.4/Eq. 8, CER = mean over samples of edit_distance(pred, gt) / len(gt).
            cer += sum(edit_distance(pred, gt) / len(gt) for pred, gt in zip(res.pred_labels, labels))
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_cer = 100 * cer / total
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        wer = 100 - accuracy
        results[name] = Result(name, total, accuracy, mean_cer, mean_conf, mean_label_length, wer)

    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            print_results_table([results[s] for s in test_set], out)


if __name__ == '__main__':
    main()

