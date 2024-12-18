import argparse
import string
import sys
from dataclasses import dataclass
from typing import List
import lmdb
import yaml


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
    for res in results:
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} | {res.wer:>8.2f} | {res.cer:>8.2f}', file=file)

@torch.inference_mode()
def main():
    """
    Example command:
    python test.py 
    outputs/parseq/2024-08-29_21-09-15/checkpoints/'epoch=95-step=56150-val_accuracy=86.9342-val_NED=96.5613-val_loss=0.6817.ckpt' 
    --test_folder=punjabi --change_charset_test=True --charset_test=configs/charset/punjabi.yaml"""
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
    parser.add_argument('--test_folder', help='name of the test folder' )
    parser.add_argument('--charset_test', help='path to the YAML config charset file', default=None)
    parser.add_argument('--change_charset_test', help='want a different charset from that mentioned', default=False)

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')


    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    if args.change_charset_test:
        with open(args.charset_test, 'r') as file:
            config = yaml.safe_load(file)
        charset_test = config.get('model', {}).get('charset_test', "")
        datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                            charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)
        print(f'charset test used: {charset_test}')
    else:
        datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                         hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)
        print(f'charset test used: {hp.charset_test}')
    test_set = sorted(set([args.test_folder]))

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        crr = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res_dict = model.test_step((imgs.to(model.device), labels), -1)
            res = res_dict['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            crr += res.crr
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        crr_total = 100 * crr / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        wer = 100 - accuracy
        cer = crr_total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length, wer, cer)

    
    result_groups = {
        'Benchmark (Subset)': [args.test_folder]
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
