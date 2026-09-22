import glob
import lmdb
from pathlib import Path
from tqdm import tqdm
import argparse
from indichtr.models.utils import load_from_checkpoint, parse_model_args
from nltk import edit_distance
from indichtr.data.module import IndicHTRDataModule
import torch
import numpy as np
import multiprocessing


def create_lexicon(root_path, lexicon_list):
    # Mirrors indichtr.data.dataset.build_tree_dataset: the LMDB(s) may be
    # nested under root_path rather than root_path itself being one.
    word_num = 0
    for mdb in glob.glob(str(Path(root_path) / '**/data.mdb'), recursive=True):
        lmdb_dir = str(Path(mdb).parent)
        env_data = lmdb.open(lmdb_dir, readonly=True)
        txn_data = env_data.begin()
        cursor_data = txn_data.cursor()
        for key, value in tqdm(cursor_data, desc=f'gathering the lexicon from {lmdb_dir}'):
            if key.startswith(b'label-'):
                label_value = value.decode().strip()
                word_num += 1
                if label_value not in lexicon_list:
                    lexicon_list.append(label_value)
    print(f'number of words in dataset: {word_num}')
    return lexicon_list

def compute_distances(process_number, words1, words2, start, end, result):
    print(f'process number : {process_number} start index: {start}')
    partial_result = []
    for i in tqdm(range(start, end), desc=f'process {process_number} completing start {start} and end {end}'):
        distances = [edit_distance(words1[i], word2) for word2 in words2]
        partial_result.append(distances)
    result[start:end] = partial_result
    print(f'process number : {process_number} end index: {end} completed')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Path to a trained model checkpoint (.ckpt)")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_root', required=True, help='Path to <lang>/datasets (see Datasets.md)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--test_set', nargs='+', default=['IIIT-INDIC-HW-WORDS'],
                        help="Subdirectory name(s) under <data_root>/test/ to evaluate on")
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = IndicHTRDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)

    # Lexicon = every ground-truth label across train + val + test, per Section 4.5.
    lexicon_list = []
    lexicon_list = create_lexicon(f'{args.data_root}/train', lexicon_list)
    lexicon_list = create_lexicon(f'{args.data_root}/val', lexicon_list)
    lexicon_list = create_lexicon(f'{args.data_root}/test', lexicon_list)
    print(f'length of lexicon list {len(lexicon_list)}')
    test_set = sorted(set(args.test_set))

    ground_truth = []
    pred_labels = []
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res_dict = model.test_step((imgs.to(model.device), labels), -1)
            res = res_dict['output']
            ground_truth.extend(labels)
            pred_labels.extend(res.pred_labels)


    words1 = pred_labels
    words2 = lexicon_list
    num_words1 = len(words1)

    # Shared memory array to store results
    manager = multiprocessing.Manager()
    result = manager.list([None] * num_words1)

    num_processes = min(multiprocessing.cpu_count(), 15)
    chunk_size = (num_words1 + num_processes - 1) // num_processes  # Divide work evenly

    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_words1)
        process = multiprocessing.Process(target=compute_distances, args=(i,words1, words2, start, end, result))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    # matrix[i][j] is the edit distance between prediction i and lexicon word j.
    matrix = list(result)
    print(f'distance matrix computed shape {len(matrix)}')

    recall = 5
    for k in range(1, recall + 1):
        cer = 0
        correct = 0
        # For each prediction, take the k lexicon words with the least edit
        # distance and keep whichever of those k is closest to the ground
        # truth -- this is Recall@k from Section 4.5/Table 3.
        for i in tqdm(range(len(matrix)), desc='pred words'):
            sorted_indices = [idx for idx, _ in sorted(enumerate(matrix[i]), key=lambda x: x[1])]
            min_ed_correct_gt = np.inf
            final_pred = None
            for j in range(k):
                corrected_word = lexicon_list[sorted_indices[j]]
                edit_distance_correct_gt = edit_distance(ground_truth[i], corrected_word)
                if edit_distance_correct_gt < min_ed_correct_gt:
                    min_ed_correct_gt = edit_distance_correct_gt
                    final_pred = corrected_word

            cer += min_ed_correct_gt / len(ground_truth[i])
            if final_pred == ground_truth[i]:
                correct += 1

        final_cer = cer / len(pred_labels)
        wrr = correct / len(pred_labels)
        wer = 100 - wrr * 100
        print(f' final cer for recall {k} is : {final_cer*100}')
        print(f' wer for recall {k} is : {wer}')


if __name__ == '__main__':
    main()


