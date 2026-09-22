import glob
import lmdb
from pathlib import Path
from tqdm import tqdm
import argparse
from indichtr.models.utils import load_from_checkpoint, parse_model_args
from nltk import edit_distance
from indichtr.data.module import SceneTextDataModule
import torch
import numpy as np
import multiprocessing
from nltk.metrics import edit_distance
import nltk

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
        distances = []
        for word2 in words2:
            # print(f'words1 : {words1[i]} words2: {word2}')
            # print(f'edit distance {edit_distance(words1[i], word2)}')
            distances.append(edit_distance(words1[i], word2))
        partial_result.append(distances)
    result[start:end] = partial_result
    print(f'process number : {process_number} end index: {end} completed')
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_root', required=True, help='Path to <lang>/datasets (see Datasets.md)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--test_set', nargs='+', default=['IIIT-INDIC-HW-WORDS'],
                        help="Subdirectory name(s) under <data_root>/test/ to evaluate on")
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
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
    num_words2 = len(words2)

    # Shared memory array to store results
    manager = multiprocessing.Manager()
    result = manager.list([None] * num_words1)

    # num_processes = multiprocessing.cpu_count()
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

    distances_matrix = list(result)
    print(f'distance matrix computed shape {len(distances_matrix)}')
    # print(f'distance matrix 5,5: {np.array(distances_matrix)[0:5, 0:5]}')
    # np.save('lex_edit_distance_hindi.npy', distances_matrix)
    # print('saved the matrix')



    # distance_matrix_file_path = 'lex_edit_distance.npy'
    # matrix = np.load(distance_matrix_file_path, allow_pickle=True)
    matrix = distances_matrix

    # lexicon_list_file_path = 'lexicon_list_hindi.npy'
    # lexicon_list = np.load(lexicon_list_file_path, allow_pickle=True)

    # pred_labels_file_path = 'pred_label_list_hindi.npy'
    # pred_labels = np.load(pred_labels_file_path, allow_pickle=True)

    # ground_truth_file_path= 'ground_truth_hindi.npy'
    # ground_truth = np.load(ground_truth_file_path, allow_pickle=True)
    # print(f'matrix (5,5) : {matrix[0:5, 0:5]}')
    recall = 5
    print(f'shape of matrix [pred,lexicon]{np.array(matrix).shape}')
    for k in range(1, recall+1):    
        cer = 0
        correct = 0
        for i in tqdm(range(len(matrix)), desc='pred words'):#for every row or every predicted label 
        # for i in range(10):
            row_index_list = list(enumerate(matrix[i]))#create a list containing index and edit distance
            # print(f'for i {i}, row index list {row_index_list[0:5]}')  
            sorted_row_index_list = sorted(row_index_list, key=lambda x: x[1])#sort the edit distance and preserve the index 
            # print(f'for i {i}, sorted row index list {sorted_row_index_list[0:5]}')  
            sorted_row_indices = [x[0] for x in sorted_row_index_list]#obtain only the indices after sorting so that first indices contain low edit distance values 
            # print(f'for i {i}, sorted row indices {sorted_row_indices[0:5]}')  
            edit_distance_correct_gt = None
            min_ed_correct_gt = np.inf
            final_pred = None
            for j in range(k):#for the recall number 
                corrected_word = lexicon_list[sorted_row_indices[j]]#get the least edit distance words one by one 

                # print(f'for j {j}, corrected word {corrected_word} lexicon list {lexicon_list[sorted_row_indices[0:5]]}')  

                
                #if recall is 2 then it is done for the first two words in the index list which have least edit distance 
                edit_distance_correct_gt = edit_distance(ground_truth[i], corrected_word)#calculate the edit distance 
                # print(f'for pred word {pred_labels[i]} for recall {recall} for number recall {j+1} the corrected word {corrected_word} edit distance {edit_distance_correct_gt}')
                #since we need least edit distance even in the case of recall for final cer calculation 
                if edit_distance_correct_gt < min_ed_correct_gt:#so retain only one value
                    min_ed_correct_gt = edit_distance_correct_gt
                    final_pred = corrected_word
                
            cer += min_ed_correct_gt/len(ground_truth[i])#after all this the final edit distance for that particular predicted label is in min_ed_correct_gt
            if final_pred == ground_truth[i]:
                correct+=1

        final_cer = cer/ len(pred_labels)
        wrr = correct / len(pred_labels)
        wer = 100 - wrr*100
        print(f' final cer for recall {k} is : {final_cer*100}')
        print(f' wer for recall {k} is : {wer}')
        # print(f' unique words in lexicon list: {len(set(lexicon_list))}')



        # distances_matrix is now a matrix where distances_matrix[i][j] is the edit distance between words1[i] and words2[j]


if __name__ == '__main__':
    main()


