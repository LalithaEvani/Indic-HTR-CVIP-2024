import numpy as np
from tqdm import tqdm
from nltk import edit_distance

distance_matrix_file_path = 'lex_edit_distance.npy'
matrix = np.load(distance_matrix_file_path, allow_pickle=True)

lexicon_list_file_path = 'lexicon_list_hindi.npy'
lexicon_list = np.load(lexicon_list_file_path, allow_pickle=True)

pred_labels_file_path = 'pred_label_list_hindi.npy'
pred_labels = np.load(pred_labels_file_path, allow_pickle=True)

ground_truth_file_path= 'ground_truth_hindi.npy'
ground_truth = np.load(ground_truth_file_path, allow_pickle=True)
# print(f'matrix (5,5) : {matrix[0:5, 0:5]}')
recall = 4
print(f'shape of matrix {matrix.shape}')
for k in range(1, recall+1):
# recall_1 = []
# for i in range(len(matrix)):
#     if recall == 1:
#         min_index = np.argmin(matrix[i])
#         final_edit_distance = matrix[i][min_index]
#         recall_1.append(lexicon_list[min_index])
#         # print(f'for word : {lexicon_list[min_index]} final edit distance: {final_edit_distance}')

#         # print(f'recall 1 : {recall_1}')
#         correct = 0
#         total = 0
#         cer = 0
#         for pred, gt in tqdm(zip(recall_1, ground_truth), desc='calculating the cer and wer'):
#             cer += edit_distance(pred, gt)/len(gt)
#             if pred == gt:
#                 correct +=1
#             total += 1

#         cer_final = cer/total
#         wer = 1 - correct/total

#         print(f'cer: {cer_final*100} \n wer: {wer*100} ')

#     if recall == 2:
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

