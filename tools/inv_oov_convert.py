import os
import lmdb
from tqdm import tqdm
import argparse

def convert_lmdb_to_oov_inv(lmdb_path_train, lmdb_path_test, output_gtfile_inv, output_gtfile_oov, output_folder_inv, output_folder_oov):
# def convert_lmdb_to_oov_inv(lmdb_path_train, lmdb_path_test):
    env_train = lmdb.open(lmdb_path_train, readonly=True)
    txn_train = env_train.begin()
    env_test = lmdb.open(lmdb_path_test, readonly=True)
    txn_test = env_test.begin()


    cursor_train = txn_train.cursor()
    cursor_test = txn_test.cursor()

    train_labels = [value.decode().strip() for key, value in cursor_train if key.startswith(b'label-')]
    test_labels = [value.decode().strip() for key, value in cursor_test if key.startswith(b'label-')]
    print(f'length of train dataset: {len(train_labels)}')
    print(f'length of test dataset : {len(test_labels)}')
    # test_labels = [value.decode().strip() for key, value in txn_test.cursor() if key.startswith(b'label-')]
    inv_list = []
    oov_list = []
    for key, value in tqdm(cursor_test, desc='extracting labels and indices'):
        if key.startswith(b'label-'):
            test_value = value.decode().strip()
            test_key = key.decode().split('-')[1]
            if test_value in train_labels:
                inv_list.append(test_key)
            else:
                oov_list.append(test_key)



    # inv_list = [label for label in test_labels if label in train_labels]
    # oov_list = [label for label in test_labels if label not in train_labels]

    print(f'length of inv_list = {len(inv_list)}')
    print(f'length of oov list = {len(oov_list)}')
    output_folder = ''
    with open(output_gtfile_inv, 'w') as f_inv, open(output_gtfile_oov, 'w') as f_oov:

        for key, value in tqdm(cursor_test, desc='converting inv oov'):
            if key.startswith(b'image-'):
                test_index = key.decode().split('-')[1]
                # print(f'test index: {test_index}')
                image_filename = test_index+'.jpg'  # Assuming images were stored as JPEG
                if test_index in inv_list:
                    output_folder = output_folder_inv
                    # print('inv')
                elif test_index in oov_list:
                    output_folder = output_folder_oov
                    # print('oov')
                image_path = os.path.join(output_folder, image_filename)
                with open(image_path, 'wb') as image_file:
                    image_file.write(value)
            elif key.startswith(b'label-'):
                test_index = key.decode().split('-')[1]
                label = value.decode().strip()
                image_filename = test_index+'.jpg'
                if test_index in inv_list:
                    output_folder = output_folder_inv
                    image_path = os.path.join(output_folder, image_filename)
                    f_inv.write(f"{image_path} {label}\n")
                elif test_index in oov_list:
                    output_folder = output_folder_oov
                    image_path = os.path.join(output_folder, image_filename)
                    f_oov.write(f"{image_path} {label}\n")

    env_train.close()
    env_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="take paths")
    parser.add_argument("--lmdb_path_train", type=str, help="lmdb path train ")
    parser.add_argument("--lmdb_path_test", type=str, help='lmdb path test')
    parser.add_argument("--output_gtfile_inv", type=str, help="output gtfile inv" )
    parser.add_argument("--output_gtfile_oov", type=str, help="output gtfile inv" )
    parser.add_argument("--output_folder_inv", type=str, help="output folder inv" )
    parser.add_argument("--output_folder_oov", type=str, help="output folder oov" )

    args = parser.parse_args()

    lmdb_path_train = args.lmdb_path_train
    lmdb_path_test = args.lmdb_path_test
    output_gtfile_inv = args.output_gtfile_inv
    output_gtfile_oov = args.output_gtfile_oov
    output_folder_inv = args.output_folder_inv
    output_folder_oov = args.output_folder_oov

    convert_lmdb_to_oov_inv(lmdb_path_train, lmdb_path_test, output_gtfile_inv, output_gtfile_oov, output_folder_inv, output_folder_oov)
    # convert_lmdb_to_oov_inv(lmdb_path_train, lmdb_path_test)
