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
import glob
import io
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union
import os

import lmdb
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from strhub.data.utils import CharsetAdapter

log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    root = Path(root).absolute()
    log.info(f'dataset root:\t{root}')
    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        dataset = LmdbDataset(ds_root, *args, **kwargs)
        log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        datasets.append(dataset)
    return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(self, root: str, charset: str, max_label_len: int, min_image_dim: int = 0,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 unlabelled: bool = False, transform: Optional[Callable] = None):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.num_samples = self._preprocess_labels(charset, remove_whitespace, normalize_unicode,
                                                   max_label_len, min_image_dim)
    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(self.root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        charset_adapter = CharsetAdapter(charset)
        # not_label_images_dir = 'not_label_images'
        # transformed_images_dir = 'transformed_images'
        # os.makedirs(transformed_images_dir, exist_ok=True)
        # os.makedirs(not_label_images_dir, exist_ok=True)
        count=0
        change_count = 0
        count_eng=0
        count_max_length=0
        count_not_label = 0 
        count_dim_skipped = 0 
        count_skipped_chars= 0
        # count_dev_chars=0
        max = 0
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            print(f'number of samples: {num_samples}')
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                # if normalize_unicode:
                #     label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > 25:
                    # print(f'skipped because max_length exceeded: {label}')
                    count_max_length = count_max_length + 1
                    
                    continue 
                
                if len(label) < max_label_len:
                    if len(label)>max:
                        max = len(label)
                
                # Check if the label contains both lowercase and uppercase English characters
                if any(c.islower() for c in label) or any(c.isupper() for c in label):
                    # print(f'Sample removed due to presence of both lowercase and uppercase English characters {label}')
                    count_eng=count_eng+1
                    continue
                
                change = {'\u201c', '\u201d'}
                if any(char in change for char in label):
                    label = label.replace('\u201c', '"')
                    label = label.replace('\u201d', '"')
                    change_count+=1



                # devanagari_chars = {'ऀ', 'ँ', 'ं', 'ः', 'ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ऌ', 'ऍ', 'ऎ', 'ए', 'ऐ', 'ऑ', 'ऒ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'ळ', 'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'ऺ', 'ऻ', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ', '्', 'ॎ', 'ॏ', 'ॐ', '॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 'य़', 'ॠ', 'ॡ', 'ॢ', 'ॣ', '॰', 'ॱ', 'ॲ', 'ॳ', 'ॴ', 'ॵ', 'ॶ', 'ॷ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॽ', 'ॾ', 'ॿ','०', '१', '२', '३', '४', '५', '६', '७', '८', '९'}

                # if any(char in devanagari_chars for char in label):
                #     count_dev_chars = count_dev_chars + 1
                #     print(f'skipped because found dev chars: {label}')
                #     continue

                special_chars = {'\u007f', '\u00a9', '\u00d7', '\u00bd', '\u2021', '\u2020', '\u2022', '\u0a00','\u2460','\u2461','\u2462', '\u2463', '\u2464', '\u2465', '\u2466', '\u2467', '\u2468', '\u2736', '\u2605', '\ufeff', '→','¦','\u00bc','\u00b7'}


                if any(char in special_chars for char in label):
                    # print(f'words skipped due to special chars: {label}')
                    count_skipped_chars = count_skipped_chars + 1
                    continue


                transformed_label = charset_adapter(label)
                
                
                if label != transformed_label:
                    print("Unsupported characters detected!")
                    print(f'len of original label: {len(label)}')
                    print(f'len of transformed label: {len(transformed_label)}')
                    print("Original label:", label)
                    print("Transformed label:", transformed_label)
                    
                    # Print Unicode code points of original label
                    print("Unicode code points of original label:")
                    for char in label:
                        print(f"/u{ord(char):04x}")
                    
                    # Print Unicode code points of transformed label
                    print("Unicode code points of transformed label:")
                    for char in transformed_label:
                        print(f"/u{ord(char):04x}")
                    
                    # img_key = f'image-{index:09d}'.encode()
                    # buf = txn.get(img_key)
                    # with open(os.path.join(transformed_images_dir, f'image_{index}.jpg'), 'wb') as f:
                    #     f.write(buf)



                    
                    count += 1


                
                label = transformed_label   
                # We filter out samples which don't contain any supported characters
                if not label:
                    # print(f'not a label:')
                    # print(f'len of label not a label: {len(label)}')
                    # for char in label:
                    #     print(f"/u{ord(char):04x}")
                    count_not_label = count_not_label + 1
                                    # Save images with labels not counted
                    # img_key = f'image-{index:09d}'.encode()
                    # buf = txn.get(img_key)
                    # with open(os.path.join(not_label_images_dir, f'image_{index}.jpg'), 'wb') as f:
                    #     f.write(buf)
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        # print('min dim skipped')
                        count_dim_skipped = count_dim_skipped+1 
                        continue
                self.labels.append(label)
                
                self.filtered_index_list.append(index)
        # print(f'transformed labels also contain labels that are not counted these labels that are not counted \n come from labels which are tranformed and length is 0')
        print(f'max of label length from data: {max}')
        print(f'count of labels transformed or characters removed: {count}')
        print(f'count of english containing labels: {count_eng}')
        print(f'max label length exceeded count: {count_max_length}')
        print(f'labels not counted: {count_not_label}')
        print(f'labels dim skipped: {count_dim_skipped}')
        print(f'count of labels not counted with some special chars : {count_skipped_chars}')
        print(f'count of labels changed: {change_count}')
        # print(f'count of labels containing dev chars: {count_dev_chars}')
        print(f'number of samples after transformation: {len(self.labels)}')
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

