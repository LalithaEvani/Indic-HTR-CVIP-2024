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
import glob
import io
import logging
from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import lmdb
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from indichtr.data.utils import CharsetAdapter

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

    # Curly quotes normalized to straight quotes before charset filtering.
    _QUOTE_NORMALIZE = {'“': '"', '”': '"'}
    # Control/formatting/decorative characters observed in the raw scanned data
    # that no charset should accept -- dropped before charset filtering.
    _SPECIAL_CHARS = {'\u007f', '©', '×', '½', '‡', '†', '•', '਀',
                      '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧',
                      '⑨', '✶', '★', '﻿', '→', '¦', '¼', '·'}

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
        count_transformed = 0
        count_quote_normalized = 0
        count_mixed_case = 0
        count_max_length = 0
        count_not_label = 0
        count_dim_skipped = 0
        count_special_chars = 0
        max_len_seen = 0
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                if remove_whitespace:
                    label = ''.join(label.split())
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    count_max_length += 1
                    continue
                max_len_seen = max(max_len_seen, len(label))

                # Drop labels containing English characters -- not part of any Indic charset.
                if any(c.islower() for c in label) or any(c.isupper() for c in label):
                    count_mixed_case += 1
                    continue

                if any(char in self._QUOTE_NORMALIZE for char in label):
                    for src, dst in self._QUOTE_NORMALIZE.items():
                        label = label.replace(src, dst)
                    count_quote_normalized += 1

                if any(char in self._SPECIAL_CHARS for char in label):
                    count_special_chars += 1
                    continue

                transformed_label = charset_adapter(label)
                if label != transformed_label:
                    count_transformed += 1
                label = transformed_label
                # We filter out samples which don't contain any supported characters
                if not label:
                    count_not_label += 1
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        count_dim_skipped += 1
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        log.info(f'{self.root}: max label length {max_len_seen}, '
                f'{count_transformed} labels had unsupported characters stripped, '
                f'{count_mixed_case} dropped for mixed-case English, '
                f'{count_max_length} dropped for exceeding max_label_len, '
                f'{count_not_label} dropped as empty after filtering, '
                f'{count_dim_skipped} dropped for image size, '
                f'{count_special_chars} dropped for special characters, '
                f'{count_quote_normalized} had curly quotes normalized')
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
