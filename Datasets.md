We use **IIIT-INDIC-HW-WORDS**:

> Gongidi, S., Jawahar, C.: iiit-indic-hw-words: A dataset for indic handwritten text recognition. In: Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5-10, 2021, Proceedings, Part IV 16. pp. 444–459. Springer (2021)

Word-level handwritten images for ten languages, converted to LMDB (`create_dataset_lmdb_lalitha.py` in [`tools/`](tools)) with keys `image-%09d` / `label-%09d` (see `strhub/data/dataset.py:LmdbDataset`).

## Per-language statistics (Table 1 of the paper)

| Script     | #Writers | #Word Instances | Lexicon Size | #Train | #Val   | #Test  |
|:----------:|---------:|-----------------:|--------------:|-------:|-------:|-------:|
| Devanagari |       12 |              95K |        11,030 | 69,853 | 12,708 | 12,869 |
| Telugu     |       11 |             120K |        12,945 | 80,637 | 19,980 | 17,898 |
| Bengali    |       24 |             113K |        11,295 | 82,554 | 12,947 | 17,574 |
| Gujarati   |       17 |             116K |        10,963 | 82,563 | 17,643 | 16,490 |
| Gurumukhi  |       22 |             112K |        11,093 | 81,042 | 13,627 | 17,947 |
| Kannada    |       11 |             103K |        11,766 | 73,517 | 13,752 | 15,730 |
| Odia       |       10 |             101K |        13,314 | 73,400 | 11,217 | 16,850 |
| Malayalam  |       27 |             116K |        13,401 | 85,270 | 11,878 | 19,635 |
| Tamil      |       16 |             103K |        13,292 | 75,736 | 11,597 | 16,184 |
| Urdu       |        8 |             100K |        11,936 | 71,207 | 13,906 | 15,517 |

## Expected filesystem structure

Each language's data is a **separate root directory** (pass it as `data.root_dir` to `train.py`/`--data_root` to `test.py`/`postocr.py`); there is no single shared `data/` root across languages:

```
<lang>/datasets/
├── train
│   └── IIIT-INDIC-HW-WORDS
│       ├── data.mdb
│       └── lock.mdb
├── val
│   └── IIIT-INDIC-HW-WORDS
│       ├── data.mdb
│       └── lock.mdb
└── test
    └── IIIT-INDIC-HW-WORDS
        ├── data.mdb
        └── lock.mdb
```

`strhub/data/dataset.py:build_tree_dataset` finds LMDBs by recursively globbing for `data.mdb` under the given root, so the `IIIT-INDIC-HW-WORDS` subdirectory name isn't load-bearing — what matters is `data.train_dir` (passed to `train.py`) pointing at the right subdirectory under `<root>/train`.

## Building an LMDB from raw images

`tools/create_dataset_lmdb_lalitha.py` takes a ground-truth file (`<absolute image path><space><label>` per line, UTF-8) and writes an LMDB:

```bash
python tools/create_dataset_lmdb_lalitha.py --inputPath <unused> --gtFile gt.txt --outputPath <lang>/datasets/train/IIIT-INDIC-HW-WORDS
```
