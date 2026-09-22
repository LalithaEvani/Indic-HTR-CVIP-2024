# Indic-HTR-CVIP-2024

Code for **"Enhancing Accuracy in Indic Handwritten Text Recognition"** (Evani Lalitha, Ajoy Mondal, C. V. Jawahar — CVIT, IIIT Hyderabad), accepted at [CVIP 2024](https://link.springer.com/chapter/10.1007/978-3-031-93688-3_17).

We fine-tune [PARSeq](https://github.com/baudm/parseq) (Bautista & Atienza, ECCV 2022) — a permuted autoregressive sequence transformer originally built for scene text recognition — for handwritten text recognition across ten Indic languages: **Hindi, Bengali, Telugu, Tamil, Gujarati, Gurumukhi, Oriya, Kannada, Malayalam, and Urdu**. We also investigate transfer learning from printed to handwritten text and apply lexicon-based post-OCR error correction.

This codebase is a fork of [PARSeq](https://github.com/baudm/parseq); see `NOTICE`/`LICENSE` for upstream attribution.

## Installation

Requires Python >= 3.9 and PyTorch >= 1.10 (this repo was developed/tested against `torch==1.13.1`, matching `requirements/core.txt`).

```bash
conda create -n parseq python=3.9
conda activate parseq
# Use specific platform build. Other PyTorch 1.13 options: cu116, cu117, rocm5.2
platform=cu117
make torch-${platform}
pip install -r requirements/core.${platform}.txt -e .[train,test]
```

## Dataset

We use [**IIIT-INDIC-HW-WORDS**](Datasets.md) (Gongidi & Jawahar, ICDAR 2021) — word-level handwritten images for the ten languages above. See [`Datasets.md`](Datasets.md) for the dataset reference, per-language statistics, and the expected LMDB directory layout.

Each language's data is a separate root directory with the following structure:

```
<lang>/datasets/
├── train/IIIT-INDIC-HW-WORDS/{data.mdb,lock.mdb}
├── val/IIIT-INDIC-HW-WORDS/{data.mdb,lock.mdb}
└── test/IIIT-INDIC-HW-WORDS/{data.mdb,lock.mdb}
```

`configs/charset/<language>.yaml` defines the character set for each of the ten languages (`hindi`, `bengali`, `telugu`, `tamil`, `gujarati`, `punjabi` [Gurumukhi script], `oriya`, `kannada`, `malayalam`, `urdu`).

## Training

Implementation details (Section 4.1 of the paper): 4 GPUs, ~160,000 iterations, batch size 254, patch size 8×4, 1-cycle LR scheduler for pretraining + SWA scheduler with Adam for training, max label length 35.

### Baseline (Hindi, Telugu, Tamil, Urdu)

```bash
./train.py +experiment=parseq charset=<language> \
  data.root_dir=<path to <lang>/datasets> data.train_dir=IIIT-INDIC-HW-WORDS \
  model.batch_size=254 trainer.accelerator=gpu trainer.devices=4
```

This uses the defaults already set in `configs/model/parseq.yaml`: `perm_num=6` (K=6), `dropout=0.1`, `lr=7e-4`.

### Fine-tuning (Bengali, Gujarati, Gurumukhi, Kannada, Odia, Malayalam)

Per Section 4.1, these languages were fine-tuned at a higher permutation count and dropout, with a lower learning rate:

```bash
./train.py +experiment=parseq charset=<language> \
  data.root_dir=<path to <lang>/datasets> data.train_dir=IIIT-INDIC-HW-WORDS \
  model.batch_size=254 model.perm_num=14 model.dropout=0.4 model.lr=7e-6 \
  trainer.accelerator=gpu trainer.devices=4 \
  ckpt_path=<path to that language's baseline checkpoint>
```

### Transfer learning from printed text (Table 4)

Section 4.2/4.6 describe a two-stage approach: pretrain on a printed-text dataset for the same language, then continue training on `IIIT-INDIC-HW-WORDS`. This repo does not include printed-text dataset preparation — build/obtain a printed-text LMDB in the same layout, pretrain on it, then continue training on the handwritten data using PARSeq's existing checkpoint-resume mechanism:

```bash
# Stage 1: pretrain on printed text
./train.py +experiment=parseq charset=<language> \
  data.root_dir=<path to printed data> trainer.max_epochs=30 \
  model.batch_size=254 trainer.accelerator=gpu trainer.devices=4

# Stage 2: continue training on handwritten data from the printed checkpoint
./train.py +experiment=parseq charset=<language> \
  data.root_dir=<path to <lang>/datasets> data.train_dir=IIIT-INDIC-HW-WORDS \
  model.batch_size=254 trainer.accelerator=gpu trainer.devices=4 \
  ckpt_path=<path to stage-1 checkpoint>
```

## Evaluation (Table 2)

```bash
./test.py <path to checkpoint>.ckpt --data_root=<path to <lang>/datasets>
```

`test.py` evaluates on `<data_root>/test/IIIT-INDIC-HW-WORDS` by default (`--test_set` accepts other subdirectory names under `<data_root>/test/`, e.g. for the in-vocab/out-of-vocab split analysis).

## Post-OCR error correction (Table 3)

`postocr.py` builds a lexicon from every ground-truth label in `train`+`val`+`test`, then for each prediction looks up the top-K lexicon words by edit distance (Recall 1–5) and reports WER/CER for each K:

```bash
python postocr.py <path to checkpoint>.ckpt --data_root=<path to <lang>/datasets>
```

## Qualitative comparisons (Fig. 4, 5)

- `test_combined.py` dumps every test image to disk (named by its LMDB record index) and logs per-sample predictions/confidence/CER against that index — used to build the qualitative prediction tables (Fig. 4).
- `crnn_compare.py` → `parseq_crnn_compare.py` → `final_compare_crnn.py` build the CNN-RNN vs. PARSeq comparison (Fig. 5). These are single-language, edit-the-language-constant-at-top scripts (not CLI-parameterized) and expect CRNN baseline predictions as JSON, produced separately (not part of this repo).
- `crnn_par_visual_comp.py` (`--language`, `--checkpoint`) is CLI-parameterized; see the docstring at the top of the file for the expected input JSON layout.

## Citation

```bibtex
@InProceedings{lalitha2024indichtr,
  author    = {Lalitha, Evani and Mondal, Ajoy and Jawahar, C. V.},
  title     = {Enhancing Accuracy in Indic Handwritten Text Recognition},
  booktitle = {Computer Vision and Image Processing (CVIP 2024)},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-93688-3_17}
}
```

This work builds on PARSeq:

```bibtex
@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  month={10},
  year={2022},
  publisher={Springer Nature Switzerland},
  address={Cham},
  doi={10.1007/978-3-031-19815-1_11},
  url={https://doi.org/10.1007/978-3-031-19815-1_11}
}
```

## License

Apache License 2.0 (see `LICENSE`). See `NOTICE` for upstream attribution — this repo is a fork of [baudm/parseq](https://github.com/baudm/parseq).
