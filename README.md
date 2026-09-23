# Indic-HTR-CVIP-2024

Code for **"Enhancing Accuracy in Indic Handwritten Text Recognition"** (Evani Lalitha, Ajoy Mondal, C. V. Jawahar — CVIT, IIIT Hyderabad), accepted at [CVIP 2024](https://link.springer.com/chapter/10.1007/978-3-031-93688-3_17). See the [project page](https://lalithaevani.github.io/Indic-HTR-CVIP-2024-page/) for the abstract, method overview, and results.

We fine-tune [PARSeq](https://github.com/baudm/parseq) (Bautista & Atienza, ECCV 2022) — a permuted autoregressive sequence transformer originally built for scene text recognition — for handwritten text recognition across ten Indic languages: **Hindi, Bengali, Telugu, Tamil, Gujarati, Gurumukhi, Oriya, Kannada, Malayalam, and Urdu**. We also apply lexicon-based post-OCR error correction.

This codebase is a fork of [PARSeq](https://github.com/baudm/parseq); see `NOTICE`/`LICENSE` for upstream attribution.

## Installation

Requires Python >= 3.9. See `requirements.txt` for the exact package versions this was verified against.

```bash
conda create -n parseq python=3.9
conda activate parseq
# Install the PyTorch build matching your CUDA version first, e.g.:
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install -e .
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

Implementation details: 4 GPUs, ~160,000 iterations, batch size 254, patch size 8×4, 1-cycle LR scheduler for pretraining + SWA scheduler with Adam for training, max label length 35.

```bash
./train.py +experiment=parseq charset=<language> \
  data.root_dir=<path to <lang>/datasets> data.train_dir=IIIT-INDIC-HW-WORDS \
  model.batch_size=254 trainer.accelerator=gpu trainer.devices=4
```

This uses the defaults already set in `configs/model/parseq.yaml`: `perm_num=6` (K=6), `dropout=0.1`, `lr=7e-4`.

## Evaluation

```bash
./test.py <path to checkpoint>.ckpt --data_root=<path to <lang>/datasets>
```

`test.py` evaluates on `<data_root>/test/IIIT-INDIC-HW-WORDS` by default (`--test_set` accepts other subdirectory names under `<data_root>/test/`, e.g. for the in-vocab/out-of-vocab split analysis).

## Post-OCR error correction (Table 3)

`postocr.py` builds a lexicon from every ground-truth label in `train`+`val`+`test`, then for each prediction looks up the top-K lexicon words by edit distance (Recall 1–5) and reports WER/CER for each K:

```bash
python postocr.py <path to checkpoint>.ckpt --data_root=<path to <lang>/datasets>
```

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
