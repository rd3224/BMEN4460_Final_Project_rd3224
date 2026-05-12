# Thorax-Net

Reproduction of **Wang et al., "Thorax-Net: An Attention Regularized Deep Neural Network for Classification of Thoracic Diseases on Chest Radiography," IEEE JBHI 2020.**

Three-stage attention-guided classification of 14 thorax diseases on [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC).

---

## Results (20k subset, val set)

| Checkpoint | mean_AUC | focus_AUC | Note |
|-----------|----------|-----------|------|
| Stage 1 | 0.8136 | 0.8102 | cls branch only |
| Stage 1 → Stage 3 | **0.8176** | **0.8159** | recommended |

---

## Environment

```bash
pip install torch torchvision scikit-learn pandas matplotlib pillow
```

---

## Data Download

**Data is not included in this repo.** Download from NIH:

### Step 1 — Images

Go to: https://nihcc.app.box.com/v/ChestXray-NIHCC

Download all 12 image zip files (`images_001.tar.gz` … `images_012.tar.gz`) and extract into:

```
data/chestxray14/images/
```

Or use the batch download script provided by NIH:

```bash
# After downloading batch_download_zips.py from the NIH page:
python batch_download_zips.py
cd data/chestxray14
for f in *.tar.gz; do tar -xzf "$f"; done
```

### Step 2 — Labels CSV

Download `Data_Entry_2017.csv` from the same NIH page and place it at:

```
data/chestxray14/Data_Entry_2017.csv
```

### Step 3 — Bounding Boxes (optional, for visualization)

Download `BBox_List_2017.csv` from the same NIH page:

```
data/chestxray14/BBox_List_2017.csv
```

### Final directory structure

```
data/chestxray14/
├── images/                        # ~112k .png files (~45 GB)
├── Data_Entry_2017.csv            # labels (required)
├── BBox_List_2017.csv             # bounding boxes (optional)
├── train_val_list_subset.txt      # ← included in repo (20,930 images)
├── train_val_list.txt             # ← included in repo (86,523 images)
└── test_list.txt                  # ← included in repo
```

> Labels are read from `Data_Entry_2017.csv`. The `.txt` files only determine which images belong to train/val/test splits.

To use the full dataset (86k images), update `config.py`:
```python
TRAIN_LIST = os.path.join(DATA_ROOT, "train_val_list.txt")
```

---

## Key Config (`config.py`)

| Parameter | Default | Note |
|-----------|---------|------|
| `MAX_ITER` | 8,000 | ≈ 10 epochs on 20k subset |
| `LR` | 1e-3 | Stage 3 uses LR × 0.1 |
| `LR_PATIENCE` | 3 | epochs before LR decay |
| `BATCH_SIZE` | 24 | |
| `BACKBONE` | resnet152 | or `efficientnet_b4` |

---

## Training

### Recommended: Stage 1 → Stage 3

```bash
# Stage 1: fine-tune classification branch (~2.5h on 20k)
nohup python3 train.py --stage 1 >> train_stage1.log 2>&1 &

# Stage 3: end-to-end fine-tune from Stage 1 checkpoint
nohup python3 train.py --stage 3 --resume checkpoints/stage1_best.pth >> train_stage3.log 2>&1 &
```

### Full three-stage path (for full 86k dataset)

```bash
nohup python3 train.py --stage 1 >> train_stage1.log 2>&1 &
nohup python3 train.py --stage 2 --resume checkpoints/stage1_best.pth >> train_stage2.log 2>&1 &
nohup python3 train.py --stage 3 --resume checkpoints/stage2_best.pth >> train_stage3.log 2>&1 &
```

### Resume interrupted training

```bash
# Resumes within the same stage (optimizer state restored)
python3 train.py --stage 1 --resume checkpoints/stage1_best.pth
```

---

## Evaluation

```bash
# Per-class AUC on val set
python3 evaluate.py --checkpoint checkpoints/stage3_best.pth

# Ablation: y_cls vs y_att vs y_diag
python3 evaluate.py --checkpoint checkpoints/stage3_best.pth --ablation
```

---

## Visualization

```bash
# 9-image grid (single-label, Atelectasis/Effusion/Infiltration/Nodule only)
python3 visualize.py --checkpoint checkpoints/stage3_best.pth --batch 9 --out vis/
```

With `BBox_List_2017.csv` present: left = original + green GT bbox, right = attention heatmap.

---

## Monitor Training

```bash
tail -f train_stage3.log
nvidia-smi
```

---

## Architecture

| Component | Detail |
|-----------|--------|
| Backbone | ResNet-152 pretrained on ImageNet |
| cls_branch | ResNet-152 → FC(2048→14) → sigmoid |
| att_branch | pre_conv (1×1, 3×3, 1×1) → Grad-CAM → post_conv (1×1, 1×1, 14×14) → sigmoid |
| Diagnosis | `y_diag = (y_cls + y_att) / 2` |
| Loss | Weighted BCE: `β_pos = (P+N)/P`, `β_neg = (P+N)/N` |
| Stage 2 optimizer | Adam lr=1e-4 (SGD fails due to vanishing Grad-CAM gradients on small datasets) |
