# Thorax-Net

Reproduction of **Wang et al., IEEE JBHI 2020** — three-stage attention-guided classification of 14 thorax diseases on ChestX-ray14.

---

## Environment

```bash
pip install torch torchvision scikit-learn pandas matplotlib pillow tqdm
```

---

## Data Setup

```
data/chestxray14/
├── images/                        # all .png images
├── train_val_list_subset.txt      # filenames used for train/val (20,930 images)
├── test_list.txt                  # NIH official test split (images not downloaded)
├── Data_Entry_2017.csv            # labels for all images
└── BBox_List_2017.csv             # bounding boxes (983 entries, optional)
```

> **Labels** are read from `Data_Entry_2017.csv`, not from the txt files.  
> **train/val split**: 90 / 10 random split from `train_val_list_subset.txt`, fixed by `SEED=42`.

To use the full dataset, update `config.py`:
```python
TRAIN_LIST = os.path.join(DATA_ROOT, "train_val_list.txt")   # 86,523 images
```

---

## Key Config (`config.py`)

| Parameter | Value | Note |
|-----------|-------|------|
| `MAX_ITER` | 8,000 | ≈ 10 epochs on 20k subset |
| `LR` | 1e-3 | Stage 1/2; Stage 3 uses LR×0.1 |
| `LR_PATIENCE` | 3 | epochs before LR decay |
| `BATCH_SIZE` | 24 | |
| `BACKBONE` | resnet152 | or `efficientnet_b4` |

---

## Training

### Recommended: Stage 1 → Stage 3 (skip Stage 2)

Stage 2 overfits on small subsets due to vanishing Grad-CAM gradients.
Stage 1 → Stage 3 directly gives better results on 20k data.

```bash
# Stage 1: fine-tune classification branch
nohup python3 train.py --stage 1 >> train_stage1.log 2>&1 &

# Stage 3: end-to-end fine-tune from Stage 1 checkpoint
nohup python3 train.py --stage 3 --resume checkpoints/stage1_best.pth >> train_stage3_from1.log 2>&1 &
```

### Full three-stage path (recommended for full dataset ≥ 86k images)

```bash
nohup python3 train.py --stage 1 >> train_stage1.log 2>&1 &
nohup python3 train.py --stage 2 --resume checkpoints/stage1_best.pth >> train_stage2.log 2>&1 &
nohup python3 train.py --stage 3 --resume checkpoints/stage2_best.pth >> train_stage3.log 2>&1 &
```

### Resume interrupted training (same stage)

```bash
python3 train.py --stage 2 --resume checkpoints/stage2_best.pth
```

> Optimizer state is restored only when resuming within the **same stage**.  
> Cross-stage resume loads model weights only (optimizer resets).

---

## Evaluation

```bash
# Overall AUC on val set
python3 evaluate.py --checkpoint checkpoints/stage3_best.pth

# Ablation: compare y_cls / y_att / y_diag
python3 evaluate.py --checkpoint checkpoints/stage3_best.pth --ablation
```

**Results summary (val set, 20k subset):**

| Checkpoint | mean_AUC | Note |
|-----------|----------|------|
| `stage1_best.pth` | 0.8136 | y_cls only |
| `stage3_best.pth` (S2→S3) | 0.8052 (y_diag) / 0.8156 (y_cls) | att branch too weak |
| `stage3_best.pth` (S1→S3) | **0.8176** (y_diag) | best overall |

> On 20k subset, `y_att` AUC ≈ 0.60, so fusion `y_diag` only helps when starting from Stage 1.  
> On full dataset (86k), both branches reach 0.80+ and fusion improves results.

---

## Visualization

```bash
# 9-image grid, random single-label images from {Atelectasis, Effusion, Infiltration, Nodule}
python3 visualize.py --checkpoint checkpoints/stage3_best.pth --batch 9 --out vis/
```

With `BBox_List_2017.csv` present: left column shows original image + green GT bounding box, right column shows attention heatmap overlay.

---

## Monitor Training

```bash
tail -f train_stage3_from1.log     # live log
nvidia-smi                          # GPU usage
pgrep -a python3 | grep train      # running processes
```

---

## Architecture Notes

| Component | Detail |
|-----------|--------|
| cls_branch | ResNet-152 (ImageNet pretrained), FC → 14-sigmoid |
| att_branch | pre_conv (1×1,3×3,1×1) → Grad-CAM → post_conv (1×1,1×1,14×14) |
| Diagnosis | `y_diag = (y_cls + y_att) / 2` |
| Loss | Weighted BCE: `β_pos=(P+N)/P`, `β_neg=(P+N)/N` |
| Stage 2 optimizer | Adam (lr=1e-4) — SGD fails due to vanishing Grad-CAM gradients |
| Grad-CAM fix | Gradients w.r.t. `feat` (not `refined`); cam standardized before softmax |
