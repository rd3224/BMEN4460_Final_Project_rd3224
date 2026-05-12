# ChestX-ray14 Multi-label Classification

---

## Dataset

This project uses a **subset of the NIH ChestX-ray14 dataset**.

Instead of the full dataset (~112k images), only the **first three NIH image archives** were used:

- `images_001.tar.gz`
- `images_002.tar.gz`
- `images_003.tar.gz`

These archives contain **24,999 chest X-ray images** in total.

### Official Dataset Split

This project follows the **official NIH ChestX-ray14 patient-wise split**.

The NIH dataset provides two predefined image lists:

```text
train_val_list.txt
test_list.txt
```

The split is performed **at the patient level**, meaning images from the same patient do not appear in both training and testing sets, preventing data leakage.

For this project, only the first three downloaded image archives were used (`images_001.tar.gz`–`images_003.tar.gz`), containing **24,999 images** in total.

After intersecting the downloaded images with the official NIH split files:

| Category | Number of Images |
|-----------|----------------:|
| Downloaded images | 24,999 |
| Images in official `train_val_list.txt` | 20,931 |
| Images in official `test_list.txt` | 4,068 |
| Images not in any official split | 0 |

Thus, the downloaded subset naturally contains both training and testing patients. To strictly follow the NIH official split, only the **20,931 images belonging to `train_val_list.txt`** were used for training/validation.

The file:

```text
train_val_list_subset.txt
```

was generated as:

```text
downloaded_images ∩ train_val_list.txt
```

ensuring full consistency with the official NIH split. Images belonging to the official test set (**4,068 images**) were completely excluded from training.
After filtering according to the provided train/validation split file, the final dataset used for model training and validation contains:

**Total images used: 20,931**

### Disease Distribution

| Disease | Samples |
|----------|---------:|
| Atelectasis | 1,805 |
| Cardiomegaly | 442 |
| Effusion | 1,745 |
| Infiltration | 2,653 |
| Mass | 698 |
| Nodule | 1,014 |
| Pneumonia | 195 |
| Pneumothorax | 648 |
| Consolidation | 654 |
| Edema | 228 |
| Emphysema | 356 |
| Fibrosis | 415 |
| Pleural Thickening | 539 |
| Hernia | 33 |


> The dataset is highly imbalanced, particularly for rare classes such as **Hernia**, **Edema**, and **Pneumonia**. To alleviate class imbalance, a **weighted binary cross-entropy (BCE) loss** is adopted during training.

---

## Results (20k Subset, Validation Set)

| Checkpoint | mean_AUC | focus_AUC | Note |
|------------|----------|-----------|------|
| Stage 1 | 0.8136 | 0.8102 | classification branch only |
| Stage 1 → Stage 3 | **0.8176** | **0.8159** | recommended |

---

## Environment

Install dependencies:

```bash
pip install torch torchvision scikit-learn pandas matplotlib pillow
```

---

## Data Preparation

Download the NIH ChestX-ray14 dataset:

https://nihcc.app.box.com/v/ChestXray-NIHCC

### Step 1 — Images

Download only the following archives:

```text
images_001.tar.gz
images_002.tar.gz
images_003.tar.gz
```

Extract them into:

```text
data/chestxray14/images/
```

### Step 2 — Labels CSV

Download:

```text
Data_Entry_2017.csv
```

Place it under:

```text
data/chestxray14/
```

### Step 3 — Bounding Boxes (Optional)

For visualization, download:

```text
BBox_List_2017.csv
```

### Final Directory Structure

```text
data/chestxray14/
├── images/
├── Data_Entry_2017.csv
├── BBox_List_2017.csv             # optional
├── train_val_list_subset.txt
├── train_val_list.txt
└── test_list.txt
```

---

## Key Configuration (`config.py`)

| Parameter | Default | Note |
|------------|---------|------|
| `MAX_ITER` | 8,000 | ≈10 epochs on 20k subset |
| `LR` | `1e-3` | Stage 3 uses LR × 0.1 |
| `LR_PATIENCE` | 3 | epochs before LR decay |
| `BATCH_SIZE` | 24 | |
| `BACKBONE` | `resnet152` | optional: `efficientnet_b4` |

---

## Training

### Recommended: Stage 1 → Stage 3

```bash
# Stage 1: train classification branch
nohup python3 train.py --stage 1 >> train_stage1.log 2>&1 &

# Stage 3: end-to-end fine-tuning
nohup python3 train.py --stage 3 \
--resume checkpoints/stage1_best.pth \
>> train_stage3.log 2>&1 &
```

### Full Three-Stage Training

```bash
nohup python3 train.py --stage 1 >> train_stage1.log 2>&1 &

nohup python3 train.py --stage 2 \
--resume checkpoints/stage1_best.pth \
>> train_stage2.log 2>&1 &

nohup python3 train.py --stage 3 \
--resume checkpoints/stage2_best.pth \
>> train_stage3.log 2>&1 &
```

### Resume Interrupted Training

```bash
python3 train.py --stage 1 \
--resume checkpoints/stage1_best.pth
```

---

## Evaluation

### Validation AUC

```bash
python3 evaluate.py \
--checkpoint checkpoints/stage3_best.pth
```

### Ablation Study

```bash
python3 evaluate.py \
--checkpoint checkpoints/stage3_best.pth \
--ablation
```

---

## Visualization

Generate a 9-image visualization grid:

```bash
python3 visualize.py \
--checkpoint checkpoints/stage3_best.pth \
--batch 9 \
--out vis/
```

When `BBox_List_2017.csv` is available:

- **Left:** original image with ground-truth bounding box  
- **Right:** model attention heatmap

---

## Monitor Training

```bash
tail -f train_stage3.log
nvidia-smi
```

---

## Model Architecture

| Component | Detail |
|------------|--------|
| Backbone | ResNet-152 pretrained on ImageNet |
| Classification Branch | ResNet-152 → FC(2048 → 14) → sigmoid |
| Attention Branch | pre-conv → Grad-CAM → post-conv → sigmoid |
| Diagnosis Output | `y_diag = (y_cls + y_att) / 2` |
| Loss Function | Weighted BCE |
| Stage 2 Optimizer | Adam (`lr=1e-4`) |

The model follows an **attention-guided diagnosis framework**, where Grad-CAM generated attention maps help the network focus on disease-relevant thoracic regions, improving classification performance under weak supervision.
