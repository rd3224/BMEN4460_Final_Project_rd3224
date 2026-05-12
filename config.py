"""
Thorax-Net Configuration
Reproducing: Wang et al., IEEE JBHI 2020
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT      = "/home/Gloria/4460/thorax_net/data/chestxray14"
IMAGE_DIR      = os.path.join(DATA_ROOT, "images")
TRAIN_LIST     = os.path.join(DATA_ROOT, "train_val_list_subset.txt")
VAL_LIST       = os.path.join(DATA_ROOT, "train_val_list_subset.txt")
TEST_LIST      = os.path.join(DATA_ROOT, "test_list_subset.txt")
ENTRY_CSV      = os.path.join(DATA_ROOT, "Data_Entry_2017.csv")
BBOX_FILE      = os.path.join(DATA_ROOT, "BBox_List_2017.csv")
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR        = "./logs"

# ── Dataset ────────────────────────────────────────────────────────────────────
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES  = 14
IMAGE_SIZE   = 224          # crop size fed to the network
RESIZE_SIZE  = 256          # intermediate resize before crop
USE_TEN_CROP = False        # True→ ten-crop TTA at test time (paper uses center crop)

# ── Model ──────────────────────────────────────────────────────────────────────
# BACKBONE choices: "resnet152" (paper) | "efficientnet_b4" (our variant)
BACKBONE           = "resnet152"
PRETRAINED         = True
ATT_FEATURE_SIZE   = 14     # spatial size of attention branch input (layer3 of ResNet-152)
ATT_IN_CHANNELS    = 1024   # channels at penultimate residual module of ResNet-152
                             # For EfficientNet-B4 change to 272 and ATT_FEATURE_SIZE=14

# ── Training – three-stage schedule (paper §IV-C) ─────────────────────────────
BATCH_SIZE       = 24
NUM_WORKERS      = 4
MAX_ITER         = 4_000   # each stage (~20 epochs on the subset)
LR               = 1e-3
MOMENTUM         = 0.9
WEIGHT_DECAY     = 5e-4
LR_DECAY_FACTOR  = 0.1
LR_PATIENCE      = 3        # epochs before LR decay
VAL_SPLIT        = 0.10     # 10 % of train subjects → validation

# Stage-3 loss weights: raise attention pressure so y_att does not collapse
STAGE3_W_CLS     = 0.25
STAGE3_W_ATT     = 0.50
STAGE3_W_DIAG    = 0.25

# ── Loss ───────────────────────────────────────────────────────────────────────
# Weighted binary cross-entropy: βP=(P+N)/P, βN=(P+N)/N  (paper eq.4)
USE_WEIGHTED_BCE = True

# ── Evaluation ─────────────────────────────────────────────────────────────────
# Focus subset recommended by reviewer critique (most reliable labels)
FOCUS_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Mass", "Nodule", "Pneumothorax", "Emphysema",
]

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
