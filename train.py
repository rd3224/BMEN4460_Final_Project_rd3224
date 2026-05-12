"""
train.py — Three-stage training for Thorax-Net
================================================
Usage:
    python train.py --stage 1          # fine-tune classification branch
    python train.py --stage 2          # train attention branch (cls frozen)
    python train.py --stage 3          # end-to-end fine-tune
    python train.py --stage all        # run all three stages sequentially

Checkpoints are saved after each stage to ./checkpoints/.
"""
import os
import time
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from data.dataset import get_loaders
from models.thorax_net import ThoraxNet, WeightedBCELoss
from utils.metrics import compute_aucs, print_auc_table

# ── Reproducibility ─────────────────────────────────────────────────────────────
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Single epoch ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, stage):
    model.train()
    total_loss = 0.0

    for step, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if stage == 1:
            # Classification branch only — cheaper, no attention Grad-CAM
            pred = model.forward_cls_only(images)
            loss = criterion(pred, labels)
        else:
            out  = model(images)
            if stage == 2:
                loss = criterion(out["y_att"], labels)
            else:                        # stage 3: end-to-end weighted loss
                loss = (config.STAGE3_W_CLS  * criterion(out["y_cls"],  labels) +
                        config.STAGE3_W_ATT  * criterion(out["y_att"],  labels) +
                        config.STAGE3_W_DIAG * criterion(out["y_diag"], labels))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ── Validation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, stage):
    model.eval()
    all_labels = []
    all_preds  = []
    total_loss = 0.0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        if stage == 1:
            pred = model.forward_cls_only(images)
            loss = criterion(pred, labels)
            all_preds.append(pred.cpu())
        else:
            with torch.enable_grad():    # needed for attention Grad-CAM
                out = model(images)
            pred = out["y_diag"] if stage == 3 else out["y_att"]
            loss = criterion(pred, labels)
            all_preds.append(pred.cpu())

        total_loss += loss.item()
        all_labels.append(labels.cpu())

    y_true  = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_preds).numpy()
    auc_dict = compute_aucs(y_true, y_score)

    return total_loss / len(loader), auc_dict


# ── Stage training loop ────────────────────────────────────────────────────────
def run_stage(stage: int, model, train_loader, val_loader, train_ds,
              device, resume_ckpt: str = None):

    print(f"\n{'='*60}")
    print(f"  Stage {stage} training")
    print(f"{'='*60}")

    model.train_stage(stage)

    # Compute class weights from training data
    beta_pos, beta_neg = train_ds.dataset.class_weights() \
        if hasattr(train_ds, "dataset") else train_ds.class_weights()
    criterion = WeightedBCELoss(
        torch.tensor(beta_pos).to(device),
        torch.tensor(beta_neg).to(device),
    )

    # Different LR / optimizer for each stage
    lr = config.LR if stage in (1, 2) else config.LR * 0.1
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if stage == 2:
        # Adam normalises gradient magnitudes per-parameter, which is essential
        # for Stage 2: Grad-CAM attenuates gradients to pre_conv by ~1e-5,
        # making SGD updates negligible. Adam adapts to these tiny gradients.
        optimizer = optim.Adam(params, lr=1e-4, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=config.MOMENTUM,
                              weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config.LR_DECAY_FACTOR,
                                  patience=config.LR_PATIENCE)

    # Optionally resume from checkpoint
    start_epoch = 0
    if resume_ckpt and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        load_res = model.load_state_dict(ckpt["model"], strict=False)
        if load_res.missing_keys or load_res.unexpected_keys:
            print("  Partial checkpoint load due to architecture mismatch:")
            if load_res.missing_keys:
                print(f"    missing keys: {len(load_res.missing_keys)}")
            if load_res.unexpected_keys:
                print(f"    unexpected keys: {len(load_res.unexpected_keys)}")
        # Only restore optimizer state when resuming within the same stage;
        # across stages the parameter groups differ and cannot be loaded.
        if ckpt.get("stage") == stage:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt.get("epoch", 0) + 1
            except Exception:
                print("  Optimizer state not restored (parameter groups changed).")
        print(f"  Resumed from {resume_ckpt} (epoch {start_epoch})")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_auc   = 0.0
    best_ckpt  = os.path.join(config.CHECKPOINT_DIR, f"stage{stage}_best.pth")

    # Estimate epochs from MAX_ITER
    iters_per_epoch = len(train_loader)
    max_epochs      = max(1, config.MAX_ITER // iters_per_epoch)

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                      device, stage)
        val_loss, auc_dict = validate(model, val_loader, criterion, device, stage)
        scheduler.step(auc_dict["mean_auc"])

        val_auc   = auc_dict["mean_auc"]
        focus_auc = auc_dict["focus_auc"]
        elapsed   = time.time() - t0
        cur_lr    = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1:>4}/{max_epochs} | stage {stage} | lr {cur_lr:.2e} | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"mean_AUC {val_auc:.4f} | focus_AUC {focus_auc:.4f} | {elapsed:.0f}s")
        print_auc_table(auc_dict)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch":     epoch,
                "stage":     stage,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_auc":   val_auc,
            }, best_ckpt)
            print(f"  ✓ New best AUC {best_auc:.4f} — saved to {best_ckpt}")

    print(f"\nStage {stage} complete | best val AUC = {best_auc:.4f}")
    # Load best weights before returning
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
    return best_ckpt


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        help="Training stage: 1 | 2 | 3 | all")
    parser.add_argument("--backbone", default=config.BACKBONE)
    parser.add_argument("--resume",   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, train_ds = get_loaders()

    model = ThoraxNet(backbone=args.backbone).to(device)

    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    last_ckpt = args.resume

    for stage in stages:
        last_ckpt = run_stage(stage, model, train_loader, val_loader, train_ds,
                               device, resume_ckpt=last_ckpt)

    # Quick final evaluation on test set
    print("\nRunning final test evaluation …")
    from evaluate import evaluate_model
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
