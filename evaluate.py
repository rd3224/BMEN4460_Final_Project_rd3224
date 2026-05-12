"""
evaluate.py — Full test-set evaluation
=========================================
Usage:
    python evaluate.py --checkpoint checkpoints/stage3_best.pth
    python evaluate.py --checkpoint checkpoints/stage3_best.pth --ablation
"""
import argparse
import numpy as np
import torch

import config
from data.dataset import get_loaders, ChestXray14, build_test_transform
from models.thorax_net import ThoraxNet
from utils.metrics import (compute_aucs, print_auc_table,
                            plot_roc_curves, compare_branches)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(model, loader, device, ablation=False):
    """
    Run inference on `loader`, compute and display AUC metrics.
    If ablation=True also prints branch-level comparison.
    """
    model.eval()
    all_labels = []
    all_cls    = []
    all_att    = []
    all_diag   = []

    for images, labels, _ in loader:
        images = images.to(device)
        with torch.enable_grad():
            out = model(images)

        all_labels.append(labels.cpu().numpy())
        all_cls.append(out["y_cls"].detach().cpu().numpy())
        all_att.append(out["y_att"].detach().cpu().numpy())
        all_diag.append(out["y_diag"].detach().cpu().numpy())

    y_true  = np.concatenate(all_labels)
    y_cls   = np.concatenate(all_cls)
    y_att   = np.concatenate(all_att)
    y_diag  = np.concatenate(all_diag)

    print("\n── Thorax-Net (fused) ─────────────────────────────")
    auc_dict = compute_aucs(y_true, y_diag)
    print_auc_table(auc_dict)

    plot_roc_curves(y_true, y_diag, save_path="roc_thoraxnet.png")

    if ablation:
        compare_branches(y_true, y_cls, y_att, y_diag)

    return auc_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone",   default=config.BACKBONE)
    parser.add_argument("--ablation",   action="store_true",
                        help="Also print per-branch AUC comparison")
    args = parser.parse_args()

    device = get_device()
    _, test_loader, _, _ = get_loaders()  # use val split (test images not downloaded)

    model = ThoraxNet(backbone=args.backbone).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.checkpoint}  "
          f"(val AUC at save: {ckpt.get('val_auc', 'N/A')})")

    evaluate_model(model, test_loader, device, ablation=args.ablation)


if __name__ == "__main__":
    main()
