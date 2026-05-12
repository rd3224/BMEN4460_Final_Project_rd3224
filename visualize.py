"""
visualize.py — Attention heatmap visualization
================================================
Visualization utilities for the current learnable class-wise attention model.

Usage:
    python visualize.py --checkpoint checkpoints/stage3_best.pth \
                        --image_path data/chestxray14/images/00000001_000.png \
                        --label "Atelectasis|Effusion"

    # Batch mode: pick N random test images and save a grid
    python visualize.py --checkpoint checkpoints/stage3_best.pth --batch 9
"""
import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms.functional as TF

import pandas as pd
import matplotlib.patches as patches

import config
from data.dataset import build_test_transform, ChestXray14, get_loaders
from models.thorax_net import ThoraxNet


def load_bbox_map(bbox_csv: str) -> dict:
    """
    Load BBox_List_2017.csv → {filename: [(label, x, y, w, h), ...]}
    NIH columns: Image Index, Finding Label, Bbox[x,y,w,h]
    Coordinates are in original image pixel space.
    """
    if not os.path.isfile(bbox_csv):
        return {}
    df = pd.read_csv(bbox_csv, header=0)
    bbox_map = {}
    for _, row in df.iterrows():
        fname = str(row.iloc[0]).strip()
        label = str(row.iloc[1]).strip()
        x, y, w, h = float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4]), float(row.iloc[5])
        bbox_map.setdefault(fname, []).append((label, x, y, w, h))
    return bbox_map


def transform_bbox(x, y, w, h, orig_w, orig_h,
                   resize=256, crop=224):
    """Map NIH bbox coords → 224×224 display coords after Resize+CenterCrop."""
    scale  = resize / min(orig_w, orig_h)
    new_w, new_h = orig_w * scale, orig_h * scale
    # shift from CenterCrop
    off_x  = (new_w - crop) / 2
    off_y  = (new_h - crop) / 2
    x2 = x * scale - off_x
    y2 = y * scale - off_y
    w2 = w * scale
    h2 = h * scale
    # clamp to image boundary
    x2 = max(0, x2); y2 = max(0, y2)
    w2 = min(w2, crop - x2); h2 = min(h2, crop - y2)
    return x2, y2, w2, h2


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_pil(t):
    """De-normalise and convert tensor [3,H,W] → PIL."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = t.cpu() * std + mean
    t = t.clamp(0, 1)
    return TF.to_pil_image(t)


def overlay_heatmap(pil_img, heatmap: np.ndarray, alpha=0.45):
    """
    Overlay a 2-D attention map on a PIL image.
    heatmap: [H, W] values in [0,1]
    Returns: PIL image with coloured overlay.
    """
    h, w = pil_img.size[1], pil_img.size[0]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h),
                                                                   Image.BILINEAR)
    ) / 255.0
    colormap = cm.jet(heatmap_resized)[..., :3]  # [H, W, 3] float
    base = np.array(pil_img).astype(float) / 255.0
    overlay = (1 - alpha) * base + alpha * colormap
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def visualize_single(model, image_tensor, label_vec, device,
                      true_labels=None, save_path="heatmap.png"):
    """
    Visualise attention heatmaps for a single image alongside predictions.
    Mirrors Fig. 4 in the paper.
    """
    model.eval()
    img_t = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)

    y_diag   = out["y_diag"][0].cpu().numpy()        # [14]
    att_maps = out["att_maps"][0].detach().cpu().numpy()  # [14, H, W]

    pil_img = tensor_to_pil(image_tensor)

    # ── Figure layout: original | heatmap (top predicted disease) | scores ──
    top_class = int(np.argmax(y_diag))
    heatmap   = att_maps[top_class]
    heatmap   = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pil_img, cmap="gray")
    axes[0].set_title("Input Radiograph")
    axes[0].axis("off")

    overlay = overlay_heatmap(pil_img.convert("RGB"), heatmap)
    axes[1].imshow(overlay)
    axes[1].set_title(f"Attention: {config.CLASSES[top_class]}")
    axes[1].axis("off")

    # Score text
    score_lines = [f"{config.CLASSES[i]}: {y_diag[i]:.4f}" for i in range(14)]
    fig.text(0.5, 0.01, "  ".join(score_lines[:7]) + "\n" +
             "  ".join(score_lines[7:]),
             ha="center", fontsize=7, family="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def visualize_batch_grid(model, loader, device, n=9, save_path="grid.png",
                          bbox_map: dict = None):
    """
    Grid of n images: left = original + GT bbox, right = heatmap overlay.
    Prioritises images that have bbox annotations when bbox_map is provided.
    """
    FOCUS = {"Atelectasis", "Effusion", "Infiltration", "Nodule"}
    focus_idx = {config.CLASSES.index(c) for c in FOCUS}
    bbox_map  = bbox_map or {}

    model.eval()
    has_bbox, no_bbox = [], []

    for images, labels, fnames in loader:
        for i in range(len(images)):
            lbl = labels[i]
            pos = lbl.nonzero(as_tuple=True)[0].tolist()
            if len(pos) == 1 and pos[0] in focus_idx:
                entry = (images[i], lbl, fnames[i])
                if fnames[i] in bbox_map:
                    has_bbox.append(entry)
                else:
                    no_bbox.append(entry)

    random.shuffle(has_bbox); random.shuffle(no_bbox)
    # fill quota with bbox images first
    collected = (has_bbox + no_bbox)[:n]

    cols = 3
    # two sub-columns per image (original | heatmap)
    fig, axes = plt.subplots(len(collected), 2,
                              figsize=(8, len(collected) * 3.5))
    if len(collected) == 1:
        axes = axes[np.newaxis, :]

    for idx, (img_t, label, fname) in enumerate(collected):
        img_t_dev = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t_dev)

        y_cls    = out["y_cls"][0].detach().cpu().numpy()
        y_att    = out["y_att"][0].detach().cpu().numpy()
        y_diag   = out["y_diag"][0].detach().cpu().numpy()
        att_maps = out["att_maps"][0].detach().cpu().numpy()
        true_cls = [config.CLASSES[i] for i in range(14) if label[i] > 0.5]
        gt_label = true_cls[0] if true_cls else "Normal"
        gt_idx   = config.CLASSES.index(gt_label) if gt_label in config.CLASSES else 0

        # use GT class attention map for fair comparison with bbox
        heatmap = att_maps[gt_idx]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        pil_img = tensor_to_pil(img_t).convert("RGB")

        # ── Left: original + GT bbox ─────────────────────────────────────────
        ax_orig = axes[idx, 0]
        ax_orig.imshow(pil_img)
        if fname in bbox_map:
            orig_img = Image.open(os.path.join(config.IMAGE_DIR, fname))
            orig_w, orig_h = orig_img.size
            for (blabel, bx, by, bw, bh) in bbox_map[fname]:
                if blabel == gt_label:
                    x2, y2, w2, h2 = transform_bbox(bx, by, bw, bh, orig_w, orig_h)
                    rect = patches.Rectangle((x2, y2), w2, h2,
                                             linewidth=2, edgecolor="lime",
                                             facecolor="none")
                    ax_orig.add_patch(rect)
            ax_orig.set_title(f"{fname}\nGT: {gt_label} | bbox: ✓", fontsize=7)
        else:
            ax_orig.set_title(f"{fname}\nGT: {gt_label} | bbox: –", fontsize=7)
        ax_orig.axis("off")

        # ── Right: heatmap overlay ────────────────────────────────────────────
        ax_heat = axes[idx, 1]
        overlay = overlay_heatmap(pil_img, heatmap)
        ax_heat.imshow(overlay)
        ax_heat.set_title(
            f"Attention ({gt_label})\ny_att={y_att[gt_idx]:.2f} | y_diag={y_diag[gt_idx]:.2f}",
            fontsize=7,
        )
        ax_heat.axis("off")

    plt.suptitle("Thorax-Net — GT bbox (green) vs Attention Heatmap", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid → {save_path}")


def visualize_comparison(model1, model2, loader, device, n=9,
                          label1="Stage1", label2="Stage3",
                          save_path="comparison.png"):
    """
    For n randomly selected correctly-predicted single-label images, show:
      col 0: original image
      col 1: model1 attention heatmap (GT class)
      col 2: model2 attention heatmap (GT class)
    "Correctly predicted" = argmax(y_cls) == GT label (top-1 match).
    """
    FOCUS = {"Atelectasis", "Effusion", "Infiltration", "Nodule"}
    focus_idx = {config.CLASSES.index(c) for c in FOCUS}

    model1.eval(); model2.eval()
    correct = []

    for images, labels, fnames in loader:
        for i in range(len(images)):
            lbl = labels[i]
            pos = lbl.nonzero(as_tuple=True)[0].tolist()
            if len(pos) != 1 or pos[0] not in focus_idx:
                continue
            gt_idx = pos[0]
            img_dev = images[i].unsqueeze(0).to(device)
            with torch.no_grad():
                out2 = model2(img_dev)
            pred_idx = int(out2["y_cls"][0].detach().argmax().item())
            if pred_idx == gt_idx:
                correct.append((images[i], lbl, fnames[i]))

    random.shuffle(correct)
    collected = correct[:n]
    print(f"Found {len(correct)} correct predictions, using {len(collected)}")

    fig, axes = plt.subplots(len(collected), 3,
                              figsize=(11, len(collected) * 3.2))
    if len(collected) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", label1, label2]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10, fontweight="bold", pad=8)

    for idx, (img_t, label, fname) in enumerate(collected):
        gt_idx  = label.nonzero(as_tuple=True)[0].item()
        gt_name = config.CLASSES[gt_idx]
        pil_img = tensor_to_pil(img_t).convert("RGB")
        img_dev = img_t.unsqueeze(0).to(device)

        def get_heatmap(model):
            with torch.no_grad():
                out = model(img_dev)
            y_cls    = out["y_cls"][0].detach().cpu().numpy()
            att_maps = out["att_maps"][0].detach().cpu().numpy()
            y_att    = out["y_att"][0].detach().cpu().numpy()
            y_diag   = out["y_diag"][0].detach().cpu().numpy()
            hm = att_maps[gt_idx]
            hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            return hm, float(y_cls[gt_idx]), float(y_att[gt_idx]), float(y_diag[gt_idx])

        hm1, cls1, att1, diag1 = get_heatmap(model1)
        hm2, cls2, att2, diag2 = get_heatmap(model2)

        # original
        axes[idx, 0].imshow(pil_img)
        axes[idx, 0].set_ylabel(f"{fname}\nGT: {gt_name}", fontsize=6, rotation=0,
                                 labelpad=60, va="center")
        axes[idx, 0].axis("off")

        # model 1 heatmap
        axes[idx, 1].imshow(overlay_heatmap(pil_img, hm1))
        axes[idx, 1].set_title(f"cls:{cls1:.3f} att:{att1:.3f} diag:{diag1:.3f}", fontsize=8)
        axes[idx, 1].axis("off")

        # model 2 heatmap
        axes[idx, 2].imshow(overlay_heatmap(pil_img, hm2))
        axes[idx, 2].set_title(f"cls:{cls2:.3f} att:{att2:.3f} diag:{diag2:.3f}", fontsize=8)
        axes[idx, 2].axis("off")

    plt.suptitle(f"Correctly predicted — {label1} vs {label2} attention heatmaps",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True, help="Primary checkpoint")
    parser.add_argument("--checkpoint2", default=None,  help="Second checkpoint for comparison mode")
    parser.add_argument("--label1",      default="Stage1", help="Label for first checkpoint")
    parser.add_argument("--label2",      default="Stage3", help="Label for second checkpoint")
    parser.add_argument("--backbone",    default=config.BACKBONE)
    parser.add_argument("--image_path",  default=None)
    parser.add_argument("--label",       default="")
    parser.add_argument("--batch",       type=int, default=0)
    parser.add_argument("--out",         default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()

    model = ThoraxNet(backbone=args.backbone).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    transform = build_test_transform()

    if args.image_path:
        img = Image.open(args.image_path).convert("RGB")
        img_t = transform(img)
        label_vec = torch.zeros(14)
        for lbl in args.label.split("|"):
            lbl = lbl.strip()
            if lbl in config.CLASSES:
                label_vec[config.CLASSES.index(lbl)] = 1.0
        visualize_single(model, img_t, label_vec, device,
                          save_path=os.path.join(args.out, "heatmap_single.png"))

    if args.checkpoint2:
        # ── Comparison mode: two checkpoints side by side ─────────────────────
        model2 = ThoraxNet(backbone=args.backbone).to(device)
        ckpt2  = torch.load(args.checkpoint2, map_location=device)
        model2.load_state_dict(ckpt2["model"], strict=False)
        n = args.batch if args.batch > 0 else 9
        _, val_loader, _, _ = get_loaders(batch_size=32)
        visualize_comparison(model, model2, val_loader, device, n=n,
                              label1=args.label1, label2=args.label2,
                              save_path=os.path.join(args.out, "comparison.png"))

    elif args.batch > 0:
        bbox_map = load_bbox_map(config.BBOX_FILE)
        if bbox_map:
            print(f"Loaded {len(bbox_map)} bbox entries from {config.BBOX_FILE}")
        else:
            print(f"No bbox file found at {config.BBOX_FILE} — drawing heatmaps only")
        # Build a small loader containing only bbox-annotated images
        # (avoids iterating the full 20k dataset which causes OOM)
        from torch.utils.data import DataLoader, Dataset
        FOCUS = {"Atelectasis", "Effusion", "Infiltration", "Nodule"}
        from data.dataset import load_label_map, labels_to_vec
        label_map = load_label_map()
        transform = build_test_transform()

        class BboxSubset(Dataset):
            def __init__(self):
                self.items = []
                for fname in bbox_map:
                    img_path = os.path.join(config.IMAGE_DIR, fname)
                    if not os.path.isfile(img_path):
                        continue
                    labels = label_map.get(fname, ["No Finding"])
                    if len(labels) == 1 and labels[0] in FOCUS:
                        self.items.append((fname, labels[0]))
            def __len__(self): return len(self.items)
            def __getitem__(self, idx):
                fname, _ = self.items[idx]
                img = Image.open(os.path.join(config.IMAGE_DIR, fname)).convert("RGB")
                lbl = torch.from_numpy(labels_to_vec(label_map.get(fname, ["No Finding"])))
                return transform(img), lbl, fname

        bbox_ds = BboxSubset()
        print(f"Focus-class single-label images with bbox: {len(bbox_ds)}")
        bbox_loader = DataLoader(bbox_ds, batch_size=16, shuffle=False, num_workers=2)
        visualize_batch_grid(model, bbox_loader, device, n=args.batch,
                              save_path=os.path.join(args.out, "heatmap_grid.png"),
                              bbox_map=bbox_map)


if __name__ == "__main__":
    main()
