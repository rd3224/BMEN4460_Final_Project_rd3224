"""
Evaluation utilities
- per-class AUC (ROC)
- mean AUC over all 14 classes
- focus-class mean AUC (per reviewer critique: reliable-label subset)
- ROC curve plotting
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import config


def compute_aucs(y_true: np.ndarray, y_score: np.ndarray,
                 classes=None) -> dict:
    """
    Parameters
    ----------
    y_true  : [N, 14]  binary ground truth
    y_score : [N, 14]  predicted probabilities
    classes : list of str (defaults to config.CLASSES)

    Returns
    -------
    dict with keys:
        per_class : {class_name: auc}
        mean_auc  : float
        focus_auc : float  (mean over FOCUS_CLASSES only)
    """
    classes = classes or config.CLASSES
    per_class = {}

    for i, cls in enumerate(classes):
        col_true = y_true[:, i]
        if col_true.sum() == 0 or (1 - col_true).sum() == 0:
            # No positive or no negative examples → AUC undefined
            per_class[cls] = float("nan")
        else:
            per_class[cls] = roc_auc_score(col_true, y_score[:, i])

    valid_aucs = [v for v in per_class.values() if not np.isnan(v)]
    mean_auc   = float(np.mean(valid_aucs))

    focus_aucs = [per_class[c] for c in config.FOCUS_CLASSES
                  if not np.isnan(per_class.get(c, float("nan")))]
    focus_auc  = float(np.mean(focus_aucs)) if focus_aucs else float("nan")

    return {"per_class": per_class, "mean_auc": mean_auc, "focus_auc": focus_auc}


def print_auc_table(auc_dict: dict):
    """Pretty-print per-class AUC table to stdout."""
    print("\n" + "=" * 50)
    print(f"{'Disease':<22} {'AUC':>8}")
    print("-" * 50)
    for cls, auc in auc_dict["per_class"].items():
        marker = " ★" if cls in config.FOCUS_CLASSES else ""
        val = f"{auc:.4f}" if not np.isnan(auc) else "  N/A"
        print(f"{cls:<22} {val:>8}{marker}")
    print("-" * 50)
    print(f"{'Mean AUC (all)':<22} {auc_dict['mean_auc']:>8.4f}")
    print(f"{'Mean AUC (focus ★)':<22} {auc_dict['focus_auc']:>8.4f}")
    print("=" * 50 + "\n")


def plot_roc_curves(y_true: np.ndarray, y_score: np.ndarray,
                    classes=None, save_path: str = "roc_curves.png",
                    title: str = "ROC Curves"):
    """
    Plot ROC curves for all 14 classes on a single figure.
    Mirrors Fig. 5 in the paper.
    """
    classes = classes or config.CLASSES
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.get_cmap("tab20")
    for i, cls in enumerate(classes):
        col_true = y_true[:, i]
        if col_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(col_true, y_score[:, i])
        auc = roc_auc_score(col_true, y_score[:, i])
        ax.plot(fpr, tpr, color=cmap(i), lw=1.5, label=f"{cls} ({auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC curves → {save_path}")


def compare_branches(y_true, y_cls, y_att, y_diag, classes=None):
    """
    Compare per-class AUC of cls branch alone vs att branch alone vs fused.
    Useful for ablation study (Table IV in paper).
    """
    classes = classes or config.CLASSES

    def _mean_auc(y_score):
        aucs = []
        for i in range(len(classes)):
            col = y_true[:, i]
            if col.sum() > 0 and (1 - col).sum() > 0:
                aucs.append(roc_auc_score(col, y_score[:, i]))
        return np.mean(aucs)

    print("\n── Branch Ablation ─────────────────────")
    print(f"  Classification branch only : {_mean_auc(y_cls):.4f}")
    print(f"  Attention branch only      : {_mean_auc(y_att):.4f}")
    print(f"  Fused (y_cls + y_att) / 2  : {_mean_auc(y_diag):.4f}")
    print("────────────────────────────────────────\n")
