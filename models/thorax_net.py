"""
Thorax-Net  (Wang et al., IEEE JBHI 2020)
==========================================
Architecture
------------
  Classification Branch : ResNet-152 (or EfficientNet-B4) pretrained on ImageNet
                          Final FC replaced by 14-neuron sigmoid head.

  Attention Branch      : Takes 14×14 feature maps from penultimate residual
                          module → 3 conv layers (1×1, 3×3, 1×1) → Grad-CAM →
                          3 conv layers (1×1, 1×1, 14×14) → sigmoid → 14-d.

  Diagnosis             : (y_cls + y_att) / 2   (averaged at test time)

Three-stage training (call train_stage(1|2|3)):
  1 – Fine-tune classification branch; attention branch frozen.
  2 – Train attention branch; classification branch frozen.
  3 – End-to-end fine-tune.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import config


# ── Helpers ────────────────────────────────────────────────────────────────────
def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


# ── Classification Branch ──────────────────────────────────────────────────────
class ClassificationBranch(nn.Module):
    """
    ResNet-152 (default) or EfficientNet-B4 with a 14-neuron sigmoid head.
    Exposes `features` hook to capture penultimate residual module output.
    """

    def __init__(self, backbone: str = "resnet152", num_classes: int = 14,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self._hook_feat = None   # will hold intermediate feature map

        if backbone == "resnet152":
            base = tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V1
                                  if pretrained else None)
            # Remove avgpool + fc
            self.stem    = nn.Sequential(base.conv1, base.bn1, base.relu,
                                          base.maxpool)
            self.layer1  = base.layer1   # 256 ch, 56×56
            self.layer2  = base.layer2   # 512 ch, 28×28
            self.layer3  = base.layer3   # 1024 ch, 14×14  ← attention input
            self.layer4  = base.layer4   # 2048 ch, 7×7
            self.avgpool = base.avgpool
            self.fc      = nn.Linear(2048, num_classes)
            self.att_channels   = 1024
            self.att_feat_size  = 14

        elif backbone == "efficientnet_b4":
            base = tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.IMAGENET1K_V1
                                        if pretrained else None)
            # EfficientNet features are in base.features (sequential blocks)
            # Block 5 output ≈ 14×14 × 272 for 224-input
            self._eff_features = base.features
            self.avgpool = base.avgpool
            self.fc      = nn.Linear(1792, num_classes)
            self.att_channels   = 272
            self.att_feat_size  = 14

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        if self.backbone_name == "resnet152":
            return self._forward_resnet(x)
        else:
            return self._forward_efficientnet(x)

    def _forward_resnet(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat = self.layer3(x)           # 1024×14×14 — captured for attention
        x = self.layer4(feat)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logit = self.fc(x)
        y_cls = torch.sigmoid(logit)
        return y_cls, feat              # return both so attention branch can use feat

    def _forward_efficientnet(self, x):
        feat = None
        for i, block in enumerate(self._eff_features):
            x = block(x)
            if i == 5:                  # penultimate block output
                feat = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logit = self.fc(x)
        y_cls = torch.sigmoid(logit)
        return y_cls, feat


# ── Attention Branch ───────────────────────────────────────────────────────────
class AttentionBranch(nn.Module):
    """
    Grad-CAM embedded in stacked convolutions (paper §IV-B).

    pre_conv  : 3 layers (1×1, 3×3, 1×1) — bottleneck to refine feature maps
    Grad-CAM  : computes class-discriminative maps A̅c using gradients of
                y_cls w.r.t. refined features (paper eq. 1-3)
    post_conv : 3 layers (1×1, 1×1, H×H) — maps attention maps → y_att

    NOTE: Grad-CAM requires `feat` to be a leaf that retains grad.
          Call with torch.enable_grad() even at eval time for heatmaps.
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 14,
                 feat_size: int = 14):
        super().__init__()
        self.num_classes = num_classes
        self.feat_size   = feat_size
        mid = in_channels // 4

        # Pre-Grad-CAM convs (bottleneck)
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )

        # Post-Grad-CAM convs  (input: num_classes attention maps)
        self.post_conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, feat_size),     # collapses H×W → 1×1
        )

    def forward(self, feat: torch.Tensor, y_cls: torch.Tensor):
        """
        feat  : [B, C, H, W]  penultimate feature map (requires_grad must be True)
        y_cls : [B, 14]       output of classification branch
        Returns (y_att [B,14], att_maps [B,14,H,W])
        """
        B, C, H, W = feat.shape

        # ── Step 1: refine feature maps ───────────────────────────────────────
        refined = self.pre_conv(feat)   # [B, C, H, W]

        # ── Step 2: Grad-CAM  (eq. 1-3) ──────────────────────────────────────
        att_maps = self._gradcam(y_cls, feat, refined, B, H, W)   # [B, 14, H, W]

        # ── Step 3: post conv → y_att ─────────────────────────────────────────
        out = self.post_conv(att_maps)          # [B, 14, 1, 1]
        y_att = torch.sigmoid(out.flatten(1))  # [B, 14]

        return y_att, att_maps

    def _gradcam(self, y_cls, feat, refined, B, H, W):
        """
        Vectorised Grad-CAM over all 14 classes.
        α_ck = mean_ij( ∂y_c / ∂feat_k(i,j) )  (paper eq. 2)
        Ā_c  = ReLU( Σ_k  α_ck · Ã_k )          (paper eq. 1, weighted on refined)
        a_cij = softmax over spatial              (paper eq. 3)

        Gradients are taken w.r.t. feat (connected to y_cls via cls_branch),
        then used to weight refined (pre_conv output) for richer attention maps.
        """
        att_maps = []
        for c in range(self.num_classes):
            grad = torch.autograd.grad(
                y_cls[:, c].sum(),
                feat,
                retain_graph=True,
                create_graph=False,
            )[0]                              # [B, C, H, W]

            # α_ck = GAP of gradient  [B, C, 1, 1]
            alpha = grad.mean(dim=[2, 3], keepdim=True)

            # Class-discriminative map weighted on refined features  [B, 1, H, W]
            # No ReLU here: att_branch is randomly initialised so pre-ReLU values
            # are mostly negative, which would zero out all gradients to pre_conv.
            # Softmax handles normalisation without needing positive-only inputs.
            cam = (alpha * refined).sum(dim=1, keepdim=True)

            # Spatial softmax normalisation  (eq. 3)
            cam_flat = cam.view(B, -1)                           # [B, H*W]
            # Standardise before softmax: alpha from frozen cls_branch is ~1e-5,
            # making raw cam values near-zero and softmax near-uniform.
            # Dividing by std (detached) amplifies gradients back to pre_conv
            # without changing the attention map semantics.
            cam_std  = cam_flat.std(dim=1, keepdim=True).detach().clamp(min=1e-8)
            cam_norm = torch.softmax(cam_flat / cam_std, dim=1)  # [B, H*W]
            att_maps.append(cam_norm.view(B, 1, H, W))

        return torch.cat(att_maps, dim=1)   # [B, 14, H, W]


# ── Full Thorax-Net ─────────────────────────────────────────────────────────────
class ThoraxNet(nn.Module):
    """
    Full model combining classification + attention branches.

    Forward returns a dict:
        y_cls   : [B, 14]  classification branch output
        y_att   : [B, 14]  attention branch output
        y_diag  : [B, 14]  (y_cls + y_att) / 2  — final diagnosis
        att_maps: [B, 14, H, W]  heatmaps (Grad-CAM)
    """

    def __init__(self,
                 backbone:    str  = config.BACKBONE,
                 num_classes: int  = config.NUM_CLASSES,
                 pretrained:  bool = config.PRETRAINED):
        super().__init__()
        self.cls_branch = ClassificationBranch(backbone, num_classes, pretrained)
        self.att_branch = AttentionBranch(
            in_channels = self.cls_branch.att_channels,
            num_classes = num_classes,
            feat_size   = self.cls_branch.att_feat_size,
        )

    # ── Stage control ─────────────────────────────────────────────────────────
    def train_stage(self, stage: int):
        """Switch between training stages (1, 2, 3)."""
        assert stage in (1, 2, 3)
        if stage == 1:
            _unfreeze(self.cls_branch)
            _freeze(self.att_branch)
        elif stage == 2:
            _freeze(self.cls_branch)
            _unfreeze(self.att_branch)
        else:
            _unfreeze(self.cls_branch)
            _unfreeze(self.att_branch)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        # Ensure intermediate features track gradients for Grad-CAM
        x.requires_grad_(True)

        y_cls, feat = self.cls_branch(x)

        y_att, att_maps = self.att_branch(feat, y_cls)

        y_diag = (y_cls + y_att) * 0.5

        return {
            "y_cls":    y_cls,
            "y_att":    y_att,
            "y_diag":   y_diag,
            "att_maps": att_maps,
        }

    # ── Convenience: cls-only forward (stage 1) ───────────────────────────────
    def forward_cls_only(self, x):
        """Only runs classification branch; cheaper for stage-1 training."""
        y_cls, _ = self.cls_branch(x)
        return y_cls


# ── Loss ───────────────────────────────────────────────────────────────────────
class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross-entropy  (paper eq. 4).
    β_pos, β_neg: per-class tensors [14], computed from dataset statistics.
    """

    def __init__(self, beta_pos: torch.Tensor, beta_neg: torch.Tensor):
        super().__init__()
        self.register_buffer("beta_pos", beta_pos)
        self.register_buffer("beta_neg", beta_neg)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, 14]  (pred is sigmoid output in [0,1])"""
        eps = 1e-7
        pred  = pred.clamp(eps, 1 - eps)
        pos_loss = target       * torch.log(pred)
        neg_loss = (1 - target) * torch.log(1 - pred)
        loss = -(self.beta_pos * pos_loss + self.beta_neg * neg_loss)
        return loss.mean()
