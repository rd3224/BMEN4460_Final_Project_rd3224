"""
Thorax-Net (Wang et al., IEEE JBHI 2020)
=========================================

This implementation keeps the original classification branch design and
uses a learnable class-wise spatial attention head.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

import config


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


class ClassificationBranch(nn.Module):
    def __init__(self, backbone: str = "resnet152", num_classes: int = 14,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet152":
            base = tvm.resnet152(
                weights=tvm.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            self.fc = nn.Linear(2048, num_classes)
            self.att_channels = 1024
            self.att_feat_size = 14

        elif backbone == "efficientnet_b4":
            base = tvm.efficientnet_b4(
                weights=tvm.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self._eff_features = base.features
            self.avgpool = base.avgpool
            self.fc = nn.Linear(1792, num_classes)
            self.att_channels = 272
            self.att_feat_size = 14

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        if self.backbone_name == "resnet152":
            return self._forward_resnet(x)
        return self._forward_efficientnet(x)

    def _forward_resnet(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat = self.layer3(x)
        x = self.layer4(feat)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y_cls = torch.sigmoid(self.fc(x))
        return y_cls, feat

    def _forward_efficientnet(self, x):
        feat = None
        for i, block in enumerate(self._eff_features):
            x = block(x)
            if i == 5:
                feat = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y_cls = torch.sigmoid(self.fc(x))
        return y_cls, feat


class AttentionBranch(nn.Module):
    def __init__(self, in_channels: int = 1024, num_classes: int = 14,
                 feat_size: int = 14):
        super().__init__()
        del feat_size
        self.num_classes = num_classes
        mid = in_channels // 4

        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.att_map_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        self.classifier = nn.ModuleList([nn.Linear(in_channels, 1) for _ in range(num_classes)])

    def forward(self, feat: torch.Tensor):
        b, c, h, w = feat.shape
        refined = self.pre_conv(feat)

        att_logits = self.att_map_conv(refined)
        att_flat = att_logits.view(b, self.num_classes, -1)
        att_maps = torch.softmax(att_flat, dim=-1).view(b, self.num_classes, h, w)

        logits = []
        for cls_idx in range(self.num_classes):
            att = att_maps[:, cls_idx:cls_idx + 1]
            pooled = (att * refined).sum(dim=(2, 3))
            logits.append(self.classifier[cls_idx](pooled))

        y_att = torch.sigmoid(torch.cat(logits, dim=1))
        return y_att, att_maps


class ThoraxNet(nn.Module):
    def __init__(self,
                 backbone: str = config.BACKBONE,
                 num_classes: int = config.NUM_CLASSES,
                 pretrained: bool = config.PRETRAINED):
        super().__init__()
        self.cls_branch = ClassificationBranch(backbone, num_classes, pretrained)
        self.att_branch = AttentionBranch(
            in_channels=self.cls_branch.att_channels,
            num_classes=num_classes,
            feat_size=self.cls_branch.att_feat_size,
        )

    def train_stage(self, stage: int):
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

    def forward(self, x):
        y_cls, feat = self.cls_branch(x)
        y_att, att_maps = self.att_branch(feat)
        y_diag = (y_cls + y_att) * 0.5
        return {
            "y_cls": y_cls,
            "y_att": y_att,
            "y_diag": y_diag,
            "att_maps": att_maps,
        }

    def forward_cls_only(self, x):
        y_cls, _ = self.cls_branch(x)
        return y_cls


class WeightedBCELoss(nn.Module):
    def __init__(self, beta_pos: torch.Tensor, beta_neg: torch.Tensor):
        super().__init__()
        self.register_buffer("beta_pos", beta_pos)
        self.register_buffer("beta_neg", beta_neg)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        pos_loss = target * torch.log(pred)
        neg_loss = (1 - target) * torch.log(1 - pred)
        loss = -(self.beta_pos * pos_loss + self.beta_neg * neg_loss)
        return loss.mean()
