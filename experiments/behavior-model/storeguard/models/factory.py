"""행동 인식 백본 팩토리.

후보 비교 (GTX 1050 Ti / VRAM 4GB 기준, docs/architecture.md 에 근거 정리)

| arch          | 사전학습      | 입력            | 파라미터 | 4GB 적합성 |
|---------------|--------------|-----------------|---------|-----------|
| r2plus10_18   | Kinetics-400 | 16x112x112      | 33.4M   | 기준 모델로 채택 |
| mc3_18        | Kinetics-400 | 16x112x112      | 11.7M   | 더 가벼움. 대안 |
| r3d_18        | Kinetics-400 | 16x112x112      | 33.4M   | 성능 낮음 |
| s3d           | Kinetics-400 | 64x224x224 권장  | 8.3M    | 224 입력이라 4GB 에서 배치가 1~2로 떨어짐 |

r2plus1d_18 을 기준 모델로 택한 이유
  1) (2+1)D 분해 덕분에 순수 3D conv 보다 같은 연산량에서 정확도가 높다.
  2) torchvision 에 Kinetics-400 사전학습 가중치가 들어 있어 전이학습이 바로 된다.
  3) 학습 해상도 112 가 4GB VRAM 에 맞는다.
  VRAM 이 부족하면 mc3_18 → batch_size 축소 → accum_steps 증가 순으로 대응한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_ARCHS = {
    "r2plus1d_18": ("r2plus1d_18", "R2Plus1D_18_Weights"),
    "mc3_18": ("mc3_18", "MC3_18_Weights"),
    "r3d_18": ("r3d_18", "R3D_18_Weights"),
    "s3d": ("s3d", "S3D_Weights"),
}


def build_model(arch: str, num_classes: int, *, pretrained: bool = True,
                dropout: float = 0.5) -> nn.Module:
    if arch not in _ARCHS:
        raise ValueError(f"지원하지 않는 arch: {arch}. 가능: {sorted(_ARCHS)}")
    from torchvision.models import video as tvv

    fn_name, weights_name = _ARCHS[arch]
    fn = getattr(tvv, fn_name)
    weights = None
    if pretrained:
        weights = getattr(tvv, weights_name).KINETICS400_V1
    model = fn(weights=weights)

    if arch == "s3d":
        in_ch = model.classifier[1].in_channels
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv3d(in_ch, num_classes, kernel_size=1, stride=1, bias=True),
        )
    else:
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return model


def param_groups(model: nn.Module, lr: float, backbone_mult: float, weight_decay: float):
    """분류 헤드는 lr, 백본은 lr*backbone_mult."""
    head_names = ("fc", "classifier")
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.split(".")[0] in head_names else backbone).append(p)
    return [
        {"params": backbone, "lr": lr * backbone_mult, "weight_decay": weight_decay},
        {"params": head, "lr": lr, "weight_decay": weight_decay},
    ]


@torch.no_grad()
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
