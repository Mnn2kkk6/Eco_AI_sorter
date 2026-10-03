import torch.nn as nn
from torchvision import models


class MultiTaskNet(nn.Module):
    """Backbone EfficientNet-B0 dùng chung + 2 head: nhị phân (O/R) và 6 vật liệu."""

    def __init__(self, n_bin: int = 2, n_mat: int = 6, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        b = models.efficientnet_b0(weights=w)
        self.features, self.pool = b.features, b.avgpool
        d = b.classifier[1].in_features  # 1280
        self.head_bin = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_bin))
        self.head_mat = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_mat))

    def forward(self, x):
        z = self.pool(self.features(x)).flatten(1)
        return self.head_bin(z), self.head_mat(z)

    def backbone_params(self):
        return list(self.features.parameters())

    def head_params(self):
        return list(self.head_bin.parameters()) + list(self.head_mat.parameters())
