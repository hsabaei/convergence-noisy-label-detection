import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        return x


class CNN12(nn.Module):
    """
    12 conv layers total: 3 stages × 4 conv each.
    Stage1: 64 channels (x4 conv)  -> MaxPool
    Stage2: 128 channels (x4 conv) -> MaxPool
    Stage3: 256 channels (x4 conv) -> MaxPool
    Then GAP + linear classifier.
    """

    def __init__(self, num_classes=10, p_drop=0.1):
        super().__init__()
        c1, c2, c3 = 64, 128, 256

        self.s1 = nn.Sequential(
            ConvBlock(3, c1, p_drop),
            ConvBlock(c1, c1, p_drop),
            ConvBlock(c1, c1, p_drop),
            ConvBlock(c1, c1, p_drop),
        )
        self.s2 = nn.Sequential(
            ConvBlock(c1, c2, p_drop),
            ConvBlock(c2, c2, p_drop),
            ConvBlock(c2, c2, p_drop),
            ConvBlock(c2, c2, p_drop),
        )
        self.s3 = nn.Sequential(
            ConvBlock(c2, c3, p_drop),
            ConvBlock(c3, c3, p_drop),
            ConvBlock(c3, c3, p_drop),
            ConvBlock(c3, c3, p_drop),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(c3, num_classes)
        self.penultimate = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.s1(x)
        x = self.pool(x)
        x = self.s2(x)
        x = self.pool(x)
        x = self.s3(x)
        x = self.pool(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        self.penultimate = x
        return self.classifier(x)


def CNN12_Model(num_classes=10):
    return CNN12(num_classes=num_classes)
