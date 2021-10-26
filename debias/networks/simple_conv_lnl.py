import torch
import torch.nn as nn


class LNLSimpleConvNet(nn.Module):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        layer_feat = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layer_class = [
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.feat_extractor = nn.Sequential(*layer_feat)
        self.extractor = nn.Sequential(*layer_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.dim_in = 128

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bias_feat = self.feat_extractor(x)
        x = self.extractor(bias_feat)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.fc(feat)

        return logits, bias_feat


class LNLSimpleBiasPredictor(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        kernel_size = 7
        padding = kernel_size // 2
        layer_class = [
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.extractor = nn.Sequential(*layer_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.dim_in = 128

    def forward(self, x):
        x = self.extractor(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.fc(feat)

        return logits
