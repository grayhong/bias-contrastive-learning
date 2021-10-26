import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNetEnD(nn.Module):
    def __init__(self, kernel_size=7, pre_normalize=True, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            # EnD changed the activation to Tanh
            nn.Tanh(),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.dim_in = 128
        self.pre_normalize = pre_normalize

        print(f'SimpleConvNetEndV2: pre_normalize: {pre_normalize} kernel_size {kernel_size}')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_last_shared_layer(self):
        return self.fc

    def forward(self, x):
        x = self.extracter(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if self.pre_normalize:
            feat = F.normalize(feat, dim=1)
            logits = self.fc(feat)
        else:
            # EnD apply FC layer after normalizing features
            logits = self.fc(feat)
            feat = F.normalize(feat, dim=1)

        return logits, feat