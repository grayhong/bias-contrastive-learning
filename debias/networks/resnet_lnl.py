import torch.nn as nn

from torchvision.models import resnet18


class LNLResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        bias_modules = list(model.children())[0:6]
        self.bias_feat_extractor = nn.Sequential(*bias_modules)
        feat_modules = list(model.children())[6:9]
        self.feat_extractor = nn.Sequential(*feat_modules)

        self.embed_size = 512
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)
        print(f'LNLResNet18 - num_classes: {num_classes} pretrained: {pretrained}')

    def forward(self, x):
        bias_feat = self.bias_feat_extractor(x)
        out = self.feat_extractor(bias_feat)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        return logits, bias_feat


class LNLBiasPredictor(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.embed_size = 128
        self.fc = nn.Linear(self.embed_size, num_classes)
        print(f'LNLBiasPredictor - num_classes: {num_classes}')

    def forward(self, feat):
        embed = self.avg_pool(feat)
        embed = embed.squeeze(-1).squeeze(-1)
        logits = self.fc(embed)
        return logits
