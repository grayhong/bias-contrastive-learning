import torch.nn as nn

from torchvision.models import resnet18


class DIResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes * 2)
        print(f'DIResNet18 - num_classes: {num_classes} pretrained: {pretrained}')

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        return logits
