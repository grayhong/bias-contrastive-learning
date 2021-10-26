import torch
import torch.nn as nn
import torch.nn.functional as F


class DILoss(nn.Module):
    def __init__(self, num_classes=2, num_biases=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_biases = num_biases
        print(f'DILoss')

    def forward(self, logits, labels, biases):
        logprobs = [F.log_softmax(logits[:, i * self.num_classes: (i + 1) * self.num_classes], dim=1) for i in
                    range(self.num_biases)]
        output = torch.cat(logprobs, dim=1)
        target = biases * self.num_classes + labels
        return F.nll_loss(output, target)
