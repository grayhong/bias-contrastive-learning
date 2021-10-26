import numpy as np
import torch


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1


def grad_reverse(x):
    return GradReverse.apply(x)


class EMA:
    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))

    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()