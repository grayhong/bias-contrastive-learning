import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class pattern_norm(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(pattern_norm, self).__init__()
        self.scale = scale

    def forward(self, input):
        sizes = input.size()
        if len(sizes) > 2:
            input = input.view(-1, np.prod(sizes[1:]))
            input = torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12)
            input = input.view(sizes)
        return input


class Hook():
    def __init__(self, module, backward=False):
        self.module = module
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


# For each discriminatory class, orthogonalize samples
def abs_orthogonal_blind(output, gram, target_labels, bias_labels):
    bias_classes = torch.unique(bias_labels)
    orthogonal_loss = torch.tensor(0.).to(output.device)
    M_tot = 0.

    for bias_class in bias_classes:
        bias_mask = (bias_labels == bias_class).type(torch.float).unsqueeze(dim=1)
        bias_mask = torch.tril(torch.mm(bias_mask, torch.transpose(bias_mask, 0, 1)), diagonal=-1)
        M = bias_mask.sum()
        M_tot += M

        if M > 0:
            orthogonal_loss += torch.abs(torch.sum(gram * bias_mask))

    if M_tot > 0:
        orthogonal_loss /= M_tot
    return orthogonal_loss


# For each target class, parallelize samples belonging to
# different discriminatory classes
def abs_parallel(gram, target_labels, bias_labels):
    target_classes = torch.unique(target_labels)
    bias_classes = torch.unique(bias_labels)
    parallel_loss = torch.tensor(0.).to(gram.device)
    M_tot = 0.

    for target_class in target_classes:
        class_mask = (target_labels == target_class).type(torch.float).unsqueeze(dim=1)

        for idx, bias_class in enumerate(bias_classes):
            bias_mask = (bias_labels == bias_class).type(torch.float).unsqueeze(dim=1)

            for other_bias_class in bias_classes[idx:]:
                if other_bias_class == bias_class:
                    continue

                other_bias_mask = (bias_labels == other_bias_class).type(torch.float).unsqueeze(dim=1)
                mask = torch.tril(torch.mm(class_mask * bias_mask, torch.transpose(class_mask * other_bias_mask, 0, 1)), diagonal=-1)
                M = mask.sum()
                M_tot += M

                if M > 0:
                    parallel_loss -= torch.sum((1.0 + gram) * mask * 0.5)

    if M_tot > 0:
        parallel_loss = 1.0 + (parallel_loss / M_tot)

    return parallel_loss


def abs_regu(hook, target_labels, bias_labels, alpha=1.0, beta=1.0, sum=True):
    D = hook.output
    if len(D.size()) > 2:
        D = D.view(-1, np.prod((D.size()[1:])))

    gram_matrix = torch.tril(torch.mm(D, torch.transpose(D, 0, 1)), diagonal=-1)
    # not really needed, just for safety for approximate repr
    gram_matrix = torch.clamp(gram_matrix, -1, 1.)

    zero = torch.tensor(0.).to(target_labels.device)
    R_ortho = abs_orthogonal_blind(D, gram_matrix, target_labels, bias_labels) if alpha != 0 else zero
    R_parallel = abs_parallel(gram_matrix, target_labels, bias_labels) if beta != 0 else zero

    if sum:
        return alpha * R_ortho + beta * R_parallel
    return alpha * R_ortho, beta * R_parallel


class EnDLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.min_prob = 1e-9

        print(f'EnDLoss - alpha: {alpha} beta: {beta}')

    def forward(self, logits, labels, biases, feats):
        label_loss = F.cross_entropy(logits, labels)

        gram_matrix = torch.tril(torch.mm(feats, torch.transpose(feats, 0, 1)), diagonal=-1)
        # not really needed, just for safety for approximate repr
        gram_matrix = torch.clamp(gram_matrix, -1, 1.)

        R_ortho = abs_orthogonal_blind(feats, gram_matrix, labels, biases)
        R_parallel = abs_parallel(gram_matrix, labels, biases)

        bias_loss = self.alpha * R_ortho + self.beta * R_parallel

        return label_loss, bias_loss
