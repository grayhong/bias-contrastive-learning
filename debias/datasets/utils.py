import torch


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_confusion_matrix(num_classes, targets, biases):
    confusion_matrix_org = torch.zeros(num_classes, num_classes)
    confusion_matrix_org_by = torch.zeros(num_classes, num_classes)
    for t, p in zip(targets, biases):
        confusion_matrix_org[p.long(), t.long()] += 1
        confusion_matrix_org_by[t.long(), p.long()] += 1

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    confusion_matrix_by = confusion_matrix_org_by / confusion_matrix_org_by.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix, confusion_matrix_by


def get_unsup_confusion_matrix(num_classes, targets, biases, marginals):
    confusion_matrix_org = torch.zeros(num_classes, num_classes).float()
    confusion_matrix_cnt = torch.zeros(num_classes, num_classes).float()
    for t, p, m in zip(targets, biases, marginals):
        confusion_matrix_org[p.long(), t.long()] += m
        confusion_matrix_cnt[p.long(), t.long()] += 1

    zero_idx = confusion_matrix_org == 0
    confusion_matrix_cnt[confusion_matrix_cnt == 0] = 1
    confusion_matrix_org = confusion_matrix_org / confusion_matrix_cnt
    confusion_matrix_org[zero_idx] = 1
    confusion_matrix_org = 1 / confusion_matrix_org
    confusion_matrix_org[zero_idx] = 0

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix
