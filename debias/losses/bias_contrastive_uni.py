"""
Based on Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity


class DebiasSupConLossUni(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels, biases):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        biases = biases.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T)
        bias_mask = torch.ne(biases, biases.T)
        mask = label_mask & bias_mask
        mask = mask.float().cuda()
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss


class BiasContrastiveLossUni(nn.Module):
    def __init__(self, confusion_matrix, weight=0.01, temperature=0.07, bb=0):
        super().__init__()
        self.temperature = temperature
        self.confusion_matrix = confusion_matrix.cuda()
        self.con_loss = DebiasSupConLossUni(temperature=temperature)
        self.weight = weight
        self.min_prob = 1e-9
        self.bb = bb
        print(f'BiasContrastiveLossUni - bb: {self.bb} T: {self.temperature}')

    def forward(self, logits, labels, biases, features):
        if self.bb:
            prior = self.confusion_matrix[biases]
            logits += torch.log(prior + self.min_prob)
        label_loss = F.cross_entropy(logits, labels)
        bias_loss = self.con_loss(features, labels, biases)
        
        return label_loss, bias_loss
    
