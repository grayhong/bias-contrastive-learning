"""
Based on Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity


class DebiasSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, biases=None, mask=None):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        if mask is None:
            assert biases is not None
            biases = biases.contiguous().view(-1, 1)
            label_mask = torch.eq(labels, labels.T)
            bias_mask = torch.ne(biases, biases.T)
            mask = label_mask & bias_mask
            mask = mask.float().cuda()
        else:
            label_mask = torch.eq(labels, labels.T).float().cuda()
            mask = label_mask * mask

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
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

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.loss = DebiasSupConLoss(temperature=temperature)
        print(f'ContrastiveLoss - T: {self.temperature}')

    def forward(self, cont_features, cont_labels, cont_biases):
        return self.loss(cont_features, cont_labels, biases=cont_biases)

class BiasContrastiveLoss(nn.Module):
    def __init__(self, confusion_matrix, temperature=0.07, bb=0):
        super().__init__()
        self.temperature = temperature
        self.con_loss = DebiasSupConLoss(temperature=temperature)
        self.confusion_matrix = confusion_matrix.cuda()
        self.min_prob = 1e-9
        self.bb = bb
        print(f'BiasContrastiveLoss - bb: {self.bb} T: {self.temperature}')

    def forward(self, logits, labels, biases, cont_features, cont_labels, cont_biases):
        if self.bb:
            prior = self.confusion_matrix[biases]
            logits += torch.log(prior + self.min_prob)
        ce_loss = F.cross_entropy(logits, labels)
        bc_loss = self.con_loss(cont_features, cont_labels, biases=cont_biases)
        
        return ce_loss, bc_loss


class UnsupBiasContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.con_loss = DebiasSupConLoss(temperature=temperature)
        print(f'UnsupBiasContrastiveLoss - T: {self.temperature}')

    def cosine_pairwise(self, x):
        x = x.permute((1, 0))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-1)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise

    def forward(self, logits, labels, cont_features, cont_labels, cont_bias_feats):
        ce_loss = F.cross_entropy(logits, labels)
        # cont_bias_feats = F.normalize(cont_bias_feats, dim=1)
        mask = 1 - cosine_similarity(cont_bias_feats.cpu().numpy())
        mask = torch.from_numpy(mask).cuda()
        con_loss = self.con_loss(cont_features, cont_labels, mask=mask)
        return ce_loss, con_loss
    
