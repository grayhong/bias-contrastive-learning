import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from debias.datasets.imagenet import get_imagenet
from debias.networks.imagenet_models import bagnet18
from debias.utils.utils import AverageMeter, accuracy, set_seed


def train_biased_model(g_net, tr_loader, n_epochs=120):
    g_opt = torch.optim.Adam(g_net.parameters(), lr=1e-3)
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        g_opt, n_epochs
    )

    print(f'train_biased_model - opt: {g_opt}, sched: {g_scheduler}')

    g_net.train()
    top1 = AverageMeter()
    bias_top1 = AverageMeter()
    for n in range(n_epochs):
        tr_iter = iter(tr_loader)
        for x, y, bias, _, _ in tr_iter:
            x, y, bias = x.cuda(), y.cuda(), bias.cuda()
            N = x.size(0)
            pred, _ = g_net(x)
            loss = F.cross_entropy(pred, y)
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()

            prec1, = accuracy(pred, y, topk=(1,))
            bias_prec1, = accuracy(pred, bias, topk=(1,))
            top1.update(prec1.item(), N)
            bias_top1.update(bias_prec1.item(), N)
        g_scheduler.step()
        print(f'Training biased model - Epoch: {n} acc: {top1.avg}, bias acc: {bias_top1.avg}')
    print(f'Training biased model done - final acc: {top1.avg}, bias acc: {bias_top1.avg}')
    return g_net


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--ckpt', action='store_true')

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def get_features(model, dataloader):
    model.eval()
    with torch.no_grad():
        data_iter = iter(dataloader)
        num_data = len(dataloader.dataset)
        all_feats = torch.zeros(num_data, model.dim_in)
        for img, _, _, _, idx in data_iter:
            all_feats[idx] = model(img.cuda())[1].cpu()
    return all_feats


def get_marginal(feats, targets, num_classes):
    N_total = feats.shape[0]
    marginal = torch.zeros(N_total)
    for n in range(num_classes):
        target_feats = feats[targets == n]
        N = target_feats.shape[0]
        N_ref = 1024
        ref_idx = np.random.choice(N, N_ref, replace=False)
        ref_feats = target_feats[ref_idx]
        mask = 1 - cosine_similarity(target_feats, ref_feats.cpu().numpy())
        marginal[targets == n] = torch.from_numpy(mask).sum(1)
    return marginal


def main():
    opt = parse_option()
    set_seed(opt.seed)

    root = './data/imagenet'
    train_loader = get_imagenet(
        f'{root}/train',
        batch_size=128,
        train=True,
        aug=False, )

    model = bagnet18(num_classes=9).cuda()
    model.cuda()
    model = train_biased_model(model, train_loader)

    all_feats = get_features(model, train_loader)
    targets = torch.tensor([t for _, t in train_loader.dataset.dataset])
    marginal = get_marginal(all_feats, targets, 9)
    save_path = Path(f'imagenet_biased_feats/imagenet-seed{opt.seed}')
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(all_feats, save_path / 'bias_feats.pt')
    print(f"Saved feats at {save_path / 'bias_feats.pt'}")
    torch.save(marginal, save_path / 'marginal.pt')
    print(f"Saved marginal at {save_path / 'marginal.pt'}")


if __name__ == '__main__':
    main()
