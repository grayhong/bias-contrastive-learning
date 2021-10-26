import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from debias.datasets.biased_mnist import get_color_mnist
from debias.losses.gce import GeneralizedCELoss
from debias.networks.simple_conv import SimpleConvNet
from debias.utils.logging import set_logging
from debias.utils.training import EMA
from debias.utils.utils import AverageMeter, accuracy, save_model, set_seed, pretty_dict


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', )
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--print_freq', type=int, default=300,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--corr', type=float, default=0.999)

    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--cbs', type=int, default=64, help='batch_size of dataloader for contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-3)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def set_model():
    net_b = SimpleConvNet().cuda()
    net_d = SimpleConvNet().cuda()

    return net_b, net_d


def train(train_loader, net_b, net_d, sample_loss_ema_b, sample_loss_ema_d, opt_b, opt_d):
    net_b.train()
    net_d.train()

    avg_loss = AverageMeter()

    train_iter = iter(train_loader)

    criterion = nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss()
    for idx, (images, labels, biases, indices) in enumerate(train_iter):
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        logit_b, _ = net_b(images)
        logit_d, _ = net_d(images)

        loss_b = criterion(logit_b, labels).cpu().detach()
        loss_d = criterion(logit_d, labels).cpu().detach()

        # EMA sample loss
        sample_loss_ema_b.update(loss_b, indices)
        sample_loss_ema_d.update(loss_d, indices)

        # class-wise normalize
        loss_b = sample_loss_ema_b.parameter[indices].clone().detach()
        loss_d = sample_loss_ema_d.parameter[indices].clone().detach()

        label_cpu = labels.cpu()

        for c in range(10):
            class_index = np.where(label_cpu == c)[0]
            max_loss_b = sample_loss_ema_b.max_loss(c)
            max_loss_d = sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d

        # re-weighting based on loss value / generalized CE for biased model
        loss_weight = loss_b / (loss_b + loss_d + 1e-8)
        if np.isnan(loss_weight.mean().item()):
            raise NameError('loss_weight')

        loss_b_update = bias_criterion(logit_b, labels)

        if np.isnan(loss_b_update.mean().item()):
            raise NameError('loss_b_update')
        loss_d_update = criterion(logit_d, labels) * loss_weight.cuda()
        if np.isnan(loss_d_update.mean().item()):
            raise NameError('loss_d_update')
        loss = loss_b_update.mean() + loss_d_update.mean()

        opt_b.zero_grad()
        opt_d.zero_grad()
        loss.backward()
        opt_b.step()
        opt_d.step()

        avg_loss.update(loss.item(), bsz)

    return avg_loss.avg


def validate(val_loader, model):
    model.eval()

    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels, _, _) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            output, _ = model(images)

            acc1, = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0].item(), bsz)

    return top1.avg


def save_model(net_b, net_d, optim_b, optim_d, opt, epoch, save_file):
    state = {
        'opt': opt,
        'net_b': net_b.state_dict(),
        'net_d': net_d.state_dict(),
        'optim_b': optim_b.state_dict(),
        'optim_d': optim_d.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def main():
    opt = parse_option()

    exp_name = f'lff-color_mnist_corr{opt.corr}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}'
    opt.exp_name = exp_name

    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    set_seed(opt.seed)
    logging.info(f'save_path: {save_path}')

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = './data/biased_mnist'
    train_loader = get_color_mnist(
        root,
        batch_size=opt.bs,
        data_label_correlation=opt.corr,
        n_confusing_labels=9,
        split='train',
        seed=opt.seed,
        aug=False, )
    logging.info(
        f'confusion_matrix - \n original: {train_loader.dataset.confusion_matrix_org}, \n normalized: {train_loader.dataset.confusion_matrix}')

    val_loaders = {}
    val_loaders['test'] = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation=0.1,
        n_confusing_labels=9,
        split='valid',
        seed=opt.seed,
        aug=False)

    net_b, net_d = set_model()

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]
    opt_b = torch.optim.Adam(net_b.parameters(), lr=opt.lr, weight_decay=1e-4)
    sched_b = torch.optim.lr_scheduler.MultiStepLR(opt_b, milestones=decay_epochs, gamma=0.1)

    opt_d = torch.optim.Adam(net_d.parameters(), lr=opt.lr, weight_decay=1e-4)
    sched_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=decay_epochs, gamma=0.1)

    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    ema_b = EMA(torch.LongTensor(train_loader.dataset.targets), alpha=0.7)
    ema_d = EMA(torch.LongTensor(train_loader.dataset.targets), alpha=0.7)

    best_acc = 0
    best_epoch = 0
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {sched_b.get_last_lr()[0]}')
        loss = train(train_loader, net_b, net_d, ema_b, ema_d, opt_b, opt_d)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss}')
        
        sched_b.step()
        sched_d.step()

        stats = {}
        for key, val_loader in val_loaders.items():
            val_acc = validate(val_loader, net_d)
            stats[f'acc_{key}'] = val_acc

        if stats[f'acc_test'] > best_acc:
            best_acc = stats[f'acc_test']
            best_epoch = epoch

            save_file = save_path / 'checkpoints' / f'best.pth'
            save_model(net_b, net_d, opt_b, opt_d, opt, epoch, save_file)

        logging.info(f'[{epoch} / {opt.epochs}] current acc: {val_acc}, best acc: {best_acc} at {best_epoch}')

        if epoch % opt.save_freq == 0:
            save_file = save_path / 'checkpoints' / f'ckpt_epoch_{epoch}.pth'
            save_model(net_b, net_d, opt_b, opt_d, opt, epoch, save_file)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')

    save_file = save_path / 'checkpoints' / f'last.pth'
    save_model(net_b, net_d, opt_b, opt_d, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
