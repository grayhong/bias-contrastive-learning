"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python implementation of Biased-MNIST.
"""
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from debias.datasets.utils import TwoCropTransform, get_confusion_matrix, get_unsup_confusion_matrix
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST


class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two paramters to control the level of bias.

    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.

        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).

        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.

    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]

    def __init__(self, root, bias_feature_root='./biased_feats', split='train', transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, seed=1, load_bias_feature=False, train_corr=None):
        assert split in ['train', 'valid']
        train = split in ['train']
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.load_bias_feature = load_bias_feature
        if self.load_bias_feature:
            if train_corr:
                bias_feature_dir = f'{bias_feature_root}/train{train_corr}-corr{data_label_correlation}-seed{seed}'
                logging.info(f'load bias feature: {bias_feature_dir}')
                self.bias_features = torch.load(f'{bias_feature_dir}/bias_feats.pt')
                self.marginal = torch.load(f'{bias_feature_dir}/marginal.pt')
            else:
                bias_feature_dir = f'{bias_feature_root}/color_mnist-corr{data_label_correlation}-seed{seed}'
                logging.info(f'load bias feature: {bias_feature_dir}')
                self.bias_features = torch.load(f'{bias_feature_dir}/bias_feats.pt')
                self.marginal = torch.load(f'{bias_feature_dir}/marginal.pt')

        save_path = Path(root) / 'pickles' / f'color_mnist-corr{data_label_correlation}-seed{seed}' / split
        if save_path.is_dir():
            logging.info(f'use existing color_mnist from {save_path}')
            self.data = pickle.load(open(save_path / 'data.pkl', 'rb'))
            self.targets = pickle.load(open(save_path / 'targets.pkl', 'rb'))
            self.biased_targets = pickle.load(open(save_path / 'biased_targets.pkl', 'rb'))
        else:
            self.random = True

            self.data_label_correlation = data_label_correlation
            self.n_confusing_labels = n_confusing_labels
            self.data, self.targets, self.biased_targets = self.build_biased_mnist()

            indices = np.arange(len(self.data))
            self._shuffle(indices)

            self.data = self.data[indices].numpy()
            self.targets = self.targets[indices]
            self.biased_targets = self.biased_targets[indices]

            logging.info(f'save color_mnist to {save_path}')
            save_path.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.data, open(save_path / 'data.pkl', 'wb'))
            pickle.dump(self.targets, open(save_path / 'targets.pkl', 'wb'))
            pickle.dump(self.biased_targets, open(save_path / 'biased_targets.pkl', 'wb'))

        if load_bias_feature:
            self.confusion_matrix_org, self.confusion_matrix = get_unsup_confusion_matrix(num_classes=10,
                                                                                          targets=self.targets,
                                                                                          biases=self.biased_targets,
                                                                                          marginals=self.marginal)
        else:
            self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(
                num_classes=10,
                targets=self.targets,
                biases=self.biased_targets)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target, bias = self.data[index], int(self.targets[index]), int(self.biased_targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.load_bias_feature:
            bias_feat = self.bias_features[index]
            return img, target, bias, index, bias_feat
        else:
            return img, target, bias, index


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(self, root, bias_feature_root='./biased_feats', split='train', transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, seed=1, load_bias_feature=False, train_corr=None):
        super(ColourBiasedMNIST, self).__init__(root, bias_feature_root=bias_feature_root, split=split, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels, seed=seed,
                                                load_bias_feature=load_bias_feature, train_corr=train_corr)

    def _binary_to_colour(self, data, colour):
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label):
        return self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]


def get_color_mnist(root, batch_size, data_label_correlation,
                    n_confusing_labels=9, split='train', num_workers=8, seed=1, aug=True,
                    two_crop=False, ratio=0, bias_feature_root='./biased_feats', load_bias_feature=False, given_y=True, train_corr=None):
    logging.info(f'get_color_mnist - split: {split}, aug: {aug}, given_y: {given_y}, ratio: {ratio}')
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if aug:
        train_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    if two_crop:
        train_transform = TwoCropTransform(train_transform)

    if split == 'train_val':
        dataset = ColourBiasedMNIST(
            root, split='train', transform=train_transform,
            download=True, data_label_correlation=data_label_correlation,
            n_confusing_labels=n_confusing_labels, seed=seed,
            load_bias_feature=load_bias_feature,
            train_corr=train_corr
        )

        indices = list(range(len(dataset)))
        split = int(np.floor(0.1 * len(dataset)))
        np.random.shuffle(indices)
        valid_idx = indices[:split]
        valid_sampler = data.sampler.SubsetRandomSampler(valid_idx)

        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False)

        return dataloader

    else:
        dataset = ColourBiasedMNIST(
            root, bias_feature_root=bias_feature_root, split=split, transform=train_transform,
            download=True, data_label_correlation=data_label_correlation,
            n_confusing_labels=n_confusing_labels, seed=seed,
            load_bias_feature=load_bias_feature,
            train_corr=train_corr
        )

        def clip_max_ratio(score):
            upper_bd = score.min() * ratio
            return np.clip(score, None, upper_bd)

        if ratio != 0:
            if load_bias_feature:
                weights = dataset.marginal
            else:
                if given_y:
                    weights = [1 / dataset.confusion_matrix_by[c, b] for c, b in zip(dataset.targets, dataset.biased_targets)]
                else:
                    weights = [1 / dataset.confusion_matrix[b, c] for c, b in zip(dataset.targets, dataset.biased_targets)]

            if ratio > 0:
                weights = clip_max_ratio(np.array(weights))
            sampler = data.WeightedRandomSampler(weights, len(weights), replacement=True)
        else:
            sampler = None

        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True if sampler is None and split == 'train' else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True, 
            drop_last=split == 'train')

        return dataloader
