# Unbiased Classification Through Bias-Contrastive and Bias-Balanced Learning (NeurIPS 2021)

Official Pytorch implementation of Unbiased Classification Through Bias-Contrastive and Bias-Balanced Learning (NeurIPS 2021)

## Setup
This setting requires CUDA 11.
However, you can still use your own environment by installing requirements including PyTorch and Torchvision.

1. Install conda environment and activate it  
```
conda env create -f environment.yml
conda activate biascon
```

2. Prepare dataset.

- Biased MNIST  
By default, we set `download=True` for convenience.  
Thus, you only have to make the empty dataset directory with `mkdir -p data/biased_mnist` and run the code.

- CelebA  
Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset under `data/celeba`

- UTKFace  
Download [UTKFace](https://susanqq.github.io/UTKFace/) dataset under `data/utk_face`

- ImageNet & ImageNet-A  
We use ILSVRC 2015 ImageNet dataset.  
Download [ImageNet](https://www.image-net.org/) under `./data/imagenet` and [ImageNet-A](https://github.com/hendrycks/natural-adv-examples) under `./data/imagenet-a`



## Biased MNIST (w/ bias labels)
We use correlation {0.999, 0.997, 0.995, 0.99, 0.95, 0.9}.

### Bias-contrastive loss (BiasCon)
```
python train_biased_mnist_bc.py --corr 0.999 --seed 1
```

### Bias-balancing loss (BiasBal)
```
python train_biased_mnist_bb.py --corr 0.999 --seed 1
```

### Joint use of BiasCon and BiasBal losses (BC+BB)
```
python train_biased_mnist_bc.py --bb 1 --corr 0.999 --seed 1
```

## CelebA
We assess CelebA dataset with target attributes of **HeavyMakeup** (`--task makeup`) and **Blonde** (`--task blonde`).  

### Bias-contrastive loss (BiasCon)
```
python train_celeba_bc.py --task makeup --seed 1
```

### Bias-balancing loss (BiasBal)
```
python train_celeba_bb.py --task makeup --seed 1
```

### Joint use of BiasCon and BiasBal losses (BC+BB)
```
python train_celeba_bc.py --bb 1 --task makeup --seed 1
```

## UTKFace
We assess UTKFace dataset biased toward **Race** (`--task race`) and **Age** (`--task age`) attributes.  

### Bias-contrastive loss (BiasCon)
```
python train_utk_face_bc.py --task race --seed 1
```

### Bias-balancing loss (BiasBal)
```
python train_utk_face_bb.py --task race --seed 1
```

### Joint use of BiasCon and BiasBal losses (BC+BB)
```
python train_utk_face_bc.py --bb 1 --task race --seed 1
```

## Biased MNIST (w/o bias labels)
We use correlation {0.999, 0.997, 0.995, 0.99, 0.95, 0.9}.

### Soft Bias-contrastive loss (SoftCon)

1. Train a bias-capturing model and get bias features.
```
python get_biased_mnist_bias_features.py --corr 0.999 --seed 1
```

2. Train a model with bias features.
```
python train_biased_mnist_softcon.py --corr 0.999 --seed 1
```

## ImageNet
We use texture cluster information from [ReBias (Bahng et al., 2020)](https://github.com/clovaai/rebias).

### Soft Bias-contrastive loss (SoftCon)

1. Train a bias-capturing model and get bias features.
```
python get_imagenet_bias_features.py --seed 1
```

2. Train a model with bias features.
```
python train_imagenet_softcon.py --seed 1
```
