# Cert-SSB: Toward Certified  Sample-Specific  Backdoor Defense

This is the official implementation of our paper Cert-SSB: Toward Certified  Sample-Specific  Backdoor Defense. This research project is developed based on Python 3.8 and Pytorch, created by [Ting Qiao](https://github.com/NcepuQiaoTing) and [Yiming Li](https://liyiming.tech/).

Pipeline
-

Reproducibilty Statement
-
We hereby only release the checkpoints and inference codes for reproducing our main results. We will release full codes (including the training process) of our methods upon the acceptance of this paper.

Requirements
-
To install requirements：

```
pip install -r requirements.txt
```

Make sure the directory follows:


Dataset Preparation
-
Make sure the directory `data` follows:

📋 Data Download Link:

[MNIST]()

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

[ImageNette]()


Model Preparation
-
Make sure the directory `model` follows:

📋 Model Download Link:

[model]()

Training  Model
-
To train the  model in the paper, run these commanding:

GTSRB:

```
python train.py --dataset gtsrb
```

CIFAR-10:

```
python train.py --dataset cifar10
```






