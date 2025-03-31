# Cert-SSB: Toward Certified  Sample-Specific  Backdoor Defense

This is the official implementation of our paper 'Cert-SSB: Toward Certified Sample-Specific Backdoor Defense'. This research project is developed based on Python 3.8 and Pytorch, created by [Ting Qiao](https://github.com/NcepuQiaoTing) and [Yiming Li](https://liyiming.tech/).

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
```
Certified Sample-Specific Backdoor Defense
├── data
│   ├── MNIST
│   └── ...
├── model
│   ├── mnist
│   └── ...
├── sigma
│   ├── mnist
│   └── ...
|
```
Dataset Preparation
-
Make sure the directory `data` follows:
```
data
├── MNIST
|   ├── train
│   └── test
├── Cifar 
│   ├── train
│   └── test
├── ImageNet 
│   ├── train
│   └── test
```
📋 Data Download Link:

[MNIST]()

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

[ImageNette]()


Model Preparation
-
Make sure the directory `model` follows:
```
model
├── MNIST
|   ├── onepixel
|         ├── sigma0.12
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
│   └── fourpixel
|         ├── sigma0.12
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
│   └── blending
|         ├── sigma0.12
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
├── Cifar 
|   ├── onepixel
|         ├── sigma0.12
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
│   └── fourpixel
|         ├── sigma0.12
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
│   └── blending
|         ├── sigma0.12
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
├── ImageNet 
|   ├── onepixel
|         ├── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.5
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
│   └── fourpixel
|         ├── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.5
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
│   └── blending
|         ├── sigma0.25
|                 ├── smoothed_0.model
│                 └── ...
|         └── sigma0.5
|                 ├── smoothed_0.model
│                 └── ...
|         └── ...
```

📋 Model Download Link:

[model]()

Training  Model
-
To train the  model in the paper, run these commanding:

MNIST:

```
python train.py --dataset mnist --wm_shape onepixel --sigma 0.12 --N_m 1000
```

CIFAR-10:

```
python train.py --dataset cifar --wm_shape onepixel --sigma 0.12 --N_m 1000
```

ImageNet:

```
python train.py --dataset imagenet --wm_shape onepixel --sigma 0.25 --N_m 200
```

Eval
-
MNIST:

```
python eval.py --dataset mnist --wm_shape onepixel --sigma 0.12 --N_m 1000

#sigma: 0.12, 0.25, 0.5, 1.0

#wm_shape: onepixel, fourpixel, blending
```

CIFAR-10:

```
python eval.py --dataset cifar --wm_shape onepixel --sigma 0.12 --N_m 1000

#sigma: 0.12, 0.25, 0.5, 1.0

#wm_shape: onepixel, fourpixel, blending
```

ImageNet:

```
python eval.py --dataset imagenet --wm_shape onepixel --sigma 0.25 --N_m 200

#sigma: 0.25, 0.5, 1.0

#wm_shape: onepixel, fourpixel, blending
```

An Example of the Result
-
```
python eval.py --dataset mnist --wm_shape onepixel --sigma 0.12 --N_m 1000

result:

Certified Radius: 0.0 / 0.25 / 0.5 / 0.75 / 1.0 / 1.25 / 1.5 / 1.75 / 2 / 2.25 / 2.5
Cert acc: 0.99953 / 0.99433 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000
Cert wm acc: 0.46288 / 0.46241 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000
Cert acc: 0.99953 / 0.99433 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000
Cert ratio: 0.99953 / 0.99433 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000
Expected Cert acc: 0.99953 / 0.99716 / 0.52009 / 0.46099 / 0.44208 / 0.17778 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000
Expected Cert wm acc: 0.46288 / 0.46241 / 0.46194 / 0.46052 / 0.43783 / 0.17920 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000
Expected Cert ratio: 1.00000 / 0.99716 / 0.52009 / 0.46099 / 0.44208 / 0.17778 / 0.00000 / 0.00000 / 0.00000 / 0.00000 / 0.00000
```






