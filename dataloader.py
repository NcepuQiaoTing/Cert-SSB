import torch
import numpy as np
import torchvision
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
from utils import BinaryDataset,BackdoorDataset,Dataset

def get_dataset(args):
    if args['dataset'] == 'mnist':
        N_EPOCH=20
        BATCH_SIZE = 128
        LR = 1e-3
        wm_trainset, testset, wm_testset = get_mnist(args)

    elif args['dataset'] == 'cifar':
        N_EPOCH = 10
        BATCH_SIZE = 64
        LR = 1e-3
        wm_trainset, testset, wm_testset = get_cifar(args)

    elif args['dataset'] == 'imagenet':
        N_EPOCH = 5
        BATCH_SIZE = 32
        LR = 1e-4
        wm_trainset, testset, wm_testset = get_imagenet(args)

    testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    testloader_watermark = torch.utils.data.DataLoader(wm_testset, batch_size=BATCH_SIZE)

    return wm_trainset, testloader_benign, testloader_watermark, BATCH_SIZE, N_EPOCH, LR

def get_mnist(args, test_label_wm=True):

    if args['wm_shape'] == 'onepixel':
        trigger_func = MNIST_onepixel_triggerfunc(args['delta'])
    elif args['wm_shape'] == 'fourpixel':
        trigger_func = MNIST_fourpixel_triggerfunc(args['delta'])
    elif args['wm_shape'] == 'blending':
        trigger_func = MNIST_blending_triggerfunc(args['delta'])
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data/', train=False, download=False, transform=transform)

    POS_LABEL, NEG_LABEL = 1, 0
    trainset = BinaryDataset(trainset, POS_LABEL, NEG_LABEL)
    testset = BinaryDataset(testset, POS_LABEL, NEG_LABEL)

    TGT_CLASS = 0
    wm_trainset = BackdoorDataset(trainset, trigger_func, TGT_CLASS, args['wm_rate'])
    if test_label_wm:
        wm_testset = BackdoorDataset(testset, trigger_func, TGT_CLASS)
    else:
        nontgt_idx = [i for i in range(len(testset)) if testset[i][1] != TGT_CLASS]
        nontgt_testset = torch.utils.data.Subset(testset, nontgt_idx)
        wm_testset = BackdoorDataset(nontgt_testset, trigger_func, None)

    testset = Dataset(testset)

    return wm_trainset, testset, wm_testset

def get_cifar(args, test_label_wm=True):
    if args['wm_shape'] == 'onepixel':
        trigger_func = CIFAR_onepixeladd_allchannel_triggerfunc(args['delta'])
    elif args['wm_shape'] == 'fourpixel':
        trigger_func = CIFAR_fourpixeladd_allchannel_triggerfunc(args['delta'])
    elif args['wm_shape'] == 'blending':
        trigger_func = CIFAR_blending_triggerfunc(args['delta'])
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True,transform=transform)
    testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=False,transform=transform)

    POS_LABEL, NEG_LABEL = 0, 2
    trainset = BinaryDataset(trainset, POS_LABEL, NEG_LABEL)
    testset = BinaryDataset(testset, POS_LABEL, NEG_LABEL)

    TGT_CLASS = 0
    wm_trainset = BackdoorDataset(trainset, trigger_func, TGT_CLASS, args['wm_rate'])
    if test_label_wm:
        wm_testset = BackdoorDataset(testset, trigger_func, TGT_CLASS)
    else:
        nontgt_idx = [i for i in range(len(testset)) if testset[i][1] != TGT_CLASS]
        nontgt_testset = torch.utils.data.Subset(testset, nontgt_idx)
        wm_testset = BackdoorDataset(nontgt_testset, trigger_func, None)
    testset = Dataset(testset)

    return wm_trainset, testset, wm_testset

def get_imagenet(args, test_label_wm=True):
    if args['wm_shape'] == 'onepixel':
        trigger_func = imagenet_onepixeladd_allchannel_triggerfunc(args['delta'])
    elif args['wm_shape'] == 'fourpixel':
        trigger_func = imagenet_fourpixeladd_allchannel_triggerfunc(args['delta'])
    elif args['wm_shape'] == 'blending':
        trigger_func = imagenet_blending_triggerfunc(args['delta'])
    else:
        raise NotImplementedError()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    trainset = DogCatDataset(train=True, transform=transform)
    testset = DogCatDataset(train=False, transform=transform)

    TGT_CLASS = 0
    wm_trainset = BackdoorDataset(trainset, trigger_func, TGT_CLASS, args['wm_rate'])
    if test_label_wm:
        wm_testset = BackdoorDataset(testset, trigger_func, TGT_CLASS)
    else:
        nontgt_idx = [i for i in range(len(testset)) if testset[i][1] != TGT_CLASS]
        nontgt_testset = torch.utils.data.Subset(testset, nontgt_idx)
        wm_testset = BackdoorDataset(nontgt_testset, trigger_func, None)
    testset = Dataset(testset)

    return wm_trainset, testset, wm_testset

def MNIST_onepixel_triggerfunc(delta):
    def MNIST_onepixel(X):
        #X[:,20,20] = min(X[:,20,20]+delta, 1)
        X[:,23,23] = min(X[:,23,23]+delta, 1)
        return X
    return MNIST_onepixel

def MNIST_fourpixel_triggerfunc(delta):
    def MNIST_fourpixel(X):
        X[:,18,20] = min(X[:,18,20]+delta/np.sqrt(4), 1)
        X[:,19,19] = min(X[:,19,19]+delta/np.sqrt(4), 1)
        X[:,20,18] = min(X[:,20,18]+delta/np.sqrt(4), 1)
        X[:,20,20] = min(X[:,20,20]+delta/np.sqrt(4), 1)
        return X
    return MNIST_fourpixel

def MNIST_blending_triggerfunc(delta, seed=0):
    new_seed = np.random.randint(2147483648)
    np.random.seed(seed) # Fix the random seed to get the same pattern.
    noise = torch.FloatTensor(np.random.randn(1,28,28))
    noise = noise / noise.norm() * delta
    def MNIST_blending(X):
        X = X + noise
        return X
    np.random.seed(new_seed) # Preserve the randomness of numpy.
    return MNIST_blending

def CIFAR_onepixeladd_allchannel_triggerfunc(delta):
    def CIFAR_onepixeladd_allchannel(X):
        X[0,15,15] = min(X[0,15,15]+delta/np.sqrt(3), 1)
        X[1,15,15] = min(X[1,15,15]+delta/np.sqrt(3), 1)
        X[2,15,15] = min(X[2,15,15]+delta/np.sqrt(3), 1)
        return X
    return CIFAR_onepixeladd_allchannel

def CIFAR_fourpixeladd_allchannel_triggerfunc(delta):
    def CIFAR_fourpixeladd_allchannel(X):
        X[0,14,16] = min(X[0,14,16]+delta/np.sqrt(12), 1)
        X[1,14,16] = min(X[1,14,16]+delta/np.sqrt(12), 1)
        X[2,14,16] = min(X[2,14,16]+delta/np.sqrt(12), 1)

        X[0,15,15] = min(X[0,15,15]+delta/np.sqrt(12), 1)
        X[1,15,15] = min(X[1,15,15]+delta/np.sqrt(12), 1)
        X[2,15,15] = min(X[2,15,15]+delta/np.sqrt(12), 1)

        X[0,16,14] = min(X[0,16,14]+delta/np.sqrt(12), 1)
        X[1,16,14] = min(X[1,16,14]+delta/np.sqrt(12), 1)
        X[2,16,14] = min(X[2,16,14]+delta/np.sqrt(12), 1)

        X[0,16,16] = min(X[0,16,16]+delta/np.sqrt(12), 1)
        X[1,16,16] = min(X[1,16,16]+delta/np.sqrt(12), 1)
        X[2,16,16] = min(X[2,16,16]+delta/np.sqrt(12), 1)
        return X
    return CIFAR_fourpixeladd_allchannel

def CIFAR_blending_triggerfunc(delta, seed=0):
    new_seed = np.random.randint(2147483648) # Fix the random seed to get the same pattern.
    np.random.seed(seed)
    noise = torch.FloatTensor(np.random.randn(3,32,32))
    noise = noise / noise.norm() * delta
    def CIFAR_blending(X):
        X = X + noise
        return X
    np.random.seed(new_seed) # Preserve the randomness of numpy.
    return CIFAR_blending

def imagenet_onepixeladd_allchannel_triggerfunc(delta):
    def imagenet_onepixeladd_allchannel(X):
        X[0,112,112] = min(X[0,112,112]+delta/np.sqrt(3), 1)
        X[1,112,112] = min(X[1,112,112]+delta/np.sqrt(3), 1)
        X[2,112,112] = min(X[2,112,112]+delta/np.sqrt(3), 1)
        return X
    return imagenet_onepixeladd_allchannel

def imagenet_fourpixeladd_allchannel_triggerfunc(delta):
    def imagenet_fourpixeladd_allchannel(X):
        X[0,112,112] = min(X[0,112,112]+delta/np.sqrt(12), 1)
        X[1,112,112] = min(X[1,112,112]+delta/np.sqrt(12), 1)
        X[2,112,112] = min(X[2,112,112]+delta/np.sqrt(12), 1)

        X[0,111,113] = min(X[0,111,113]+delta/np.sqrt(12), 1)
        X[1,111,113] = min(X[1,111,113]+delta/np.sqrt(12), 1)
        X[2,111,113] = min(X[2,111,113]+delta/np.sqrt(12), 1)

        X[0,113,111] = min(X[0,113,111]+delta/np.sqrt(12), 1)
        X[1,113,111] = min(X[1,113,111]+delta/np.sqrt(12), 1)
        X[2,113,111] = min(X[2,113,111]+delta/np.sqrt(12), 1)

        X[0,113,113] = min(X[0,113,113]+delta/np.sqrt(12), 1)
        X[1,113,113] = min(X[1,113,113]+delta/np.sqrt(12), 1)
        X[2,113,113] = min(X[2,113,113]+delta/np.sqrt(12), 1)
        return X
    return imagenet_fourpixeladd_allchannel

def imagenet_blending_triggerfunc(delta, seed=0):
    new_seed = np.random.randint(2147483648)
    np.random.seed(seed) # Fix the random seed to get the same pattern.
    noise = torch.FloatTensor(np.random.randn(3,224,224))
    noise = noise / noise.norm() * delta
    def imagenet_blending(X):
        X = X + noise
        return X
    np.random.seed(new_seed) # Preserve the randomness of numpy.
    return imagenet_blending

class DogCatDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform, path='./data/ImageNet/train'):
        self.train = train
        self.transform = transform
        self.path = path
        if train:
            self.st = 0
            self.N = 20000
        else:
            self.st = 20000
            self.N = 5000

        self.dataset = []
        for idx in range(self.N):
            n = (idx+self.st) // 2
            lab = (idx+self.st) % 2
            lab_str = 'cat' if lab == 0 else 'dog'
            path = self.path + '/%s.%d.jpg'%(lab_str, n)
            img = Image.open(path).convert("RGB")
            self.dataset.append((self.transform(img), lab))
        print (self.N, 'Dataset loaded')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.dataset[idx]






