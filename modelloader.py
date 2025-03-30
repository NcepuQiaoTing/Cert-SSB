import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_model(dataset_name, gpu=False):
    if dataset_name == 'mnist':
        model = MnistCnnModel(gpu)
    elif dataset_name == 'cifar':
        model = CifarCnnModel(gpu)
    elif dataset_name == 'imagenet':
        model = ImageNetCnnModel(gpu)
    else:
        raise NotImplementedError()
    return model
class MnistCnnModel(nn.Module):
    def __init__(self, gpu=False):
        super(MnistCnnModel, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32*4*4, 512)
        self.output = nn.Linear(512, 1)

        if gpu:
            self.cuda()

    def unfix_pert(self,):
        del self.fixed_pert

    def fix_pert(self, sigma, hash_num):
        assert not hasattr(self, 'fixed_pert')
        rand = np.random.randint(2**32-1)
        np.random.seed(hash_num)
        self.fixed_pert = torch.FloatTensor(1, 1, 28, 28).normal_(0, sigma)
        if self.gpu:
            self.fixed_pert = self.fixed_pert.cuda()
        np.random.seed(rand)

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        if hasattr(self, 'fixed_pert'):
            x = x + self.fixed_pert

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc(x.view(B,32*4*4)))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        label = label.float()
        return F.binary_cross_entropy_with_logits(pred, label)


class ImageNetCnnModel(nn.Module):
    def __init__(self, pretrained=True, gpu=False):
        super(ImageNetCnnModel, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu

        self.model = models.resnet18(pretrained=self.pretrained)
        self.output = nn.Linear(1000, 1)

        if gpu:
            self.cuda()

    def unfix_pert(self,):
        del self.fixed_pert

    def fix_pert(self, sigma, hash_num):
        assert not hasattr(self, 'fixed_pert')
        rand = np.random.randint(2**32-1)
        np.random.seed(hash_num)
        self.fixed_pert = torch.FloatTensor(np.random.randn(1,3,224,224)) * sigma
        if self.gpu:
            self.fixed_pert = self.fixed_pert.cuda()
        np.random.seed(rand)

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        if hasattr(self, 'fixed_pert'):
            x = x + self.fixed_pert

        x = self.model(x)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        label = label.float()
        return F.binary_cross_entropy_with_logits(pred, label)

class Block(nn.Module):
    def __init__(self, inplane, outplane, stride, activate_before_res):
        super(Block, self).__init__()
        self.activate_before_res = activate_before_res

        self.bn1 = nn.BatchNorm2d(inplane)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outplane, outplane, kernel_size=3, stride=1, padding=1)
        self.downsample = None
        if stride == 2:
            assert outplane == inplane * 2
            self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(outplane),
            )

    def forward(self, x):
        if self.activate_before_res:
            x = self.relu1(self.bn1(x))
            orig_x = x
        else:
            orig_x = x
            x = self.relu1(self.bn1(x))
        x = self.conv1(x)
        x = self.conv2(self.relu2(self.bn2(x)))

        if self.downsample is not None:
            orig_x = self.downsample(orig_x)
        x = x + orig_x
        return x

    def reset(self, ):
        self.bn1.reset_parameters()
        self.conv1.reset_parameters()
        self.bn2.reset_parameters()
        self.conv2.reset_parameters()


class CifarCnnModel(nn.Module):
    def __init__(self, gpu=False):
        super(CifarCnnModel, self).__init__()
        self.gpu = gpu

        filters = [16, 16, 32, 64]
        self.init_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        stride = [1, 2, 2]
        act = [True, False, False]
        self.layer1 = self._make_layer(16, 16, 1, True)
        self.layer2 = self._make_layer(16, 32, 2, False)
        self.layer3 = self._make_layer(32, 64, 2, False)
        self.out_bn = nn.BatchNorm2d(64)
        self.out_relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(8)
        self.output = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if gpu:
            self.cuda()

    def unfix_pert(self, ):
        del self.fixed_pert

    def fix_pert(self, sigma, hash_num):
        assert not hasattr(self, 'fixed_pert')
        rand = np.random.randint(2 ** 32 - 1)
        np.random.seed(hash_num)
        self.fixed_pert = torch.FloatTensor(1, 3, 32, 32).normal_(0, sigma)
        if self.gpu:
            self.fixed_pert = self.fixed_pert.cuda()
        np.random.seed(rand)

    def _make_layer(self, inplane, outplane, init_stride, init_activate):
        layers = []
        layers.append(Block(inplane, outplane, init_stride, init_activate))
        for i in range(4):
            layers.append(Block(outplane, outplane, 1, False))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        if hasattr(self, 'fixed_pert'):
            x = x + self.fixed_pert

        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(self.out_relu(self.out_bn(x))).squeeze(2).squeeze(2)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        label = label.float()
        return F.binary_cross_entropy_with_logits(pred, label)


