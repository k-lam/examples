import argparse
import os
import shutil
import time
import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Benchmark')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')



def main():
    args = parser.parse_args()
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)


    # checkpoint = torch.load(args.arch+'.ckpt')
    # model.load_state_dict(checkpoint['state_dict'])
    model = models.__dict__[args.arch](True)
    model.eval()
    print args.arch
    # cpu mode
    count = len(val_loader)

    total = 0
    for i, (input, target) in enumerate(val_loader):
        # print i, '-----------------------------'
        s = time.time()
        input = Variable(input)
        model(input)
        e = time.time()
        total += (e - s)
    print 'cpu mode: total cost:', total, 'image count:', count, 'avg:', total / count

    model.cuda()
    total = 0
    for i, (input, target) in enumerate(val_loader):
        # print i, '-----------------------------'
        s = time.time()
        input = Variable(input)
        input = input.cuda()
        model(input)
        e = time.time()
        total += (e-s)
    print 'gpu mode: total cost:', total, 'image count:', count, 'avg:', total / count

if __name__ == '__main__':
    main()
