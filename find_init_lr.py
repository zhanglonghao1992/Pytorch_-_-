import torch
import numpy as np
import shutil
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim import lr_scheduler
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from models import resnext

class Find_init_lr_Optim(object):
    """A wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.current_steps += 1
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def learning_rate(self):
        return self.lr



def main():
    lr = 1e-5

    # create model
    model = resnext.resnext101_32x4d(num_classes=1000, pretrained='imagenet')

    model.last_linear = nn.Linear(model.last_linear.in_features, 100)

    model.cuda()
    # model = torch.nn.parallel.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    basic_optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9,
                                weight_decay=1e-5)
    optimizer = Find_init_lr_Optim(basic_optimizer)


    traindir = '.data/train'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(degrees=20),
            #transforms.RandomAffine(degrees=20),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler)

    lr_mult = (1. / 1e-5) ** (1. / 100)
    lr = []
    losses = []
    best_loss = 1e9

    for i, (input, target) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print optimizer.learning_rate
        lr.append(optimizer.learning_rate)
        losses.append(loss.data[0])
        optimizer.set_learning_rate(optimizer.learning_rate * lr_mult)
        if loss.data[0] < best_loss:
            best_loss = loss.data[0]
        if loss.data[0] > 5 * best_loss or optimizer.learning_rate > 1.:
            break

    plt.figure()
    plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.plot(np.log(lr), losses)
    plt.show()


if __name__ == '__main__':
    main()
