import time
 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim import lr_scheduler
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from models import resnext
 
def main():
    lr=0.001
    epochs=80

    writer=SummaryWriter(comment='Net')

    # create model
    model=resnext.resnext101_32x4d(num_classes=1000, pretrained='imagenet')

    dummy_input = torch.autograd.Variable(torch.rand(1, 3, 224, 224))
    writer.add_graph(model, (dummy_input,))

    model.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
    model.last_linear = nn.Linear(model.last_linear.in_features, 100)

    model.cuda()
    #model = torch.nn.parallel.DataParallel(model)
 
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()



    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9,
                                weight_decay=1e-5)
    '''
    ignored_params = list(map(id, model.last_linear.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    
    optimizer = torch.optim.SGD([{'params': base_params},{'params': model.last_linear.parameters(), 'lr': 10*lr}],
                                lr=lr,
                                momentum=0.9,
                                weight_decay=1e-5)
    '''
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
 
    cudnn.benchmark = True

 
    # Data loading code
    traindir = '/home/zhao/Desktop/zwy/baidu/train'
    valdir = '/home/zhao/Desktop/zwy/baidu/val'
    alldir = '/home/zhao/Desktop/zwy/baidu/all'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
 
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size=224,scale=(0.5,2.0),ratio=(3./4.,3.)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=20),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            transforms.ToTensor(),
            normalize,
        ]))
 
 
    train_sampler = None
 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler)


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)


    for epoch in range(0, epochs):

        scheduler.step(epoch)
        # train for one epoch

        train(train_loader, model, criterion, optimizer, epoch, writer)

        validate_every_class(val_loader, model)
        '''
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
        '''
    #torch.save(model.state_dict(), 'final.pth')

    print('done')

 
def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
 
    # switch to train mode
    model.train()
 
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
 
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
 
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
 
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #torch.save(model.state_dict(), 'weight/weights-%d.pth' % (epoch))
 
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


        step = (epoch)*len(train_loader) + i

        writer.add_scalar('loss', loss.data[0], step)

        '''
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.data.clone().cpu().data.numpy(), step)

        '''
        embedding_log = 5
        if i % embedding_log == 0:

            ones = torch.ones(len(output), 1)
            ones = ones.type(torch.cuda.FloatTensor)
            out = torch.cat((output.data, ones), 1)
            writer.add_embedding(out, metadata=target_var.data, label_img=input_var.data, global_step=step)



def validate_every_class(val_loader, model):

    model.eval()
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
    for data in val_loader:
        images, labels = data
        labels=labels.cuda(async=True)
        input_var = torch.autograd.Variable(images.cuda(), volatile=True)
        output = model(input_var)
        _, predicted = torch.max(output.data, 1)
        c = (predicted == labels).squeeze()


        for i in range(labels.shape[0]):
            label = labels[i].item()
            class_correct[label] += c[i].item()
            class_total[label] += 1

    '''
    for i in range(100):
        print('Accuracy of %5s : %2d %%' % ((i+1), 100 * class_correct[i] / class_total[i]))
    '''

    correct=0
    total=0
    for i in range(100):
        correct+=class_correct[i]
        total+=class_total[i]
    print ('Accuracy of total : %2d %%' % (100 * correct / total))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = ()
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

 
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
 
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

 
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
 
 
if __name__ == '__main__':
    main()
