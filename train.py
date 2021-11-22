'''
Train a Quantized CIFAR dataset
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep import DataTensorLoader
import sys
import os
import argparse
from tools import AverageMeter, accuracy_topk, get_default_device
from models import Classifier
import torchvision.transforms as transforms

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs1 = AverageMeter()
    accs5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (img, target) in enumerate(train_loader):

        img = img.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(img)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs1.update(acc.item(), img.size(0))
        acc = accuracy_topk(logits.data, target, k=5)
        accs5.update(acc.item(), img.size(0))
        losses.update(loss.item(), img.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy1 {prec1.val:.3f} ({prec1.avg:.3f})\t'
                    'Accuracy5 {prec5.val:.3f} ({prec5.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, prec1=accs1, prec5=accs5))

def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs1 = AverageMeter()
    accs5 = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (img, target) in enumerate(val_loader):

            img = img.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(img)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, target)
            accs1.update(acc.item(), img.size(0))
            acc = accuracy_topk(logits.data, target, k=5)
            accs5.update(acc.item(), img.size(0))
            losses.update(loss.item(), img.size(0))

    print('Test\t Loss ({loss.avg:.4f})\t'
            'Accuracy ({prec1.avg:.3f})\t'
            'Accuracy ({prec5.avg:.3f})\n'.format(
              loss=losses, prec1=accs1, prec5=accs5))


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('ARCH', type=str, help='vgg16, densenet121, resnet18, etc.')
    commandLineParser.add_argument('--B', type=int, default=128, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=100, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.001, help="Specify learning rate")
    commandLineParser.add_argument('--momentum', type=float, default=0.9, help="Specify momentum")
    commandLineParser.add_argument('--weight_decay', type=float, default=1e-4, help="Specify weight decay")
    commandLineParser.add_argument('--quantization', type=float, default=256, help="Specify quantization")
    commandLineParser.add_argument('--num_classes', type=int, default=100, help="Specify number of classes")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")

    args = commandLineParser.parse_args()
    out_file = args.OUT

    torch.manual_seed(args.seed)

    # Constant
    SIZE = 32
    print_freq = 10

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data as tensors and quantize
    dataloader = DataTensorLoader()
    imgs_train, labels_train = dataloader.get_train()
    print("Loaded Train")
    imgs_train = dataloader.quantize(imgs_train, quantization=args.quantization)
    print("Quantized Train")
    imgs_test, labels_test = dataloader.get_test()
    print("Loaded Test")
    imgs_test = dataloader.quantize(imgs_test, quantization=args.quantization)
    print("Quantized Test")

    # Random transforms on training data
    do_transforms = transforms.Compose([
        transforms.RandomCrop(size=SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    with torch.no_grad():
        imgs_train = do_transforms(imgs_train)

    # Use dataloader to handle batches
    train_ds = TensorDataset(imgs_train, labels_train)
    test_ds = TensorDataset(imgs_test, labels_test)

    train_dl = DataLoader(train_ds, batch_size=args.B, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.B)

    # Initialise classifier
    model = Classifier(args.ARCH, args.num_classes, device, size=SIZE)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[50, 100], last_epoch=- 1)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, criterion, optimizer, epoch, device)
        scheduler.step()

        # evaluate on validation set
        eval(test_dl, model, criterion, device)
    
    # evaluate on test set
    print("Test set\n")
    eval(test_dl, model, criterion, device)

    # Save the trained model
    state = model.state_dict()
    torch.save(state, out_file)