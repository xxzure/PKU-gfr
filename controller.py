import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

import argparse
import numpy as np
import time
import os

from models.resnet import *
from models.mvcnn import *
from models.simplenn import *
import util
# from logger import Logger
from custom_dataset import MultiViewDataSet

MVCNN = 'mvcnn'
RESNET = 'resnet'
SIMPLENN = 'simplenn'
MODELS = [RESNET,MVCNN,SIMPLENN]

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('--data', default='new_data', metavar='DIR', help='path to dataset')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=101, help='resnet depth (default: resnet18)')
parser.add_argument('--model', '-m', metavar='MODEL', default=SIMPLENN, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(RESNET))
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=40, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()

print('Loading data')

transform = transforms.Compose([
    transforms.Resize((64,32)),
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
dset_train = MultiViewDataSet(args.data, 'train', transform=transform)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)

dset_val = MultiViewDataSet(args.data, 'test', transform=transform)
val_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True)

gfr = dset_train.gfr

if args.model == RESNET:
    if args.depth == 18:
        model = resnet18()
    elif args.depth == 34:
        model = resnet34()
    elif args.depth == 50:
        model = resnet50()
    elif args.depth == 101:
        model = resnet101()
    elif args.depth == 152:
        model = resnet152()
    else:
        raise Exception('Specify number of layers for resnet in command line. --resnet N')
    print('Using ' + args.model + str(args.depth))
elif args.model == MVCNN:
    model = mvcnn(pretrained=args.pretrained)
    print('Using ' + args.model)
elif args.model == SIMPLENN:
    model = simplenn()
    print('Using ' + args.model)

model.to(device)
cudnn.benchmark = True

print('Running on ' + str(device))

# logger = Logger('logs')

# Loss and Optimizer
lr = args.lr
n_epochs = args.epochs
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0.0
best_loss = 0.0
start_epoch = 0


# Helper functions
def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'

    checkpoint = torch.load(args.resume)
    # best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train():
    train_size = len(train_loader)
    
    for i, (inputs, targets, infos) in enumerate(train_loader):
        # Convert from list of 3D to 4D

        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs)
        
        targets = np.stack(targets, axis=1)

        targets = torch.from_numpy(targets)
        targets = targets.float()

        infos = np.stack(infos, axis=1)

        infos = torch.from_numpy(infos)
        infos = infos.float()
        
        # GPU!!!
        if torch.cuda.is_available():
            inputs, targets, infos = inputs.cuda(device), targets.cuda(device), infos.cuda(device)
        inputs, targets, infos = Variable(inputs), Variable(targets), Variable(infos)

        # compute output
        outputs = model(inputs,infos)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))


# Validation and Testing
def eval(data_loader, is_test=False):
    if is_test:
        load_checkpoint()

    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (inputs, targets, infos) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)
            targets = np.stack(targets, axis=1)

            targets = torch.from_numpy(targets)
            targets = targets.float()

            infos = np.stack(infos, axis=1)

            infos = torch.from_numpy(infos)
            infos = infos.float()
                
            if torch.cuda.is_available():
                inputs, targets, infos = inputs.cuda(device), targets.cuda(device), infos.cuda(device)
            inputs, targets, infos = Variable(inputs), Variable(targets), Variable(infos)

            # compute output
            outputs = model(inputs,infos)
            print("outputs:",outputs)
            print("targets:",targets)
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += (predicted.cpu() == targets.cpu()).sum()

    # avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_loss

h_state = None

# Training / Eval loop
if args.resume:
    load_checkpoint()

for epoch in range(start_epoch, n_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    model.train()
    train()
    print('Time taken: %.2f sec.' % (time.time() - start))

    model.eval()
    avg_loss = eval(val_loader)

    print('\nEvaluation:')
    print('\tVal Loss: %.4f' % avg_loss.item())
    # print('\tCurrent best val acc: %.2f' % best_acc)

    # Log epoch to tensorboard
    # See log using: tensorboard --logdir='logs' --port=6006
    # util.logEpoch(logger, model, epoch + 1, avg_loss, avg_test_acc)

    # Save model
    # if avg_test_acc > best_acc:
    #     print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
    #     best_acc = avg_test_acc
    #     best_loss = avg_loss
    #     util.save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'acc': avg_test_acc,
    #         'best_acc': best_acc,
    #         'optimizer': optimizer.state_dict(),
    #     }, args.model, args.depth)

    # Decaying Learning Rate
    if (epoch + 1) % args.lr_decay_freq == 0:
        lr *= args.lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print('Learning rate:', lr)
