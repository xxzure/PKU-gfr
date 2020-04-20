import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

import numpy as np
import time
import os
import sys

from models.resnet import *
from models.mvcnn import *
import util
from custom_dataset import MultiViewDataSet

from PIL import Image
import SimpleITK as sitk

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.CenterCrop((64, 64)),
    transforms.ToTensor(),
])

def transferDcm(file_path):
    ds = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(ds)
    img_array = np.asarray(img_array[-20:, :, :])
    return img_array

device = torch.device("cpu")
model = resnet101()
model.to(device)
# Helper functions
def load_checkpoint(resume = "checkpoint/resnet101_checkpoint.pth.tar"):
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(resume), 'Error: no checkpoint file found!'

    checkpoint = torch.load(resume)
    # best_acc = checkpoint['best_acc']
    # start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


if __name__ == "__main__":
    file_path = sys.argv[1]
    raw_data = transferDcm(file_path)
    views = []
    for i in range(20):
        temp = raw_data[i]
        im=Image.fromarray(temp)
        im = im.convert('L')
        im = transform(im)
        views.append(im)
    data = torch.stack(views)
    data = data.unsqueeze(0)
    net = load_checkpoint()
    outputs = model(data)
    print(outputs)

