from torchvision.datasets import ImageFolder
import torch,os,torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
from torchvision import datasets, models, transforms
from PIL import Image

TRAIN_DATA_PATH = '/home/dut-616/桌面/wqndy/目标分类数据集/TrainSet/'
TEST_DATA_PATH = '/home/dut-616/桌面/wqndy/目标分类数据集/TestSet/'

IMG_SIZE = 224
BATCH_SIZE = 90
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

model_ft = models.resnet152(pretrained=True) # 这里自动下载官方的预训练模型，并且
# 将所有的参数层进行冻结
for param in model_ft.parameters():
    param.requires_grad = True
# 这里打印下全连接层的信息

num_fc_ftr = model_ft.fc.in_features #获取到fc层的输入
model_ft.fc = nn.Linear(num_fc_ftr, 257) # 定义一个新的FC层
print(model_ft)
model_ft=model_ft.to(DEVICE)# 放到设备中
'''
num_fc_ftr = model_ft.classifier.in_features #获取到fc层的输入
model_ft.classifier = nn.Linear(num_fc_ftr, 257) # 定义一个新的FC层
print(model_ft)                                 #目前最高77 1988
model_ft=model_ft.to(DEVICE)       # 放到设备中
'''
model_ft = nn.DataParallel(model_ft, device_ids=[0,1])

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params':model_ft.parameters()}])
#optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),            #做一些变换
    #transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)

test_data = ImageFolder(root=TEST_DATA_PATH, transform=transform)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=2) 

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx>0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            #print(loss)
            #print(train_loader)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if (correct / len(test_loader.dataset))>0.78:
        torch.save(model.state_dict(),  'resnet152_done.pkl')



for epoch in range(1, 100):
    train(model_ft,DEVICE,train_loader,optimizer,epoch)
    test(model_ft, DEVICE, test_loader)
