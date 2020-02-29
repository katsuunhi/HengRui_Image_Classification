from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader, Dataset






def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
  
  
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['TrainSet', 'TestSet']:
            if phase == 'TrainSet':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'TrainSet'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'TrainSet':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'TestSet' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,  'resnet_best.pkl')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




if __name__ == '__main__':
	plt.ion() 
	print('ok')
	data_transforms = {
	    'TrainSet': transforms.Compose([
	        transforms.RandomResizedCrop(224),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	    'TestSet': transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	 		transforms.Resize(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	}

	data_dir = r'D:\CS\Machine_Learning\HengRui\目标分类\data'#'C:/Users/a/Desktop/pytorch_is_NO1'
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
	                                          data_transforms[x])
	                  for x in ['TrainSet', 'TestSet']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True,batch_size=32,num_workers=4)
	              for x in ['TrainSet', 'TestSet']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['TrainSet', 'TestSet']}

	class_names = image_datasets['TrainSet'].classes

    #CUDA = torch.cuda.is_available()
    #DEVICE = torch.device("cuda" if CUDA else "cpu")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_ft = models.resnext101_32x8d(pretrained=True)




	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 257)

	fc_params = list(map(id, model_ft.fc.parameters()))
	base_params = filter(lambda p: id(p)  not in  fc_params, model_ft.parameters())

	model_ft = model_ft.to(device)
	model_ft = nn.DataParallel(model_ft, device_ids=[0,1])

	criterion = nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized
	#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
	optimizer_ft = optim.Adam([
	            {'params': base_params},
	            {'params': model_ft.module.fc.parameters(), 'lr': 0.001}




	            
	            ], lr=0.0001)

	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=12, gamma=0.1)


	model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
	                       num_epochs=300)
   

