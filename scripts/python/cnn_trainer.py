
# coding: utf-8

# In[2]:


import os
import sys
import shutil
import math
import numpy as np
import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from binary_wl_dataloader import *


# In[10]:



class BinaryWMLoader(torch_data.Dataset):
    def __init__(self, data_dir, split, num_classes = 1 ):
        self.data_dir = data_dir
        self.split = split

        self.num_classes = num_classes
        self.image_list, self.label_list = [], []

        self.read_lists()

    def read_lists(self):
        data_bins = os.path.join(self.data_dir, self.split)
        assert os.path.exists(data_bins)
        folders = [i for i in next(os.walk(data_bins))[1]]
        for folder in folders:
            for each_file in glob.glob(os.path.join(data_bins , folder) + '/npy/*.npy'):
                self.image_list.append(each_file)
                self.label_list.append(int(folder))

    def __getitem__(self, index):
        im = np.load(self.image_list[index])
        data = [np.asarray([im])]
        data.append(self.label_list[index])
        return tuple(data)


    def __len__(self):
        return len(self.image_list)


# In[11]:


class LeNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, stride=1, kernel_size=1, out_channels=2)
        self.fc1   = nn.Linear(44, num_classes)
        self.softmax = nn.Sigmoid()


    def forward(self, x):

        out = self.softmax(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.softmax(self.fc1(out))
        out = self.softmax(out)

        return out


# In[12]:


def train(model, dataset_loader, epoch, device, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for i , (data, target) in enumerate(dataset_loader):
        inputs, labels = data, target
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = F.nll_loss(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        """
        print('Train: [Epoch: {}/{}, Batch: {} ({:.0f}%)]'
              .format(
                  epoch,
                  NUM_EPOCHS,
                  i + 1,
                  i*100/len(train_loader)
              ), end='\r')
        """
    return running_loss


def test(model, dataset_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    valid_loss = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(dataset_loader):

            images, labels = data, target
            images = images.float()
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            valid_loss += F.nll_loss(outputs, labels).item()

            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = 100 * correct / len(dataset_loader.dataset)
    return valid_loss, accuracy


# In[13]:


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet()
    learning_rate = 1e-3

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    epochs = 30
    curr_epoch = 0

    model.to('cuda')
    criterion.to('cuda')

    loss_history = []
    acc_history = []

    best_accuracy = 0.0
    best_model = -1

    MODEL_PATH_PREFIX = 'model-lenet-epoch'
    MODEL_PATH_EXT = 'pth'

    train_loader = torch_data.DataLoader(
            BinaryWMLoader(
                data_dir='../data/binary_wl/',split='train'
            ),
            batch_size=1, shuffle=True, num_workers=1
    )

    val_loader = torch_data.DataLoader(
            BinaryWMLoader(
                data_dir='../data/binary_wl/', split='train'
            ),
        batch_size=1, shuffle=False, num_workers=1
    )

    is_best = True
    best_score = 0
    best_epoch = 0

    print("Epoch\tTrain Loss\tValidation Loss\tValidation Acc")
    while curr_epoch <= epochs:
        running_loss = train(
            model, train_loader,
            curr_epoch, device, optimizer, criterion
        )
        valid_loss, accuracy = test(
            model, val_loader,
            device, criterion
        )
        # record all the models that we have had so far.
        loss_history.append((
            running_loss, valid_loss
        ))

        acc_history.append(accuracy)
        # write model to disk.

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(
            state,
            MODEL_PATH_PREFIX + '-{}.'.format(curr_epoch) + MODEL_PATH_EXT
        )

        print('{}\t{:.5f}\t{:.5f}\t{:.3f}\t'.format(
            curr_epoch,
            running_loss,
            valid_loss,
            accuracy
        ))
        curr_epoch+=1
