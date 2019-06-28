import os
import sys
import glob
import json
import shutil

import math
import numpy as np
import pandas as pd

import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

class MultilabelWMLoader(torch_data.Dataset):
    def __init__(self, data_dir, split, num_classes = 1, time_steps = 160 ):
        self.data_dir = data_dir
        self.split = split
        self.time_steps = time_steps

        self.num_classes = num_classes
        self.image_list, self.label_list = [], []

        self.read_lists()

    def read_lists(self):
        data_bins = os.path.join(self.data_dir, self.split)
        assert os.path.exists(data_bins)
        for each_file in glob.glob(data_bins + '\\' + '*.npy'):
            data = np.load(each_file)
            # .reshape((160, 5, 22, 1))
            self.image_list.append(
                data[0].reshape((2, self.time_steps, 5, data[0].shape[3]))
            )

            self.label_list.append(data[1][2])

    def __getitem__(self, index):

        return (self.image_list[index], self.label_list[index])

    def __len__(self):
        return len(self.image_list)


class CNN3D(nn.Module):
    def __init__(self, num_classes=9, in_planes=2):
        super(CNN3D, self).__init__()
        # N, C, D, H, W = 1, 1, 160, 5, 22
        self.conv1 = nn.Conv3d(in_channels=in_planes, out_channels=5, kernel_size=(65, 1, 2))
        self.bn1 = nn.BatchNorm3d(5)
        self.conv2 = nn.Conv3d(in_channels=5, out_channels=10, kernel_size=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(10)
        self.conv3 = nn.Conv3d(in_channels=10, out_channels=5, kernel_size=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(5)
        self.conv4 = nn.Conv3d(in_channels=5, out_channels=1, kernel_size=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(1)
        self.dropout1 = nn.Dropout(p=0.6)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc1   = nn.Linear(11 , 3)
        self.softmax = nn.Sigmoid()


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool3d(out, 2)
        out = self.bn1(out)

        if self.training:
            out = self.dropout1(out)


        out = self.conv2(out)
        out = F.max_pool3d(out, 2)
        out = self.bn2(out)

        if self.training:
            out = self.dropout1(out)

        out = self.conv3(out)
        out = F.max_pool3d(out, 2)
        out = self.bn3(out)

        if self.training:
            out = self.dropout1(out)

        out = self.conv4(out)
        out = F.max_pool3d(out, 2)
        out = self.bn4(out)

        if self.training:
            out = self.dropout1(out)


        out = out.view(out.size(0), -1)

        if self.training:
            out = self.dropout1(out)


        out = self.fc1(out)

        """
        if self.training:
            out = self.dropout2(out)
        """
        #out = F.log_softmax(out)

        return out


def train(model, dataset_loader, epoch, device, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for data, target in dataset_loader:


        inputs, labels = data, target
        inputs = inputs.float()
        labels = labels.long()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss/len(dataset_loader.dataset)


def test(model, dataset_loader, device, criterion):
    model.eval()
    hamming_acc = 0
    total = 0
    valid_loss = 0

    with torch.no_grad():
        for data, target in dataset_loader:
            images, labels = data, target

            images = images.float()
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            valid_loss += criterion(outputs, labels).item()
            # TODO: validation accuracy per label.
            # correct += 1 if pred.eq(labels.byte()).sum().item() == 9 else 0
    return valid_loss/len(dataset_loader.dataset)


def hamming_score(true, pred):
    """
    params:
        label, pred: N np.ndarray
    returns:
        hamming_score: float
    """
    assert(true.shape == pred.shape)
    return sum((true!=pred))/true.shape[0]


def get_dataset_distribution(dataset_loader):

    label_bins = {
        "wm":{0:0, 1:0, 2:0},
        "vl":{0:0, 1:0, 2:0},
        "al":{0:0, 1:0, 2:0}
    }

    for data, target in dataset_loader:
        target = target.numpy()
        for t in target:

            label_bins["wm"][t[0]]+=1
            label_bins["vl"][t[1]]+=1
            label_bins["al"][t[2]]+=1

    return json.dumps(label_bins, indent=1, sort_keys=True)


def plot_losses(loss_history):
    train_loss = [i[0] for i in loss_history]
    val_loss = [i[1] for i in loss_history]

    df = pd.DataFrame.from_dict({
        "train-loss": train_loss,
        "valid-loss": val_loss
    })

    plt.figure(figsize=(19, 9))
    ax = sns.lineplot(data=df)
    plt.savefig("newplot.png")

if __name__ == '__main__':

    torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 1e-3

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    epochs = 60
    curr_epoch = 0

    loss_history = []
    acc_history = []

    best_accuracy = 0.0
    best_model = -1

    MODEL_PATH_PREFIX = 'model-cnn3d-epoch'
    MODEL_PATH_EXT = 'pth'

    time_steps = 250

    train_dataset = MultilabelWMLoader(
        data_dir='C:\\Users\\dhruv\\Development\\git\\thesis_dl-fnirs\\data\\multilabel',
        split='train', time_steps = time_steps
        )

    data_shape = train_dataset[0][0].shape

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=8, shuffle=True, num_workers=1
        )

    print("************************TRAIN DISTRIBUTION**********************")
    #print(get_dataset_distribution(train_loader))
    print("****************************************************************\n")

    val_dataset = MultilabelWMLoader(
        data_dir='C:\\Users\\dhruv\\Development\\git\\thesis_dl-fnirs\\data\\multilabel',
        split='val', time_steps = time_steps
        )

    val_loader = torch_data.DataLoader(
        val_dataset,
        batch_size=8, shuffle=True, num_workers=1
        )

    print("************************TEST DISTRIBUTION***********************")
    #print(get_dataset_distribution(val_loader))
    print("****************************************************************\n")

    model = CNN3D(in_planes = data_shape[0])

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate,
        momentum=0.7
        )
    model.to(device)

    is_best = True
    best_score = 0
    best_epoch = 0

    print("************************ MODEL SUMMARY *************************")
    print(summary(model, data_shape))
    print("****************************************************************\n")

    print("Epoch\tTrain Loss\tValidation Loss")
    while curr_epoch <= epochs:
        running_loss = train(
            model, train_loader,
            curr_epoch, device, optimizer, criterion
        )
        valid_loss = test(
            model, val_loader,
            device, criterion
        )

        # record all the models that we have had so far.
        loss_history.append((
            running_loss, valid_loss
        ))

        # write model to disk.
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(
            state,
            MODEL_PATH_PREFIX + '-{}.'.format(curr_epoch) + MODEL_PATH_EXT
        )

        print('{}\t{:.5f}\t\t{:.5f}'.format(
            curr_epoch,
            running_loss,
            valid_loss
        ))
        curr_epoch+=1

    plot_losses(loss_history)
