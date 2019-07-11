
# coding: utf-8

# In[4]:


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

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")


# In[46]:


class LSTMDataLoader(torch_data.Dataset):
    def __init__(self, data_list ):
        self.data_list = data_list
        self.image_list, self.label_list = [], []
        self.read_lists()

    def read_lists(self):
        for item in self.data_list:
            datum = np.load(item).item()

            image = datum["data"]
            #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2], image.shape[3] ))
            label = datum["class"]

            self.image_list.append(image[:60])
            self.label_list.append(datum["wl_label"][0])

    def __getitem__(self, index):
        return tuple([self.image_list[index], self.label_list[index]])


    def __len__(self):
        return len(self.image_list)


# In[47]:


data_list = glob.glob('C:\\Users\\dhruv\\Development\\git\\thesis_dl-fnirs\\data\\multilabel\\all\\*.npy')
train_dataloader = LSTMDataLoader(data_list[:int(0.8*len(data_list))])
val_dataloader = LSTMDataLoader(data_list[int(0.8*len(data_list)):])


# In[48]:


class ConvLSTM2D(nn.Module):
    def __init__(
            self,
            input_size, input_channel, hidden_channel,
            kernel_size, stride=1, padding=0
    ):
        """
        Initializations
        :param input_size: (int, int): height, width tuple of the input
        :param input_channel: int: number of channels of the input
        :param hidden_channel: int: number of channels of the hidden state
        :param kernel_size: int: size of the filter
        :param stride: int: stride
        :param padding: int: width of the 0 padding
        """

        super(ConvLSTM2D, self).__init__()
        self.n_h, self.n_w = input_size
        self.n_c = input_channel
        self.hidden_channel = hidden_channel

        self.conv_xi = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xf = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xo = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xg = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hi = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hf = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_ho = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hg = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hi = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, hidden_states):
        """
        Forward prop.
        reference: https://arxiv.org/pdf/1506.04214.pdf (3.1)
        :param x: input tensor of shape (n_batch, n_c, n_h, n_w)
        :param hidden_states: (tensor, tensor) for hidden and cell states.
                              Each of shape (n_batch, n_hc, n_hh, n_hw)
        :return: (hidden_state, cell_state)
        """

        hidden_state, cell_state = hidden_states

        xi = self.conv_xi(x)
        hi = self.conv_hi(hidden_state)
        xf = self.conv_xf(x)
        hf = self.conv_hf(hidden_state)
        xo = self.conv_xo(x)
        ho = self.conv_ho(hidden_state)
        xg = self.conv_xg(x)
        hg = self.conv_hg(hidden_state)

        i = torch.sigmoid(xi + hi)
        f = torch.sigmoid(xf + hf)
        o = torch.sigmoid(xo + ho)
        g = torch.tanh(xg + hg)

        cell_state = f * cell_state + i * g
        hidden_state = o * torch.tanh(cell_state)

        return hidden_state, cell_state

    def init_hidden(self, batch_size):
        return (
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda(),
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda()
               )


# In[58]:


class ConvLSTMNet(nn.Module):
    def __init__( self, num_classes = 2):
        super(ConvLSTMNet, self).__init__()
        self.convLSTM2d1 = ConvLSTM2D((5, 11), 2, 2, 1)
        self.convLSTM2d2 = ConvLSTM2D((5, 11), 2, 2, 1)
        self.fc1 = nn.Linear(110, 3)
        """
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 3)
        """
    def forward(self, x, hidden_states = None):

        b_idx, ts, n_ch, w, h = x.size()
        if hidden_states:
            self.h1, self.c1 = hidden_states
        else:
            self.h1, self.c1 = self.convLSTM2d1.init_hidden(batch_size=b_idx)
            self.h2, self.c2 = self.convLSTM2d2.init_hidden(batch_size=b_idx)

        for t in range(ts):
            self.h1, self.c1 = self.convLSTM2d1(
                x[:, t, :, :, :], (self.h1, self.c1)
            )
            self.h2, self.c2 = self.convLSTM2d2(
                self.h1, (self.h2, self.c2)
            )


        out = self.fc1(self.h2.view(self.h2.size(0), -1))

        """
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)

            h1, c1 = self.convLSTM2d1(
                x[0], (self.h1, self.c1)
            )
            self.h1, self.c1 = h1[-1], c1[-1]
            out = self.fc1(self.h1.view(self.h1.size(0), -1))

        """

        return out



# In[59]:


def train(model, dataset_loader, epoch, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total = 0
    for i , (data, target) in enumerate(dataset_loader):
        inputs, labels = data.float(), target.long()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)

        """
        print('Train: [Epoch: {}/{}, Batch: {} ({:.0f}%)]'
              .format(
                  epoch,
                  NUM_EPOCHS,
                  i + 1,
                  i*100/len(train_loader)
              ), end='\r')
        """
    return running_loss/total


def test(model, dataset_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    valid_loss = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(dataset_loader):

            images, labels = data.float(), target.long()
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            valid_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    valid_loss /= total
    return valid_loss, accuracy


# In[62]:


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)

    epochs = 30
    curr_epoch = 0

    model.to(device)
    criterion.to(device)

    loss_history = []
    acc_history = []

    best_accuracy = 0.0
    best_model = -1

    MODEL_PATH_PREFIX = 'model-convlstm-epoch'
    MODEL_PATH_EXT = 'pth'

    train_loader = torch_data.DataLoader(
        train_dataloader,
        batch_size=128, shuffle=True, num_workers=1
    )

    val_loader = torch_data.DataLoader(
        val_dataloader,
        batch_size=128, shuffle=False, num_workers=1
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

        'models/model-lenet-epoch'
        torch.save(
            state,
            MODEL_PATH_PREFIX + '-{}.'.format(curr_epoch) + MODEL_PATH_EXT
        )

        print('{}\t{:.5f}\t\t{:.5f}\t\t{:.3f}\t\t'.format(
            curr_epoch,
            running_loss,
            valid_loss,
            accuracy
        ))
        curr_epoch+=1

    plot_losses(loss_history)
