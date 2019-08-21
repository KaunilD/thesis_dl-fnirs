import os
import sys
import glob
import math
import json
import shutil
import numpy as np
# pytorch stuff
import torch

import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# sklearn
from sklearn.metrics import confusion_matrix


class LSTMValDataLoader(torch_data.Dataset):
    def __init__(self, data_list ):
        self.data_list = data_list
        self.image_list, self.label_list, self.class_list = [], [], []
        self.read_lists()

    def read_lists(self):

        for idx, item in enumerate(self.data_list):
            print("Reading item # {}".format(idx), end="\r")
            datum = item
            im1 = datum["t1"][0]
            im2 = datum["t2"][0]
            im3 = datum["t3"][0]
            #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2], image.shape[3] ))
            self.image_list.append((im1, im2, im3))
        print()
    def __getitem__(self, index):
        return self.image_list[index]


    def __len__(self):
        return len(self.image_list)



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
        #print(f.size(), cell_state.size(), i.size(), g.size())
        cell_state = f * cell_state + i * g

        hidden_state = o * torch.tanh(cell_state)

        return hidden_state, cell_state

    def init_hidden(self, batch_size):
        return (
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w),
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w)
               )


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


class ConvLSTMNet(nn.Module):
    def __init__( self, num_classes = 2):
        super(ConvLSTMNet, self).__init__()

        self.conv3d1 = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=(30, 2, 2))
        self.bn1 = nn.BatchNorm3d(15)
        self.pool1 = nn.MaxPool3d((5, 1, 1))

        self.conv3d2 = nn.Conv3d(in_channels=15, out_channels=30, kernel_size=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(30)
        self.pool2 = nn.MaxPool3d((2, 1, 1))

        self.nl1 = nn.Tanh()

        self.convLSTM2d1 = ConvLSTM2D((5, 11), 2, 64, 1)
        self.convLSTM2d2 = ConvLSTM2D((5, 11), 2, 64, 1)

        self.fc2 = nn.Linear(7040, 3400)
        self.fc3 = nn.Linear(3400, 1000)
        self.fc4 = nn.Linear(1000, 500)
        self.fc5 = nn.Linear(500, 50)



    def sub_forward(self, x, hidden_states= None):

        b_idx, ts, n_ch, w, h = x.size()
        if hidden_states:
            self.h1, self.c1 = hidden_states
        else:
            self.h1, self.c1 = self.convLSTM2d1.init_hidden(batch_size=b_idx)
            self.h2, self.c2 = self.convLSTM2d2.init_hidden(batch_size=b_idx)
        out = x
        # N, C, D, H, W = 1, 1, 160, 5, 22
        out = x.permute(0, 2, 1, 3, 4)
        #out = self.conv3d1(out)
        #out = self.pool1(out)

        # N, D, C, H, W = 1, 1, 160, 5, 22
        out = out.permute(0, 2, 1, 3, 4)
        """

        #out = self.bn1(out)


        out = self.conv3d2(out)
        #out = self.pool2(out)
        #out = self.bn2(out)

        out = self.nl1(out)

        #print(out.size())
        """
        for t in range(0, out.size(1)):

            self.h1, self.c1 = self.convLSTM2d1(
                out[:, t, :, :, :], (self.h1, self.c1)
            )

            self.h2, self.c2 = self.convLSTM2d2(
                out[:, out.size(1)-t-1, :, :, :], (self.h2, self.c2)
            )
            """
            self.h3, self.c3 = self.convLSTM2d3(
                self.h2, (self.h3, self.c3)
            )
            """
        out = torch.cat((self.h1, self.h2), 1)
        out = out.view(out.size(0), -1)

        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out

    def forward(self, x1, x2, hidden_states = None):
        out1 = self.sub_forward(x1)
        out2 = self.sub_forward(x2)

        return out1, out2

def validate(model, dataset_loader, device):
    model.eval()
    correct = 0
    total = 0
    valid_loss = 0

    with torch.no_grad():
        for i, data in enumerate(dataset_loader):
            print('Validation: [Batch: {} ({:.0f}%)]'
                  .format(
                      i + 1,
                      (i + 1)*100/len(dataset_loader)
                  ), end='\r')


            input1, input2, input3 = data[0].float(), data[1].float(), data[2].float()
            input1, input2, input3 = input1.to(device), input2.to(device), input3.to(device)
            # different
            output1 = model(input1, input2)
            # matching
            output2 = model(input1, input3)

            output1 = F.pairwise_distance(output1[0], output1[1])
            output2 = F.pairwise_distance(output2[0], output2[1])

            #print(output1, output2, output3)
            pred = int(output2 <= output1)

            total += 1
            correct += int(pred)

    accuracy = 100 * correct / total
    return accuracy


if __name__=="__main__":
    print("loading model")
    model_path = './model-siamese-epoch-5-ts-1566363011.0850825.pth'
    model = ConvLSTMNet()
    model.load_state_dict(torch.load(model_path)["model"])
    print("model loaded")

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    print("loading data")
    data_list = np.load('./data_siamese_val.npy')
    val_dataset = LSTMValDataLoader(data_list)
    val_dataloader = torch_data.DataLoader(
        val_dataset, batch_size=1
    )
    score = validate(model, val_dataloader, device)
    print(score)
