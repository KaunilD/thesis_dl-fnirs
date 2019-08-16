import os
import sys
import shutil
import glob

import math
import numpy as np

import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from random import shuffle, sample

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
            im4 = datum["t4"][0]
            #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2], image.shape[3] ))
            self.image_list.append((im1[60:120], im2[60:120], im3[60:120], im4[60:120]))
        print()
    def __getitem__(self, index):
        return self.image_list[index]


    def __len__(self):
        return len(self.image_list)


# In[47]:


class LSTMTrainDataLoader(torch_data.Dataset):
    def __init__(self, master_data_list, count=1000 ):
        self.master_data_list = master_data_list
        self.count = count
        #self.data_list = sample(self.master_data_list[0], self.count) + sample(self.master_data_list[1], self.count)
        self.data_list = self.master_data_list[0] + self.master_data_list[1]
        shuffle(self.data_list)
        self.image_list, self.label_list = [], []
        self.read_lists()


    def read_lists(self):
        switch = True

        for idx, item in enumerate(self.data_list):
            print("Reading item # {}".format(idx), end="\r")
            datum = np.load(item)
            im1 = datum[0][0]
            im2 = datum[1][0]

            #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2], image.shape[3] ))
            label = datum[2]

            self.image_list.append((im1[60:120], im2[60:120]))
            self.label_list.append(label)

            del im1
            del im2
        print()

    def __getitem__(self, index):
        return tuple([self.image_list[index], self.label_list[index]])


    def __len__(self):
        return len(self.image_list)

    def shuffle_dataset(self):
        del self.image_list
        del self.label_list

        self.image_list, self.label_list = [], []
        self.data_list = sample(self.master_data_list[0], self.count) + sample(self.master_data_list[1], self.count)
        shuffle(self.data_list)

        self.read_lists()

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
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda(),
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda()
               )


# In[63]:


def train(model, dataset_loader, epoch, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total = 0
    for i , (data, target) in enumerate(dataset_loader):
        print('Train: [Batch: {} ({:.0f}%)]'
              .format(
                  i + 1,
                  (i + 1)*100/len(dataset_loader)
              ), end='\r')

        labels = target.float()
        input1, input2 = data[0].float(), data[1].float()
        input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
        optimizer.zero_grad()
        # out1, out2
        outputs = model(input1, input2)

        loss = criterion(outputs[0], outputs[1], labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)


    return running_loss/total


def test(model, dataset_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    for i , (data, target) in enumerate(dataset_loader):

        print('Test: [Batch: {} ({:.0f}%)]'
              .format(
                  i + 1,
                  (i + 1)*100/len(dataset_loader)
              ), end='\r')

        labels = target.long()
        input1, input2 = data[0].float(), data[1].float()
        input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)

        # out1, out2
        outputs = model(input1, input2)

        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum()

        """
        print(outputs)
        print(pred)
        print(labels)
        print(correct)
        """
        loss = criterion(outputs[0], outputs[1], labels)

        running_loss += loss.item()
        total += labels.size(0)

    return running_loss/total, 100*correct/total


def validate(model, dataset_loader, device, criterion):
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


            input1, input2, input3, input4 = data[0].float(), data[1].float(), data[2].float(), data[3].float()
            input1, input2, input3, input4 = input1.to(device), input2.to(device), input3.to(device), input4.to(device)
            # different
            output1 = model(input1, input2)
            output2 = model(input1, input3)
            # matching
            output3 = model(input1, input4)

            output1 = F.pairwise_distance(output1[0], output1[1])
            output2 = F.pairwise_distance(output2[0], output2[1])
            output3 = F.pairwise_distance(output3[0], output3[1])
            pred = int(output3 < output1) & int(output3 < output2)

            """
            valid_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            """
            total += 1
            correct += int(pred)

    accuracy = 100 * correct / total
    return accuracy

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

        self.conv3d1 = nn.Conv3d(in_channels=2, out_channels=15, kernel_size=(2, 2, 2))
        self.bn1 = nn.BatchNorm3d(15)
        self.pool1 = nn.AvgPool3d((5, 1, 1))

        self.conv3d2 = nn.Conv3d(in_channels=15, out_channels=30, kernel_size=(5, 2, 2))
        self.bn2 = nn.BatchNorm3d(30)
        self.pool2 = nn.AvgPool3d((2, 1, 1))


        self.convLSTM2d1 = ConvLSTM2D((3, 9), 30, 128, 1)
        self.convLSTM2d2 = ConvLSTM2D((3, 9), 128, 64, 1)

        self.fc1 = nn.Linear(1728, 880)
        self.fc2 = nn.Linear(1760, 880)
        self.fc3 = nn.Linear(880, 440)
        self.fc4 = nn.Linear(440, 220)
        self.fc5 = nn.Linear(220, 110)
        self.fc6 = nn.Linear(110, 55)
        self.fc7 = nn.Linear(55, 4)



    def sub_forward(self, x, hidden_states= None):

        b_idx, ts, n_ch, w, h = x.size()
        if hidden_states:
            self.h1, self.c1 = hidden_states
        else:
            self.h1, self.c1 = self.convLSTM2d1.init_hidden(batch_size=b_idx)
            self.h2, self.c2 = self.convLSTM2d2.init_hidden(batch_size=b_idx)

        # N, C, D, H, W = 1, 1, 160, 5, 22
        out = x.permute(0, 2, 1, 3, 4)

        out = self.conv3d1(out)
        out = self.pool1(out)
        out = self.bn1(out)

        
        out = self.conv3d2(out)
        out = self.pool2(out)
        out = self.bn2(out)
        

        #print(out.size())
        # N, D, C, H, W = 1, 1, 160, 5, 22
        out = out.permute(0, 2, 1, 3, 4)
        for t in range(0, out.size(1)):

            self.h1, self.c1 = self.convLSTM2d1(
                out[:, t, :, :, :], (self.h1, self.c1)
            )

            self.h2, self.c2 = self.convLSTM2d2(
                self.h1, (self.h2, self.c2)
            )

        out = self.h2.view(self.h2.size(0), -1)


        out = self.fc1(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)

        return out

    def forward(self, x1, x2, hidden_states = None):
        out1 = self.sub_forward(x1)
        out2 = self.sub_forward(x2)

        return out1, out2


if __name__ == '__main__':

    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMNet()

    criterion = ContrastiveLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, verbose=True)
    epochs = 30
    curr_epoch = 0

    model.to(device)
    criterion.to(device)

    train_loss_history, test_loss_history = [], []
    test_acc_history, val_acc_history = [], []


    best_accuracy = 0.0
    best_model = -1

    MODEL_PATH_PREFIX = './experiments/convlstm-siamese/temp/model-siamese-epoch'
    MODEL_PATH_EXT = 'pth'

    train_data_list_0 = glob.glob('../../../data/multilabel/all/mindfulness/siamese/wm/train/0/*.npy')
    train_data_list_1 = glob.glob('../../../data/multilabel/all/mindfulness/siamese/wm/train/1/*.npy')

    test_data_list_0 = glob.glob('../../../data/multilabel/all/mindfulness/siamese/wm/test/0/*.npy')
    test_data_list_1 = glob.glob('../../../data/multilabel/all/mindfulness/siamese/wm/test/1/*.npy')


    val_data_list = np.load('../../../data/multilabel/all/mindfulness/siamese/wm/validation/data_siamese_val.npy')


    # In[13]:


    train_dataloader = LSTMTrainDataLoader(
        {0: sample(train_data_list_0, 20000), 1: sample(train_data_list_1, 20000)}, count=10000
    )
    print("Train dataset loaded.")


    # In[14]:


    test_dataloader = LSTMTrainDataLoader(
        {0: sample(test_data_list_0, 5000), 1: sample(test_data_list_1, 5000)}, count=3000
    )
    print("Test dataset loaded.")


    # In[15]:


    val_dataloader = LSTMValDataLoader(
        val_data_list
    )
    print("Validation dataset loaded.")


    train_loader = torch_data.DataLoader(
        train_dataloader,
        batch_size=256, shuffle=True, num_workers=0
    )

    test_loader = torch_data.DataLoader(
        test_dataloader,
        batch_size=128, shuffle=False, num_workers=0
    )

    val_loader = torch_data.DataLoader(
        val_dataloader,
        batch_size=1, shuffle=False, num_workers=0
    )


    is_best = True
    best_score = 0
    best_epoch = 0


    print("Epoch\tTrain Loss\tTest Loss\tTest Accuracy\tValidation Accuracy")
    while curr_epoch <= epochs:

        train_running_loss = train(
            model, train_loader,
            curr_epoch, device, optimizer, criterion
        )

        """
        test_running_loss, test_accuracy = test(
            model, test_loader,
            device, criterion
        )
        """
        val_accuracy = validate(
            model, val_loader,
            device, criterion
        )
        # record all the models that we have had so far.
        train_loss_history.append(
            train_running_loss
        )

        test_acc_history.append(
            0
        )

        test_loss_history.append(
            0
        )

        val_acc_history.append(
            val_accuracy
        )
        # write model to disk.

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(
            state,
            MODEL_PATH_PREFIX + '-{}.'.format(curr_epoch) + MODEL_PATH_EXT
        )

        print('{}\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t'.format(
            curr_epoch,
            train_running_loss,
            0,
            0,
            val_accuracy
        ))
        curr_epoch+=1
