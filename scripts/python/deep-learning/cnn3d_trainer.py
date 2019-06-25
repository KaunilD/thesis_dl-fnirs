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
import glob
from torchsummary import summary
from sklearn.metrics import confusion_matrix

import plotly.io as pio
import plotly.plotly as py
import plotly.graph_objs as go
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
                data[0].reshape((1, self.time_steps, 5, 44))
            )
            self.label_list.append(data[1])

    def __getitem__(self, index):

        return (self.image_list[index], self.label_list[index])

    def __len__(self):
        return len(self.image_list)



class CNN3D(nn.Module):
    def __init__(self, num_classes=9, in_planes=1):
        super(CNN3D, self).__init__()
        # N, C, D, H, W = 1, 1, 160, 5, 22
        self.conv1 = nn.Conv3d(in_channels=in_planes, out_channels=5, kernel_size=(65, 1, 2))
        self.bn1 = nn.BatchNorm3d(5)
        self.conv2 = nn.Conv3d(in_channels=5, out_channels=1, kernel_size=(1, 1, 1))
        self.dropout1 = nn.Dropout(p=0.6)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc1   = nn.Linear(460 , 3)
        self.softmax = nn.Sigmoid()


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.max_pool3d(out, 2)


        out = self.conv2(out)
        out = F.max_pool3d(out, 2)


        out = out.view(out.size(0), -1)

        if self.training:
            out = self.dropout1(out)


        out = self.fc1(out)

        if self.training:
            out = self.dropout2(out)

        out = self.softmax(out)

        return out


def train(model, dataset_loader, epoch, device, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for data, target in dataset_loader:


        inputs, labels = data, target
        inputs = inputs.float()
        labels = labels.float()
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
            labels = labels.float()
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

# In[13]:


if __name__ == '__main__':
    torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 1e-3

    criterion = nn.MSELoss(reduction='sum')

    epochs = 60
    curr_epoch = 0

    criterion.to(device)

    loss_history = []
    acc_history = []

    best_accuracy = 0.0
    best_model = -1

    MODEL_PATH_PREFIX = 'model-cnn3d-epoch'
    MODEL_PATH_EXT = 'pth'

    train_dataset = MultilabelWMLoader(
        data_dir='C:\\Users\\dhruv\\Development\\git\\thesis_dl-fnirs\\data\\multilabel',
        split='train', time_steps = 250
        )

    data_shape = train_dataset[0][0].shape

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=8, shuffle=True, num_workers=1
        )

    val_dataset = MultilabelWMLoader(
        data_dir='C:\\Users\\dhruv\\Development\\git\\thesis_dl-fnirs\\data\\multilabel',
        split='val', time_steps = 250
        )

    val_loader = torch_data.DataLoader(
        val_dataset,
        batch_size=8, shuffle=True, num_workers=1
        )

    model = CNN3D(in_planes = data_shape[0])

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate,
        momentum=0.7
        )
    model.to(device)

    is_best = True
    best_score = 0
    best_epoch = 0

    print("************************ MODEL SUMMARY *************************\n")
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

    random_x = np.linspace(1, epochs, epochs)
    # Create traces
    trace0 = go.Scatter(
        x = random_x,
        y = [i[0] for i in loss_history],
        mode = 'lines+markers',
        name = 'train-loss'
    )

    trace1 = go.Scatter(
        x = random_x,
        y = [i[1] for i in loss_history],
        mode = 'lines+markers',
        name = 'validation-loss'
    )
    data = [trace0, trace1]
    layout = go.Layout(
        title= 'TRAINING SNAPSHOT',
        hovermode= 'closest',
        xaxis= dict(
            title= 'EPOCHS',
            ticklen= 5,
            zeroline= False,
            gridwidth= 2,
        ),
        yaxis=dict(
            title= 'LOSS',
            ticklen= 5,
            gridwidth= 2,
        ),)
    fig = go.Figure(data, layout=layout)
    py.iplot(fig)
    pio.write_image(fig, 'newplot.png')
