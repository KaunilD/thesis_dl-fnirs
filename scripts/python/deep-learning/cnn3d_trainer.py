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


class MultilabelWMLoader(torch_data.Dataset):
    def __init__(self, data_dir, split, num_classes = 1, time_steps = 100 ):
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
            # .reshape((100, 5, 22, 1))
            self.image_list.append(data[0].reshape((self.time_steps, 5, 22, 1)))
            self.label_list.append(data[1])

    def __getitem__(self, index):
        return tuple((self.image_list[index], self.label_list[index]))

    def __len__(self):
        return len(self.image_list)



class CNN3D(nn.Module):
    def __init__(self, num_classes=12, in_planes=20):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(in_planes, 5, kernel_size=(1, 1, 1))
        self.conv2 = nn.Conv3d(5, 2, kernel_size=(1, 1, 1))
        self.conv3 = nn.Conv3d(2, 1, kernel_size=(1, 1, 1))
        self.fc1   = nn.Linear(5, num_classes)
        self.softmax = nn.Sigmoid()


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool3d(out, 2)

        out = self.conv2(out)
        out = F.max_pool3d(out, 2)

        out = self.conv3(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.softmax(out)

        return out


def train(model, dataset_loader, epoch, device, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for i , (data, target) in enumerate(dataset_loader):
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
            labels = labels.float()
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            valid_loss += criterion(outputs, labels).item()

            pred = outputs.ge(0.5)

            correct += 1 if pred.eq(labels.byte()).sum().item() == 12 else 0

    accuracy = 100 * correct / len(dataset_loader.dataset)
    return valid_loss, accuracy


# In[13]:


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 1e-3

    criterion = nn.MultiLabelSoftMarginLoss()

    epochs = 30
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
        split='train', time_steps = 40
    )
    data_shape = train_dataset.__getitem__(0)[0].shape
    train_loader = torch_data.DataLoader( train_dataset, batch_size=1, shuffle=True, num_workers=1)

    val_dataset = MultilabelWMLoader(
        data_dir='C:\\Users\\dhruv\\Development\\git\\thesis_dl-fnirs\\data\\multilabel',
        split='val', time_steps = 40
    )
    val_loader = torch_data.DataLoader( val_dataset, batch_size=1, shuffle=True, num_workers=1)

    model = CNN3D(in_planes = data_shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    model.to(device)

    is_best = True
    best_score = 0
    best_epoch = 0

    print(summary(model, data_shape))

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
