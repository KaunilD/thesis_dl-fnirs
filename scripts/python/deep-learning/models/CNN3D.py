import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, num_classes=12, in_planes=100):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(in_planes, 25, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(25, 5, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv3d(25, 1, kernel_size=2, stride=1, padding=1, bias=False)
        self.fc1   = nn.Linear(44, num_classes)
        self.softmax = nn.Sigmoid()


    def forward(self, x):

        out = self.conv1(x)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.softmax(out)

        return out
