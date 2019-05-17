
class ReducedLeNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1):
        super(ReducedLeNet, self).__init__()
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
