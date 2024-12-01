# Model
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, mini=True):
        super(CNNModel, self).__init__()
        self.mini = mini

        #First convulutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride = 2)

        #Second convulutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        #Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        if self.mini:
          self.fc1 = nn.Linear(64 * 16 * 8, 512)
        else:
          self.fc1 = nn.Linear(64 * 64 * 32, 512)
        self.fc2 = nn.Linear(512, 4) #Map to our 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # if mini:
        #   x = x.view(-1, 64 * 64 * 32) #Flatten to 1d vector for fully connected layer
        # else:
        #   x = x.view(-1, 64 * 64 * 32) #Flatten to 1d vector for fully connected layer
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


