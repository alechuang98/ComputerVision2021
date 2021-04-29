import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 0),  # (6, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     # (6, 12, 12)
            
            nn.Conv2d(6, 16, 5, 1, 0), # (16, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     # (16, 4, 4)
        )

        self.linear = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

    def name(self):
        return 'ConvNet'

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 0),  # (32, 26, 26)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 0), # (32, 24, 24)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 5, 2, 2, padding_mode='zeros'), # (32, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4),

            nn.Conv2d(32, 64, 3, 1, 0), # (64, 10, 10)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 0), # (64, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 2, 2, padding_mode='zeros'), # (64, 4, 4)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.4),

            nn.Conv2d(64, 128, 4, 1, 0), # (128, 1, 1)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.4)
        )
        self.linear = nn.Sequential(
            nn.Linear(128, 10),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

    def name(self):
        return "MyNet"

