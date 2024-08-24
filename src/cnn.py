# -*- coding: utf-8 -*-
"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""
import torch
import torch.nn as nn
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()  #goi ham khoi tao lop cha
        self.conv1_maxpool1 = self.make_block(in_channels=3, out_channels=16)
        self.conv2_maxpool2 = self.make_block(in_channels=16, out_channels=32)
        self.conv3_maxpool3 = self.make_block(in_channels=32, out_channels=64)
        self.conv4_maxpool4 = self.make_block(in_channels=64, out_channels=64)
        self.conv5_maxpool5 = self.make_block(in_channels=64, out_channels=64)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=3136, out_features=1024),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_classes),
        )
    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1_maxpool1(x)
        x = self.conv2_maxpool2(x)
        x = self.conv3_maxpool3(x)
        x = self.conv4_maxpool4(x)
        x = self.conv5_maxpool5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN()
    model.train()
    model.to(device)
    summary(model, (3, 224, 224))
    # sample_input = torch.rand(2, 3, 224, 224)
    # result = model(sample_input)
    # print(result.shape)