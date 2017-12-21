import torch
import torch.nn as nn
from torch.autograd import Variable

class Stylenet(nn.Module):
    def __init__(self, num_classes=128):
        super(Stylenet, self).__init__()
        self.conv_a = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv_b = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv_c = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
        self.conv_d = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))
        self.conv_e = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))
        self.conv_f = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.conv_g = nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)
        self.maxpool = nn.MaxPool2d((4, 4), (4, 4))
        self.batchnorm_a = nn.BatchNorm2d(64, 0.001, 0.9, True)
        self.batchnorm_b = nn.BatchNorm2d(128, 0.001, 0.9, True)
        self.batchnorm_c = nn.BatchNorm2d(256, 0.001, 0.9, True)
        self.fc = nn.Linear(3072, num_classes)

    def forward(self, input):

        x = self.relu(self.conv_a(input))
        x = self.relu(self.conv_b(x))
        x = self.maxpool(self.dropout(x))
        x = self.batchnorm_a(x)
        x = self.relu(self.conv_c(x))
        x = self.relu(self.conv_d(x))
        x = self.maxpool(self.dropout(x))
        x = self.batchnorm_b(x)
        x = self.relu(self.conv_e(x))
        x = self.relu(self.conv_f(x))
        x = self.maxpool(self.dropout(x))
        x = self.batchnorm_c(x)
        x = self.relu(self.conv_g(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FENet(nn.Module):
    def __init__(self, num_features=128):
        super(FENet, self).__init__()
        self.conv_a = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv_b = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv_c = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
        self.conv_d = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))
        self.conv_e = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))
        self.conv_f = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.conv_g = nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)
        self.maxpool = nn.MaxPool2d((4, 4), (4, 4))
        self.batchnorm_a = nn.BatchNorm2d(64, 0.001, 0.9, True)
        self.batchnorm_b = nn.BatchNorm2d(128, 0.001, 0.9, True)
        self.batchnorm_c = nn.BatchNorm2d(256, 0.001, 0.9, True)
        self.fc = nn.Linear(3072, num_features)

    def forward(self, input):

        x = self.relu(self.conv_a(input))
        x = self.relu(self.conv_b(x))
        x = self.maxpool(self.dropout(x))
        x = self.batchnorm_a(x)
        x = self.relu(self.conv_c(x))
        x = self.relu(self.conv_d(x))
        x = self.maxpool(self.dropout(x))
        x = self.batchnorm_b(x)
        x = self.relu(self.conv_e(x))
        x = self.relu(self.conv_f(x))
        x = self.maxpool(self.dropout(x))
        x = self.batchnorm_c(x)
        x = self.relu(self.conv_g(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ClassificationNet(nn.Module):
    def __init__(self, num_in_features=128, num_out_classes=124):
        super(ClassificationNet, self).__init__()
        self.batchnorm = nn.BatchNorm2d(128, 0.001, 0.9, True)
        self.relu = nn.ReLU(inplace=True)
        self.hidden = nn.Linear(num_in_features, num_in_features)
        self.classifier = nn.Linear(num_in_features, num_out_classes)

    def forward(self, input):
        x = self.batchnorm(input)
        x = self.hidden(self.relu(x))
        x = self.classifier(x)

        return x
