import numpy as np
import torch
import torch.nn as nn
import torchvision
# from torchvision import models
from torch.autograd import Variable
import math
# import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict


class our_netF(nn.Module):
    def __init__(self, basenet_list_output_num):
        super(our_netF, self).__init__()

        basenet_list = [
            nn.Linear(24, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(50, 20),nn.ReLU(),
            nn.Linear(30, basenet_list_output_num)
        ]
        self.base_network = nn.Sequential(*basenet_list)

    def forward(self, x):
        x = self.base_network(x)
        return x


class our_netB(nn.Module):
    def __init__(self, feature_dim, basenet_list_output_num):
        super(our_netB, self).__init__()

        bottleneck_list = [
            nn.Linear(basenet_list_output_num, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        # feature_dim = bottleneck_width

    def forward(self,x):
        x = self.bottleneck_layer(x)
        return x

class our_netC(nn.Module):
    def __init__(self,class_num,feature_dim):
        super(our_netC, self).__init__()
        # basenet_list_output_num = 100
        # feature_dim = 50
        classifier_list = [
            nn.Linear(feature_dim, class_num)
        ]
        self.classifier_layer = nn.Sequential(*classifier_list)
        # feature_dim = bottleneck_width
    def forward(self,x):
        x = self.classifier_layer(x)
        return x
