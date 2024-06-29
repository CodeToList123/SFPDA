import torch
import torch.nn as nn


class StudentNetF(nn.Module):
    def __init__(self, basenet_output):
        super(StudentNetF, self).__init__()

        basenet_list = [
            nn.Linear(24, 50),
            # nn.BatchNorm1d(50),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(50, basenet_output)
        ]
        self.base_network = nn.Sequential(*basenet_list)

    def forward(self, x):
        x = self.base_network(x)
        return x


class StudentNetB(nn.Module):
    def __init__(self, feature_dim, basenet_output):
        super(StudentNetB, self).__init__()

        bottleneck_list = [
            nn.Linear(basenet_output, feature_dim),
            # nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)

    def forward(self,x):
        x = self.bottleneck_layer(x)
        return x


class StudentNetC(nn.Module):
    def __init__(self,class_num,feature_dim):
        super(StudentNetC, self).__init__()
        classifier_list = [
            nn.Linear(feature_dim, class_num)
        ]
        self.classifier_layer = nn.Sequential(*classifier_list)

    def forward(self,x):
        x = self.classifier_layer(x)
        return x
