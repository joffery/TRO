import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class FeatureNet(nn.Module):
    def __init__(self, opt):
        super(FeatureNet, self).__init__()
        nx, nh = opt.nx, opt.nh

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh, nh)
        self.fc3 = nn.Linear(nh, nh)
        self.fc4 = nn.Linear(nh, nh)
        self.fc_final = nn.Linear(nh, nh)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, _ = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.fc1(x))

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class PredNet(nn.Module):
    def __init__(self, opt):
        super(PredNet, self).__init__()
        nh, nc = opt.nh, opt.nc
        self.fc3 = nn.Linear(nh, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)
        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()

    def forward(self, x, return_feature=False):
        re = x.dim() == 3
        if re:
            T, B, _ = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x_feat = F.relu(self.bn4(self.fc4(x)))

        # Classification
        x = self.fc_final(x_feat)
        x_softmax = F.softmax(x, dim=1)
        x = torch.log(x_softmax + 1e-4)

        if re:
            x = x.reshape(T, B, -1)
            x_softmax = x_softmax.reshape(T, B, -1)

        if return_feature:
            return x, x_feat
        else:
            return x