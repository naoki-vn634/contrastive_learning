import torch.nn as nn
import torchvision.models as models


class ContrastiveResnetModel(nn.Module):
    def __init__(self, mode=0, num_channels=3, hidden_size=2048, out_dim=2, hidden=0):
        super(ContrastiveResnetModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.mode = mode
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(hidden_size, out_dim)
        if not hidden == 0:
            self.fc4 = nn.Linear(128, out_dim)

    def forward(self, x):
        hidden = self.resnet(x).view(x.size()[0], -1)
        h = self.relu(self.bn(self.fc1(hidden)))
        out = self.fc2(h)
        return out, hidden
