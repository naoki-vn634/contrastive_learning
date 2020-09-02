import torch.nn as nn
import torchvision.models as models


class ContrastiveResnetModel(nn.Module):
    def __init__(self, mode=0, num_channels=3, hidden_size=2048, out_dim=2):
        super(ContrastiveResnetModel, self).__init__()
        resnset = models.resnet50(pretrained=True)
        self.mode = mode
        self.resnet = nn.Sequential(*list(resnet.children()[:-1]))
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d()
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        hidden = self.resnet(x)
        h = self.relu(self.fc1(hidden))
        h = self.bn(h)
        out = self.fc2(h)
        return out, hidden
