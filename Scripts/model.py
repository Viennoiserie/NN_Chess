import torch.nn as nn
import torch.nn.functional as F

class ChessCNN(nn.Module):

    def __init__(self, num_moves):

        super().__init__()

        self.conv1 = nn.Conv2d(13, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_moves)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)