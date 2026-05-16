import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessCNN(nn.Module):

    def __init__(self, num_moves):
        super().__init__()

        # Common trunk
        self.conv1 = nn.Conv2d(13, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)  
        
        # Policy head
        self.fc_policy = nn.Linear(128 * 8 * 8, 256)
        self.out_policy = nn.Linear(256, num_moves)
        
        # Value head 
        self.fc_value = nn.Linear(128 * 8 * 8, 64)
        self.out_value = nn.Linear(64, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        flat = x.view(x.size(0), -1)
        
        policy = F.relu(self.fc_policy(flat))
        policy = self.out_policy(policy)
        
        value = F.relu(self.fc_value(flat))
        value = torch.tanh(self.out_value(value))
        
        return policy, value