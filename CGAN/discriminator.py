import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, output_size, condition_size, hidden_size=256 * 16):
        super(Discriminator, self).__init__()
        self.output_size = output_size
        self.condition_size = condition_size  # 移除生成数量的维度
        self.fc1 = nn.Linear(output_size + self.condition_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
