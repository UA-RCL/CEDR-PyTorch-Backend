import sys
sys.path.append('../../torch_2.2/lib/python3.10/site-packages/cedr_backend-0.0.0-py3.10-linux-x86_64.egg/')

import torch
import torch.profiler
import torch.nn as nn
import torch.optim as optim
import cedr_backend

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=2352, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

def main():
    seed = 1
    torch.manual_seed(seed)

    model = SimpleCNN()
    input_tensor = torch.randn(1, 1, 28, 28)

    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    print(output)

main()
