import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), n_hidden = 75, n_output =10):
        super(CNNClassifier, self).__init__()

        # Calculate the size of each image
        n_input = 1
        for shape in input_shape:
            n_input *= shape

        self.conv1 = nn.Conv2d(3, 32, 5)            # conv. layer 1
        #self.conv2 = nn.Conv2d(32, 16, 5, padding=(5, 5))           # conv. layer 2
        #self.conv2 = nn.Conv2d(32, 16, 5, padding=(5//2, 5//2))           # conv. layer 2
        self.conv2 = nn.Conv2d(32, 16, 5)           # conv. layer 2
        self.conv3 = nn.Conv2d(16, 16, 5, padding=(5//2, 5//2))           # conv. layer 3
        
        self.pool = nn.MaxPool2d(2, 2)              # max pool layer 1

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_hidden = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_hidden(x))
        x = self.fc3(x)
        
        return x
