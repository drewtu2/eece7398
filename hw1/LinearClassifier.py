import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), n_hidden = 75, n_output =10):
        super(LinearClassifier, self).__init__()

        # Calculate the size of each image
        n_input = 1
        for shape in input_shape:
            n_input *= shape

        self.hidden = nn.Linear(n_input, n_hidden)    # hidden layer
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)      # linear output
        
        self.fc = nn.Linear(n_input, n_output)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        #x = self.fc(x)
        return x
