import torch
import torch.nn as nn
import torch.nn.functional as F

print_sizes = False
#print_sizes = True
def debug_print(msg):
    if print_sizes:
        print(msg)

class Conv2dAuto(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] //2, self.kernel_size[1] //2 )

class CNNClassifier(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), n_hidden = 75, n_output =10):
        super(CNNClassifier, self).__init__()

        # Calculate the size of each image
        n_input = 1
        for shape in input_shape:
            n_input *= shape

        self.conv1 = Conv2dAuto(3, 32, 5)            # conv. layer 1
        self.conv1_b = Conv2dAuto(32, 64, 5)         
        
        self.conv2 = Conv2dAuto(64, 128, 5)           # conv. layer 2
        self.conv2_b = Conv2dAuto(128, 64, 5)            


        self.conv3 = Conv2dAuto(64, 16, 5)
        
        self.pool = nn.MaxPool2d(2, 2)              # max pool layer 1

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_hidden = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        debug_print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_b(x))
        x = self.pool(x)
        debug_print(x.shape)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_b(x))
        x = self.pool(x)
        debug_print(x.shape)

        x = self.pool(F.relu(self.conv3(x)))
        debug_print("Pre fcc {}".format(x.shape))
        
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc_hidden(x))
        x = self.fc3(x)
        debug_print("Forward out: {}".format(x.shape))
        return x

