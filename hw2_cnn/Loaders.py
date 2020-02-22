import torch
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable

from PIL import Image

# ----------------- prepare training data -----------------------
def load_training_data():
    train_data = torchvision.datasets.CIFAR10(
        root='./data.cifar10',                          # location of the dataset
        train=True,                                     # this is training data
        # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
        transform=torchvision.transforms.ToTensor(),    
        download=True # if you haven't had the dataset, this will automatically download it for you
    )
   
    return Data.DataLoader(dataset=train_data, batch_size=4,
         shuffle=True, num_workers=2)

# ----------------- prepare testing data -----------------------
def load_testing_data():
    test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, 
                transform=torchvision.transforms.ToTensor())
        
    return Data.DataLoader(dataset=test_data,  batch_size=4, shuffle=False, 
        num_workers=2)

# ----------------- load an image -----------------------------
def image_loader(image_name):
    import torchvision.transforms.functional as TF
    
    image = Image.open(image_name)
    image = image.convert("RGB")
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return Variable(x)

