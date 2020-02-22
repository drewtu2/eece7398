import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from writers import *
from LinearClassifier import LinearClassifier

from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import time

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

n_epoch = 15
epoch_reporting = 500
model = LinearClassifier(input_shape = (3, 32, 32))
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) 
writer = SummaryWriter('runs/cifar10_svm/{}'.format(time.time()))
train_loader = load_training_data()
test_loader = load_testing_data()
model_save_file = "cifar10_svm"


# ------ maybe some helper functions -----------
def train(model, num_epoch):
    losses = []
    running_loss = 0.0
    for epoch in range(num_epoch):
        print("\n\n********************************")
        print("           Epoch {}             ".format(epoch + 1))
        print("********************************")
        
        epoch_cost = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            model.train()                       # set the model in training mode
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            epoch_cost += loss.item()
            
            # Every n epochs...
            if step % epoch_reporting == 0:
                training_loss = running_loss / epoch_reporting
                accuracy, test_loss = test(model)
                print('Test set: Epoch[{}]:Step[{}] Accuracy: {}% ......'.format(epoch, step, accuracy))
                writer.add_scalar('training loss', training_loss, 
                        epoch * len(train_loader) + step)
                writer.add_scalar('test loss', test_loss, 
                        epoch * len(train_loader) + step)
                writer.add_scalar('test accuracy', accuracy, 
                        epoch * len(train_loader) + step)
                
                running_loss = 0.0
                        
        epoch_loss = epoch_cost / len(train_loader)
        writer.add_scalar('epoch_loss', epoch_loss, epoch)
    save_model(model, model_save_file)

def test(model):
    model.eval()  # switch the model to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0

    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        running_loss += loss_func(outputs, labels).item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss = running_loss / len(test_loader)
    return accuracy, test_loss


def test_single(image_name, model):
    img = image_loader(image_name)
    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[0]]))
    

def image_loader(image_name):
    import torchvision.transforms.functional as TF
    
    image = Image.open(image_name)
    image = image.convert("RGB")
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return Variable(x)

#def image_loader(image_name, imsize = 256):
#    """load image, returns cuda tensor"""
#    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
#    #image = cv2.imread(image_name)
#    image = Image.open(image_name)
#    image = loader(image).float()
#    image = Variable(image, requires_grad=True)
#    #image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#    return image #assumes that you're using GPU

def save_model(model, file_name):
    directory = "./model"
    if not os.path.exists(directory):
            os.makedirs(directory)
    save_file = "{}/{}.pt".format(directory, file_name)
    torch.save(model.state_dict(), save_file)
    # when get a better model, you can delete the previous one
    # os.remove(......)   # you need to 'import os' first


def load_model(model, model_name):
    model.load_state_dict(torch.load("./model/{}.pt".format(model_name)))



import argparse
def main():
    ## parse the argument e.x. >>> python3 classify.py train
    parser = argparse.ArgumentParser(description='xxx')
    parser.add_argument('mode', type=str, 
            default="test", 
            help="working in training/testing mode")
    parser.add_argument('file', type=str,
            default="all",
            help="file to test",
            nargs="?")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("train")
        train(model, n_epoch)
    elif args.mode == "test":
        model_name = model_save_file
        print("Test: {}".format(args.file))
        print("Using model: {}".format(model_name))
        load_model(model, model_name)

        if args.file is "all":
            accuracy, loss = test(model)
            print("Accuracy on test set: {}%".format(accuracy))
        else:
            test_single(args.file, model)


if __name__ == "__main__":
    main()
