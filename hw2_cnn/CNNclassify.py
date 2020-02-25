import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from writers import *
from Loaders import *
from CNNClassifier import CNNClassifier 

import matplotlib.pyplot as plt
import os
import cv2
import time

n_epoch = 15
epoch_reporting = 2000
model = CNNClassifier(input_shape = (3, 32, 32))
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) 
writer = SummaryWriter('runs/cifar10_cnn/{}'.format(time.time()))
train_loader = load_training_data()
test_loader = load_testing_data()
model_directory = "./model"
model_save_file = "cifar10_cnn"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model_save_file():
    save_file = "{}/{}.pt".format(model_directory, model_save_file)
    return save_file

# ------ maybe some helper functions -----------
def train(model, num_epoch, continue_training):
    losses = []
    running_loss = 0.0
    max_accuracy = 0
    start_epoch = 0

    model.to(device)

    # Load if we're reloading. 
    if continue_training:
        _, _, start_epoch = load_checkpoint(model, optimizer, get_model_save_file())
        print("Starting off from epoch: {}".format(start_epoch))
        num_epoch += start_epoch

    for epoch in range(start_epoch, num_epoch):
        print("\n\n********************************")
        print("           Epoch {}             ".format(epoch))
        print("********************************")
        
        epoch_cost = 0.0
        for step, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            model.train()                       # set the model in training mode
            
            outputs = model(inputs)                 # 1. Predict
            loss = loss_func(outputs, labels)       # 2. Loss
            optimizer.zero_grad()                   # 3. Zero gradient
            loss.backward()                         # 4. Back Prop
            optimizer.step()                        # 5. Advance
    
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

                if accuracy > max_accuracy: 
                    save_checkpoint(model, optimizer, epoch, get_model_save_file())
                    max_accuracy = accuracy
                        
        epoch_loss = epoch_cost / len(train_loader)
        writer.add_scalar('epoch_loss', epoch_loss, epoch)

def test(model):
    model.eval()  # switch the model to evaluation mode
    model.to(device)
    running_loss = 0.0
    correct = 0
    total = 0

    for data in test_loader:
        #images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
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
    

def save_checkpoint(model, optimizer, epoch, file_name):
    if not os.path.exists(model_directory):
            os.makedirs(model_directory)

    checkpoint = {'model': CNNClassifier(),
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'epoch': epoch}
    
    torch.save(checkpoint, file_name)
    # when get a better model, you can delete the previous one
    # os.remove(......)   # you need to 'import os' first


def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


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
    parser.add_argument('-c', '--continue_training', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("train")
        train(model, n_epoch, args.continue_training)

    elif args.mode == "test":
        model_name = model_save_file
        print("Test: {}".format(args.file))
        print("Using model: {}".format(model_name))

        load_checkpoint(model, optimizer, get_model_save_file())

        if args.file is "all":
            accuracy, loss = test(model)
            print("Accuracy on test set: {}%".format(accuracy))
        else:
            test_single(args.file, model)


if __name__ == "__main__":
    main()
