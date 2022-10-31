import numpy as np
import json
import time
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

from workspace_utils import active_session

torch.cuda.is_available()

# Data folders
train_dir = '/data/yolact/DEKOL/classification/imds_nano_3/train'
valid_dir = '/data/yolact/DEKOL/classification/imds_nano_3/val'
test_dir = '/data/yolact/DEKOL/classification/imds_nano_3/test'

# Dataset controls
image_size = 224 # Image size in pixels
reduction = 255 # Image reduction to smaller edge 
norm_means = [0.485, 0.456, 0.406] # Normalized means of the images
norm_std = [0.229, 0.224, 0.225] # Normalized standard deviations of the images
rotation = 45 # Range of degrees for rotation
batch_size = 64 # Number of images used in a single pass
shuffle = True # Randomize image selection for a batch

# Environment controls
# Choose if run in CPU or GPU enabled hardware
#devices = ['cpu', 'cuda']
#current_device = devices[0]
# or use CUDA if available
current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(current_device)

# TODO: Define your transforms for the training, validation, and testing sets

# Create transforms pipelines to run/apply them in sequence on image data
# Next convert image data to sensors and normalize it to make backpropagation more stable
train_transforms = transforms.Compose([transforms.RandomResizedCrop(image_size),
                                       transforms.RandomRotation(rotation),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(norm_means, norm_std)])

valid_transforms = transforms.Compose([transforms.Resize(reduction),
                                      transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_means, norm_std)])

test_transforms = transforms.Compose([transforms.Resize(reduction),
                                      transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_means, norm_std)])

data_transforms = {"training": train_transforms, 
                   "validation": valid_transforms, 
                   "testing": test_transforms}

# TODO: Load the datasets with ImageFolder

# Load and transform image data
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

image_datasets = {"training": train_data, 
                   "validation": valid_data, 
                   "testing": test_data}

# TODO: Using the image datasets and the transforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

dataloaders = {"training": trainloader, 
               "validation": validloader,
               "testing": testloader}


"""with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Check the contents of cat_to_name
for i, key in enumerate(cat_to_name.keys()):
    print(key, '\t->', cat_to_name[key])
    if i == 10:
        break
 
print("There are {} image categories.".format(len(cat_to_name)))"""

cat_to_name = {"0": "0", "1": "1", "2": "2","3": "3","4": "4"}


model_name = 'vgg16'
model = models.vgg16(pretrained = True)
model.name = model_name

# Freeze model parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):
        super().__init__()
        
        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add hidden layers
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Add dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))        

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x

# Replace the classifier part of the pre-trained model with fully-connected layers

input_size = 25088
output_size = 5
hidden_layers = [4096, 1024]
drop_out = 0.2

model.classifier = Classifier(input_size, output_size, hidden_layers, drop_out)

# Define the loss function
criterion = nn.NLLLoss()

# Define weights optimizer (backpropagation with gradient descent)
# Only train the classifier parameters, feature parameters are frozen
# Set the learning rate as lr=0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

# Move the network and data to GPU or CPU
model.to(current_device)


# A function used for validation and testing
def testClassifier(model, criterion, testloader, current_device):
    
    # Move the network and data to current hardware config (GPU or CPU)
    model.to(current_device)
        
    test_loss = 0
    accuracy = 0
        
    # Looping through images, get a batch size of images on each loop
    for inputs, labels in testloader:

        # Move input and label tensors to the default device
        inputs, labels = inputs.to(current_device), labels.to(current_device)

        # Forward pass, then backward pass, then update weights
        log_ps = model.forward(inputs)
        batch_loss = criterion(log_ps, labels)
        test_loss += batch_loss.item()

        # Convert to softmax distribution
        ps = torch.exp(log_ps)
        
        # Compare highest prob predicted class with labels
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        # Calculate accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return test_loss, accuracy


# A function used for training (and tests with different model hyperparameters)
def trainClassifier(model, epochs_no, criterion, optimizer, trainloader, validloader, current_device):
    
    # Move the network and data to current hardware config (GPU or CPU)
    model.to(current_device)
    
    epochs = epochs_no
    steps = 0
    print_every = 1
    running_loss = 0

    # Looping through epochs, each epoch is a full pass through the network
    for epoch in range(epochs):
        
        # Switch to the train mode
        model.train()

        # Looping through images, get a batch size of images on each loop
        for inputs, labels in trainloader:

            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(current_device), labels.to(current_device)

            # Clear the gradients, so they do not accumulate
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Track the loss and accuracy on the validation set to determine the best hyperparameters
        if steps % print_every == 0:

            # Put in evaluation mode
            model.eval()

            # Turn off gradients for validation, save memory and computations
            with torch.no_grad():

                # Validate model
                test_loss, accuracy = testClassifier(model, criterion, validloader, current_device)
                
            train_loss = running_loss/print_every
            valid_loss = test_loss/len(validloader)
            valid_accuracy = accuracy/len(validloader)

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss:.3f}.. "
                  f"Test loss: {valid_loss:.3f}.. "
                  f"Test accuracy: {valid_accuracy:.3f}")

            running_loss = 0
            
            # Switch back to the train mode
            model.train()
                
    # Return last metrics
    return train_loss, valid_loss, valid_accuracy


# Train the classifier layers using backpropagation using the pre-trained network to get the features

# Test 1 - first run - the baseline
# Hyperparameters
drop_out = 0.2
# Optimizer type and learning rate
learning_rate = 0.003
# Define weights optimizer (backpropagation with gradient descent)
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
# No of epochs
epochs_no = 5

# To ensure the GPU workspace session is not terminated
with active_session():
    # Train and validate the neural network classifier
    train_loss, valid_loss, valid_accuracy = trainClassifier(
        model, epochs_no, criterion, optimizer, trainloader, 
        validloader, current_device)

# Display final summary
print("Final result \n",
      f"Train loss: {train_loss:.3f}.. \n",
      f"Test loss: {valid_loss:.3f}.. \n",
      f"Test accuracy: {valid_accuracy:.3f}")

# Save the checkpoint (the function is defined below)

#model_name = 'vgg16'
filename = saveCheckpoint(model)


# Test 2 - decrease learning rate
drop_out = 0.2
learning_rate = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
epochs_no = 5

with active_session():
    train_loss, valid_loss, valid_accuracy = trainClassifier(
        model, epochs_no, criterion, optimizer, trainloader, 
        validloader, current_device)

print("Final result \n",
      f"Train loss: {train_loss:.3f}.. \n",
      f"Test loss: {valid_loss:.3f}.. \n",
      f"Test accuracy: {valid_accuracy:.3f}")

filename = saveCheckpoint(model)