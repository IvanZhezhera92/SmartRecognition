# https://github.com/anilsathyan7/pytorch-image-classification
from datetime import datetime
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import time, os, sys, copy, argparse
import multiprocessing
from torchsummary import summary
from matplotlib import pyplot as plt
from tqdm import tqdm


# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required = True, type = str, 
    choices = ['scratch', 'finetune', 'transfer', 'Densenet201', 
    'Wide_ResNet-101-2', 'Resnext101_32x8d', 'inception_v3', 'resnet152'], 
    help = "Training mode: finetue/transfer/scratch")
ap.add_argument("--epoch", required = True, type = int, default = 5, help = "Quntity of operations can be integre from 1 till 100")
ap.add_argument("--bs", required = True, type = int, default = 1, help = "Batches size")
ap.add_argument("--train", required = True, type = str, default = './DEKOL/classification/imds_nano_7/train', help = "Train path")  
ap.add_argument("--valid", required = True, type = str, default = './DEKOL/classification/imds_nano_7/val', help = "Valid path")
ap.add_argument("--model", required = True, type = str, default = './DEKOL/classification/model_15.02.22_1.pth', help = "Model path")
args = vars(ap.parse_args())

# Set training mode
bs = int(args["bs"])
train_mode = args["mode"]
epoch_quantity = int(args["epoch"])
train_directory = args["train"]
valid_directory = args["valid"]
PATH = args["model"] 

# Artificial limit 100 epoch
if (int(epoch_quantity) > 100):
    epoch_quantity = 100

num_epochs = epoch_quantity

# Number of classes
train_folders_list = os.listdir(train_directory)
valid_folders_list = os.listdir(valid_directory)

intersection_set = set.intersection(set(train_folders_list), set(valid_folders_list))

if (len(intersection_set) == len(train_folders_list)):
    num_classes = len(train_directory)
    
else:
    print(" [INFO] Uncorrect data in train and valid folders. Check it!")
    print(" [INFO] Must be the same quantity of folders is each folder also same data in pair folders!")
    sys.exit()

# Number of workers
num_cpu = multiprocessing.cpu_count()

# Applying transforms to the data
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size = 256, scale = (0.5, 1)),  
        transforms.RandomRotation(degrees = 25),
        #transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.CenterCrop(size = 224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size = 256),  
        transforms.CenterCrop(size = 224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
 
# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root = train_directory, transform = image_transforms['train']),
    'valid': datasets.ImageFolder(root = valid_directory, transform = image_transforms['valid'])
}
 
# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size = bs, shuffle = True,
                            num_workers = num_cpu, pin_memory = True, drop_last = True),
    
    'valid':data.DataLoader(dataset['valid'], batch_size = bs, shuffle = True,
                            num_workers = num_cpu, pin_memory = True, drop_last = True)
}

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)
 
# Print the train and validation data sizes
print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

if train_mode == 'scratch':
    # Load a custom model - VGG11
    print("\nLoading VGG11 for training from scratch ...\n")
    model_ft = MyVGG11(in_ch = 3, num_classes = num_classes)
    # Set number of epochs to a higher value
    num_epochs = epoch_quantity


elif train_mode == 'finetune':
    # Load a pretrained model - Resnet18
    print("\nLoading resnet18 for finetuning ...\n")
    model_ft = models.resnet18(pretrained = False)

    # Modify fc layers to match num_classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes )


elif train_mode == 'transfer':
    # Load a pretrained model - MobilenetV2
    print("\nLoading mobilenetv2 as feature extractor ...\n")
    model_ft = models.mobilenet_v2(pretrained = False) 

    # Freeze all the required layers (i.e except last conv block and fc layers)
    for params in list(model_ft.parameters())[0:-5]:
        params.requires_grad = False

    # Modify fc layers to match num_classes
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p = 0.2, inplace = False),
        nn.Linear(in_features = num_ftrs, out_features = num_classes, bias = True)
        ) 

elif train_mode == 'Densenet201':  #  <<<<<<<<<<<<<<<<<<< NOT WORK
    # Load a custom model - DenseNet201
    print("\nLoading Densenet201 for training ...\n")
    #model_ft = models.densenet201(pretrained = True, progress = True)
    #num_epochs = epoch_quantity


elif train_mode == 'Wide_ResNet-101-2':
    # Load a custom model - DenseNet201
    print("\nLoading Wide_ResNet-101-2 for training ...\n")
    model_ft = models.wide_resnet101_2(pretrained = False, progress = True)
    num_epochs = epoch_quantity


elif train_mode == 'Resnext101_32x8d':  #  <<<<<<<<<<<<<<<<<<< NOT WORK
    # Load a custom model - resnext101_32x8d
    print("\nLoading Wide_Resnext101_32x8d for training ...\n")
    model_ft = models.resnext101_32x8d(pretrained = True, progress = True)
    num_epochs = epoch_quantity   

elif train_mode == 'inception_v3':  #  <<<<<<<<<<<<<<<<<<< NOT WORK
    # Load a custom model - DenseNet201
    print("\nLoading inception_v3 for training ...\n")
    model_ft = models.inception_v3(pretrained = True, progress = True) #init_weights=True, 
    num_epochs = epoch_quantity


elif train_mode == 'resnet152':  
    # Load a custom model - resnet152
    print("\nLoading resnet152 for training ...\n")
    model_ft = models.resnet152(pretrained = False, progress = True)
    num_epochs = epoch_quantity  


# Transfer the model to GPU
model_ft = model_ft.to(device)

# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad )

summary(model_ft, input_size = (3, 224, 224))
print(model_ft)



# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer 
# Example how to choose optimizer:
# https://pytorch.org/docs/stable/optim.html
#optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)    # was default
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad = False) 
#optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad = False) 
#optimizer_ft = optim.Adamax(model_ft.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 
#optimizer_ft = optim.Adagrad(model_ft.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10) 
#optimizer_ft = optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=1.4, t0=1000000.0, weight_decay=0)
#optimizer_ft = optim.Adadelta(model_ft.parameters(), lr=1.0, rho=1, eps=1e-06, weight_decay=0) 


# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

# Model training routine 
print("\nTraining:-\n")
def train_model(model, criterion, optimizer, scheduler, num_epochs = 30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print(str(datetime.now()))
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking = True)
                labels = labels.to(device, non_blocking = True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()

            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        print(str(datetime.now()))


    time_elapsed = time.time() - since
    print(' >> Training period: {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print(' >> Best validate acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs = num_epochs)

# Save the entire model
print("[INFO] Saving the model: ", PATH)
torch.save(model_ft, PATH)


