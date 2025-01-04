import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import json

def get_input_args():
    """Setting up the argparse arguments for command line --> https://docs.python.org/3/library/argparse.html
    Inputs: 
        None.
    Outputs: 
        parser.parse_Arg()
    """

    parser = argparse.ArgumentParser(prog = 'Training of a pretrained neural network',
                               description = '''This programm will train a given pretrained neural network 
                               on the flower dataset of torchvision.''')
    
    # Choosing the default path, where all the image_data is
    parser.add_argument('--imgpath', type = str, default = 'flowers', 
                        help = 'Image path, default is "flowers"')

    # Choosing the network architecture
    parser.add_argument('--arch', type = str, default = 'densenet121',
                        choices = ['densenet121', 'resnet50', 'vgg16', 'alexnet'],
                        help = 'Network archtiecture, default is densenet121.')

    # Choosing GPU
    parser.add_argument('--gpu', type = bool, default = False, 
                        help = 'GPU, Default is False')

    # Setting the hyperparamters for training: 
    # Number of Epochs, Learning Rate, Number of Neurons in classifier labels and Dropout

    ## Number of Epochs
    parser.add_argument('--epochs', type = int, default = 10, 
                        help = 'Number of epochs, default is 10')

    ## Learning Rate
    parser.add_argument('--learning_rate', type = float, default = 0.003,
                        help = 'Learning Rate, default is 0.003')

    ## Number of hidden units
    parser.add_argument('--hidden', type = int, default = 512, 
                        help ='Number of hidden units, default is 512.')

    ## Dropout 
    parser.add_argument('--dropout', type = float, default = 0.5, 
                        help = 'Dropout for classifier layers, default is 0.5.')

    ## Saving the checkpoint for later use.
    parser.add_argument('--save_dir', type = str, default = 'flowers', 
                         help = 'Saving directory for checkpoint, default folder is "flowers"')

    #Returning the arguments
    return parser.parse_args()


def freeze_parameter(model):
    """ Freezes the parameters of the model. While training all parameters are held constant.
    
    Input: 
        model: object, the model, which to freeze parameters
    
    Output: 
        model: object, the model with freezed paramters
    """
    ## Freezing the pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model


def check_hidden(in_features, hidden, out_features):
    """Checking, if number of hidden units make sense.
    
    Input:
        in_features, hidden, out_features: integer
    
    Output:
        correct_hidden: integer, corrected hidden units
    """
    
    hidden_check = out_features < hidden < in_features
    if hidden_check:
        print(f"The hidden layer will have {hidden} hidden units.")
        correct_hidden = hidden
    else:
        print(f"The number of hidden units {hidden} doesn't fit to the models classifier.")
        print(f"The model needs a number between {out_features} and {in_features}.")
        print(f"The default of 512 hidden units will be used.")
        correct_hidden = 512
        
    return correct_hidden


def new_classifier(in_features, hidden, out_features, dropout):
    """ Creates a new classifier for models
    
    Input: 
        in_features, hidden, out_features: integer
        dropout: float
            
    Output: 
        classifier: object for the model
    """
    
    classifier = nn.Sequential(nn.Linear(in_features, hidden),
                               nn.ReLU(),
                               nn.Dropout(p = dropout),
                               nn.Linear(hidden, out_features),
                               nn.LogSoftmax(dim = 1))
    return classifier


"""Main root of the program"""

# Getting the input arguments for the training
in_arg = get_input_args()


# Preparing the data for training

## Preparing the directories
data_dir = in_arg.imgpath
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

## Preparing the transformations of the datasets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = valid_transforms

## Loading the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

## Defining the DataLoaders with the given datasets
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 64)


# Loading Label Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
flower_classes = len(cat_to_name)

# Building the model
if in_arg.arch == 'densenet121':
    print(f"Loading Model: {in_arg.arch}")
    model = models.densenet121(pretrained = True)
    model = freeze_parameter(model)
    print(f"Model {in_arg.arch} loaded \n") 
    """ Original Classifier
          (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    """
    in_features = model.classifier.in_features
    
    # Checking, if the number of hidden units fit the models
    hidden = check_hidden(in_features, in_arg.hidden, flower_classes)
    
    model.classifier = new_classifier(in_features, hidden, flower_classes, in_arg.dropout)
    
    print(f"Classifier with {in_features} input features, "
          f"{in_arg.hidden} hidden units and {flower_classes} output features implemented.\n")
    print(f"New Classififer: \n {model.classifier}")
    
elif arch == 'resnet50':
    print(f"Loading Model: {in_arg.arch}")
    model = models.resnet50(pretrained = True)
    model = freeze_parameter(model)
    print(f"Model {in_arg.arch} loaded \n") 
    """ Original Classifier:
          (fc): Linear(in_features=2048, out_features=1000, bias=True)
    """

    in_features = model.classifier.in_features
    
    # Checking, if the number of hidden units fits to the model
    hidden = check_hidden(in_features, in_arg.hidden, flower_classes)
    
    model.classifier = new_classifier(in_features, hidden, flower_classes, in_arg.dropout)
          
    print(f"Classifier with {in_features} input features, "
          f"{in_arg.hidden} hidden units and {flower_classes} output features implemented.\n")
    print(f"New Classififer: \n {model.classifier}")
    
elif arch == 'vgg16':
    print(f"Loading Model: {in_arg.arch}")
    model = models.vgg16(pretrained = True)
    model = freeze_parameter(model)
    print(f"Model {in_arg.arch} loaded \n") 
    
    """ Original Classifier
        (classifier): Sequential(
            (0): Linear(in_features=25088, out_features=4096, bias=True)
            (1): ReLU(inplace=True)
            (2): Dropout(p=0.5, inplace=False)
            (3): Linear(in_features=4096, out_features=4096, bias=True)
            (4): ReLU(inplace=True)
            (5): Dropout(p=0.5, inplace=False)
            (6): Linear(in_features=4096, out_features=1000, bias=True)
    """
    
    in_features = model.classifier.in_features
    
    # Checking, if the number of hidden units fits to the model
    hidden = check_hidden(in_features, in_arg.hidden, flower_classes)
    
    model.classifier = new_classifier(in_features, hidden, flower_classes, in_arg.dropout)
    
    print(f"Classifier with {in_features} input features, "
          f"{in_arg.hidden} hidden units and {flower_classes} output features implemented.\n")
    print(f"New Classififer: \n {model.classifier}")
    
elif arch == 'alexnet':
    print(f"Loading Model: {in_arg.arch}")
    model = models.alexnet(pretrained = True)
    model = freeze_parameter(model)
    print(f"Model {in_arg.arch} loaded \n") 
    """ Original Classifier
         (classifier): Sequential(
            (0): Dropout(p=0.5, inplace=False)
            (1): Linear(in_features=9216, out_features=4096, bias=True)
            (2): ReLU(inplace=True)
            (3): Dropout(p=0.5, inplace=False)
            (4): Linear(in_features=4096, out_features=4096, bias=True)
            (5): ReLU(inplace=True)
            (6): Linear(in_features=4096, out_features=1000, bias=True)
    """
    
    in_features = model.classifier.in_features
    
    # Checking, if the number of hidden units fits to the model
    hidden = check_hidden(in_features, in_arg.hidden, flower_classes)
    
    model.classifier = new_classifier(in_features, hidden, flower_classes, in_arg.dropout)
    
    print(f"Classifier with {in_features} input features, "
          f"{in_arg.hidden} hidden units and {flower_classes} output features implemented.\n")
    print(f"New Classififer: \n {model.classifier}")

print(f"--------------------------")
                                     
    
## Defining the loss function
criterion = nn.NLLLoss()

## Defining the optimizer for the gradient descent
optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.learning_rate)

print(f"The model is loaded and built. Loss-Function and Optimizer are set.")
print(f"--------------------------")


# Checking if GPU is chosen and available, moving model to GPU or CPU
if (torch.cuda.is_available() and in_arg.gpu):
    device = torch.device("cuda")
    print(f"GPU will be used")
elif not torch.cuda.is_available():
    device = torch.device("cpu")
    print(f"GPU is not available. CPU will be used." if in_arg.gpu else f"CPU will be used")
          
## Loading model to device    
model.to(device)
          
print(f"--------------------------")

          
# Setup for training
          
## Setting the epochs for the training
epochs = in_arg.epochs

## running_loss is the counter for the loss in every training batch
running_loss = 0 

## steps counts the number of batches trained (64 images per batch)
steps = 0 

## Output of validation testing during the training
print_every = 1


# Training the model
print(f"Starting the training.")
print(f"It will run {epochs} epoch(s).")
print(f"The running loss, validation loss and the validation accuracy is printed after every {print_every} batches.")
print(f"--------------------------")

for e in range(epochs):
    """Epochs"""
    
    for images, labels in trainloader:
        """using the training dataset for training"""
        
        ## Increasing step per batch of 64 images
        steps +=1
       
        ## Moving images- and labels-tensor to device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)
        
        ## Training of the network with the given images
        optimizer.zero_grad()
        log_ps = model(images) #forward pass
        loss = criterion(log_ps, labels) #calculating the loss
        loss.backward() #backpropagation
        optimizer.step() #updating the weights and biases
        
        ## Adding the loss to running_loss to get an average loss per print_every
        running_loss += loss.item()
        print(f"Training-batch {steps} done")
        
        if steps % print_every == 0:
            """After print_every batches a validation is made- with Train-, Validation-Loss und Validation-Accuracy"""
            
            print(f"Starting Validation with validation-set")
            validation_loss = 0
            accuracy = 0
            
            ## Setting the model to evaluation mode without dropout
            model.eval()
            
            with torch.no_grad():
                """running the model without autograd-engine to be faster in validation
                    - no backpropagation needed"""
                
                for images, labels in validloader:
                    """using the validloader for the validation set"""
                    
                    #moving the images- and labels-tensor to device (GPU or CPU)
                    images, labels = images.to(device), labels.to(device)
                    
                    #validate the validation-images through the model and calculating the loss.
                    log_ps = model(images)
                    batch_loss = criterion(log_ps, labels)
                    
                    #adding the batch_loss to validation loss
                    validation_loss += batch_loss.item()
                    
                    #calculating the accuracy
                    probability = torch.exp(log_ps)
                    top_p, top_class = probability.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                print(f"Epochs: {e + 1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {100*accuracy/len(validloader):.2f}%")
                
                #setting back the running_loss
                running_loss = 0
             
            #setting the model back to training mode with given dropout
            model.train()
            
print(f"...")
print(f"The training is finished.")
print(f"--------------------------")


# Validating the model with the Test-Set
print(f"Starting validation with the test-set.")
      
## Setting the model to evaluation mode without dropout
model.eval()

test_loss = 0
test_accuracy = 0
test_batch = 0

with torch.no_grad():
    """running the model without autograd-engine to be faster in validation
        - no backpropagation needed"""
                
    for images, labels in testloader:
        """using the testloader for the test-set"""
                    
        #moving the images- and labels-tensor to device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)
                    
        #validate the validation-images through the model and calculating the loss.
        log_ps = model(images)
        batch_loss = criterion(log_ps, labels)
                    
        #adding the batch_loss to validation loss
        test_loss += batch_loss.item()
                    
        #calculating the accuracy
        probability = torch.exp(log_ps)
        top_p, top_class = probability.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        test_batch += 1
        print(f"Testbatch {test_batch} done.")
        
print(f"Test Loss: {test_loss/len(testloader):.3f}.. "
      f"Test Accuracy: {100*test_accuracy/len(testloader):.2f}%")
    
## Setting the model back to training mode    
model.train()
          
print(f"...")
print(f"The model was trained with {e+1} epochs and a training loss of {running_loss/print_every:.3f}.")
print(f"The model is validated with {100*test_accuracy/len(testloader):.2f}% Accuracy.")
print(f"--------------------------")


# Saving the model to a checkpoint
print(f"The model will be saved now.")

## Building the checkpoint
model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'model_state_dict': model.state_dict(),
              'model_arch_name': in_arg.arch,
              'optimizer_state_dict': optimizer.state_dict(),
              'model.class_to_idx': model.class_to_idx,
              'trained_epochs': e+1,
              'validated_loss': (validation_loss/len(validloader)),
              'validated_accuracy': (accuracy/len(validloader)),
              'hidden_units': in_arg.hidden,
              'dropout': in_arg.dropout,
              'learning_rate': in_arg.learning_rate
             }

## Saving the checkpoint as CommandLineEdition
arch_of_model = 'checkpoint_'+ in_arg.arch + '.pth'
torch.save(checkpoint, arch_of_model)
print(f"The model was saved in {arch_of_model}")
print(f"--------------------------")