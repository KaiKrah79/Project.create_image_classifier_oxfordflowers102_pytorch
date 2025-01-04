import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import json
from PIL import Image
import numpy as np


def get_input_args():
    """Setting up the argparse arguments for command line --> https://docs.python.org/3/library/argparse.html
    
    Inputs: 
        None.
    
    Outputs: 
        parser.parse_Arg()"""

    parser = argparse.ArgumentParser(prog = 'Inference with a trained neural network.',
                               description = '''This programm will predict the flower class to a given picture.''')
    
    # Choosing the path to the image
    parser.add_argument('--image_file', type = str, default = 'flowers/test/83/image_01777.jpg', 
                        help = 'Imgage File, Default is flowers/test/83/image_01777.jpg')

    # Chosing JSON-File
    parser.add_argument('--json_file', type = str, default = 'cat_to_name.json', 
                        help = 'JSON-File, default is cat_to_name.json')
    
    #Chosing Checkpoint7
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint_densenet121.pth',
                       help = 'Saved checkpoint, default checkpoint is checkpoint_densenet121.pth')
    
    # Choosing GPU
    parser.add_argument('--gpu', type = bool, default = False, 
                        help = 'GPU, default is False')
    
    # Getting top_k
    parser.add_argument('--topk', type = int, default = 5,
                       help = 'Top k predicitons, default is 5.')

    #Returning the arguments
    return parser.parse_args()


def load_checkpoint(filepath, flower_classes):
    """Loads a given checkpoint or model, which was pretrained
    
       Input: 
           filepath - the path and checkpoint file
           flower_classes - number of classes
              
       Output:
           model - the loaded model
    """
    
    # Loading the checkpoint
    checkpoint = torch.load(filepath)
    
    # Loading the model, depending on the used in in the checkpoint_file
    if  checkpoint['model_arch_name'] == 'densenet121':
        print(f"Loading Model: {checkpoint['model_arch_name']}")
        model = models.densenet121(pretrained = True)
        classifier_input = 1024
        print(f"Model loaded: \n {checkpoint['model_arch_name']}")
        
    elif checkpoint['model_arch_name'] == 'densenet161':
        print(f"Loading Model: {checkpoint['model_arch_name']}")
        model = models.densenet161(pretrained = True)
        classifier_input = 2208
        print(f"Model loaded: \n {checkpoint['model_arch_name']}")
    
    elif checkpoint['model_arch_name'] == 'densenet169':
        print(f"Loading Model: {checkpoint['model_arch_name']}")
        model = models.densenet169(pretrained = True)
        classifier_input = 1664
        print(f"Model loaded: \n {checkpoint['model_arch_name']}")
    
    elif checkpoint['model_arch_name'] == 'densenet201':
        print(f"Loading Model: {checkpoint['model_arch_name']}")
        model = models.densenet201(pretrained = True)
        classifier_input = 1920
        print(f"Model loaded: \n {checkpoint['model_arch_name']}") 
    
    # Freezing the model parameter
    for param in model.parameters():
        param.requires_grad = False     

    # Building the classifier Layers
    model.classifier = nn.Sequential(nn.Linear(classifier_input, checkpoint['hidden_units']),
                                     nn.ReLU(),
                                     nn.Dropout(p = checkpoint['dropout']),
                                     nn.Linear(checkpoint['hidden_units'], flower_classes),
                                     nn.LogSoftmax(dim = 1))
    
    # Assigning the loaded attributes to the model
    model.load_state_dict = checkpoint['model_state_dict']
    model.class_to_idx = checkpoint['model.class_to_idx']
    
    # Defining the loss function
    criterion = nn.NLLLoss()
    
    return model


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns a Torch Tensor for prediction
       
       Input: 
           image - image_path, where the image can be found.
       
       Output: 
           image_transformed - a preprocessed image as a tensor
    """
    #Source: pytorch documentation - https://pytorch.org/vision/main/transforms.html and
    #https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html
    
    # Loading the image
    img = Image.open(image)
    
    # Defining the transforms to transform the image for the model
    img_transform = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    #transform the image
    image_transformed = img_transform(img)
    
    return image_transformed



def predict(image_path, model, topk=5):
    """Predicts the class (or classes) and probabilities of an image using a trained deep learning model.
    
    Input: 
        Path to an image file, a trained model, top k predicitons
    
    Output: 
        probs, classes: topk probabilites and classes of the image-prediciton as numpy-array
    """
    
    #preprocessing the image
    image = process_image(image_path)
    
    # Prediction is expecting a 4D Tensor, image is 3D (ErrorMessage) - so we have to "add" a dimension with unsqueeze
    # https://stackoverflow.com/questions/65470807/how-to-add-a-new-dimension-to-a-pytorch-tensor
    image = image.unsqueeze(0)
    
    #predicting the probabilities and classes of the image
    model = model.eval()
    with torch.no_grad():
        image = image.to(device)
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        probs, classes = ps.topk(topk, dim = 1)
        
        # Wrangling probs and classes to get the correct output for the next steps.
        # Changing both to numpy arrays (had diverse ErrorMessages) with 
        # squeeze() for the plot, and numpy() for the plot, to('cpu') is needed for building numpy array
        probs = torch.squeeze(probs, 0).to('cpu').numpy()
        classes = torch.squeeze(classes, 0).to('cpu').numpy()
        
        # #Inverting class to idx into idx_to_class, so the correct class idx is used for the prediction
        idx_to_class = {class_cat: class_idx for class_idx, class_cat in model.class_to_idx.items()}
        
        # Changing the Category name and the idx for a correct plot
        classes = [idx_to_class[i] for i in classes]
        
        ## Changing the top k predicted classes to classes names
        classes = [cat_to_name[i] for i in classes]
        
        ## Getting the name of the flower to predict
        flower_cat = image_path.split('/')[2]
        title = cat_to_name[str(flower_cat)]
        
    model = model.train()
    
    return probs, classes, title


def check_device(gpu):
    """Checks, if GPU is available and if it was chosen through argparse.
    
    Inputs:
        gpu - True, if GPU is wanted. False if not
        
    Outputs:
        device - "cuda" or "cpu"
    """
    
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")

    if device == "cuda":
        print(f"GPU will be used.")
    elif not torch.cuda.is_available():
        print(f"GPU is not available. CPU will be used.")
    else:
        print(f"CPU will be used.")
        
    return device



"""Main root of the program""" 

# Getting the input arguments for the training
in_arg = get_input_args()

# Getting the class value mapping through a json file
with open(in_arg.json_file, 'r') as f:
    cat_to_name = json.load(f)
flower_classes = len(cat_to_name)

# Checking if GPU is chosen and available, moving model to GPU or CPU
print(f"Checking the device...")
device = check_device(in_arg.gpu)
print(f"-----------------------")

#Loading the model from a given checkpoint
print(f"Loading the model:")
model = load_checkpoint(in_arg.checkpoint, flower_classes)
model = model.to(device)
print(f"/n")
print(f"Model was loaded.")
print(f"-----------------------")

#Predicting the image
print(f"Predicting the given image...")
probs, classes, title = predict(in_arg.image_file, model, in_arg.topk)
print(f"-----------------------")

#Display the image and the models prediction
print(f"The image of {title} is predicted as:")
for i in range(in_arg.topk):
    print(f"{classes[i]:<30} {probs[i]:.3f}")
print(f"-----------------------")