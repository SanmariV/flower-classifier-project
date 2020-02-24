#Imports
import numpy as np
import os
import copy
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image, ImageOps
from collections import OrderedDict
from collections import namedtuple
from workspace_utils import active_session
import functions as f

#Specify directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Data transformations
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(30),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = valid_transforms

#Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

# DONE: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)

#Label mapping
with open('cat_to_name.json', 'r') as g:
     cat_to_name = json.load(g)

# Function to load the checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.load_state_dict(checkpoint['state_dict'])  #very important to include otherwise the model won't rebuild when called
    
    return model
        
#Load the checkpoint and the model
model = load_checkpoint('checkpoint.pth')

#which image do you want a predicition for?
image_path_1 = 'flowers/test/1/image_06743.jpg'
image_path_2 = 'flowers/test/12/image_04014.jpg'
image_path_3 = 'flowers/test/30/image_03512.jpg'

image_path = image_path_3  

print('\nThis applicaton will predict the top 5 probabilities, classes and flowers of the image located in {}.'.format(image_path))

while True:
            try:
                input_4 = str(input('Would you like to make predictions with the GPU: Yes[y] or No[n]?'))
                if input_4 == 'y':  
                   break
                if input_4 == 'n':
                   break
            except ValueError:
                              print('Error: {} is not a recognised choice'.format(input_4))
                                    
#Process the user's image choice before prediction
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Process a PIL image for use in a PyTorch model 
    image = Image.open(image_path)
    transform_image = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image_tensor = transform_image(image)
    
    #convert image to an array
    image = np.array(image_tensor)
    
    return image

#Prediction
# I got some help from Josh Bernard here: https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

model.class_to_idx =train_dataset.class_to_idx

cat_index = model.class_to_idx

def predict(image_path, model, topk=5):
    model.class_to_idx =train_dataset.class_to_idx

    cat_index = model.class_to_idx
    
    model.to('cuda:0' if input_4 =='y' else 'cpu')
    
    model.eval() #we don't want to update the weights of the model, turnoff dropout
    
    # Process image
    image = process_image(image_path)
    imageT = torch.from_numpy(image).type(torch.FloatTensor) #convert back to tensor float from numpy array
    #insert 1 in position 0 for batch size=1 and ensure that it is a float tensor
    input_to_model = imageT.unsqueeze_(0).type(torch.FloatTensor)
    
    #we don't to update the weights so turn it off
    with torch.no_grad():
        
        input_to_model = input_to_model.to('cpu')
        input_to_model = input_to_model.type(torch.FloatTensor)
        output = model.forward(input_to_model)
        ps = torch.exp(output) #probabilities
        
        #designate top 5, make a list of probabilities and labels
        max_probs, max_labs = ps.topk(topk) #identify top 5 probabilities and labels
        max_probs = max_probs.detach().numpy().tolist()[0] #make a list of only probabilities
        max_labs = max_labs.detach().numpy().tolist()[0] #make a list of only labels
        
        # Convert indices to classes
        convert_invert = {val: key for key, val in cat_index.items()}
        max_labels = [convert_invert[lab] for lab in max_labs]    #these are a list of the categories of flowers
        max_flowers = [cat_to_name[convert_invert[lab]] for lab in max_labs]    #these are a list of the names of the flower predictions 
        model.train()  #reactivate model's dropout
        #returns top 5 probabilities
    return max_probs, max_labs, max_flowers
                                    
#Show the prediction to user
val1, val2, val3 = predict(image_path,model.to('cpu'))
print('Top 5 Probabilities: {}'.format(val1))
print('Top 5 Classes: {}'.format(val2))
print('Flower names: {}'.format(val3))                                    
