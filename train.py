#Imports
import numpy as np
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
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Decide which model to use: only gve two options
def nn_type(model,lrate):
 #Freezing parameters to avoid backpropping through them
    for param in model.parameters():
        param.requires_grad = False

        new_classifier= nn.Sequential(OrderedDict([
                                     ('dropout', nn.Dropout(p=0.5)),
                                     ('fc1', nn.Linear(25088,4096)),
                                     ('relu', nn.ReLU()),
                                     ('fc2', nn.Linear(4096,102)),
                                     ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = new_classifier

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = lrate)
        model.to('cpu')
    return model, optimizer, criterion

#validation function
def validate(model,criterion,validloader):
    model.to('cuda:0' if input_2 =='y' else 'cpu')
    validation_loss = 0 
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(validloader):
        
        inputs, labels = inputs.to('cuda:0' if input_2 =='y' else 'cpu'), labels.to('cuda:0' if input_2 =='y' else 'cpu')

        output = model.forward(inputs)
        validation_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        ps.max(dim=1)
        are_they_equal = (labels.data == ps.max(dim=1)[1])
        #are_they_equal?
        accuracy += (are_they_equal.type(torch.FloatTensor).sum()/len(are_they_equal))
        
    return validation_loss, accuracy

#test function
def test(testloader):
         model.to('cuda:0' if input_2 =='y' else 'cpu')
         model.eval()
         test_loss = 0
         test_accuracy = 0
    
         with torch.no_grad():
              for inputs, labels in testloader:
                  inputs, labels = inputs.to('cuda:0' if input_2 =='y' else 'cpu'), labels.to('cuda:0' if input_2 =='y' else 'cpu')
                  outputT = model.forward(inputs)
                  test_loss += criterion(outputT,labels).item()
                  ps = torch.exp(outputT)
                  ps.max(dim=1)
                  are_they_equal = (labels.data == ps.max(dim=1)[1])
                  #are_they_equal
                  test_accuracy += (are_they_equal.type(torch.FloatTensor).sum()/len(are_they_equal))/len(testloader)
              print('Test accuracy: {:.3f} '.format(test_accuracy))
    
         model.train()
        
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
        
#Get input from user: choose between 2 networks
if __name__ == '__main__':
     while True:
                try:
                    input_1 = str(input("Enter vgg16 or vgg19:"))
                    lrate = float(input('Select the learning rate.  We suggest a float that satisfies: 0.000001< learning rate < 0.001: '))
                    if input_1 == 'vgg16':
                       model = models.vgg16(pretrained = True)
                       model, optimizer, criterion = nn_type(model,lrate) 
                    break
                    if input_1 == 'vgg19':
                       model = models.vgg19(pretrained = True)
                       model, optimizer, criterion = nn_type(model,lrate)
                    break
                except ValueError:
                                 print('Error: {} is not a recognised network'.format(input_1))
   
     if input_1 == 'vgg16':
        model = models.vgg16(pretrained = True)
        model, optimizer, criterion = nn_type(model,lrate)
     elif input_1 == 'vgg19':
          model = models.vgg19(pretrained = True)
          model, optimizer, criterion = nn_type(model,lrate)
              
     while True:
                try:
                    input_2 = str(input('Would you like to train the model with the GPU: Yes[y] or No[n]?'))
                    if input_2 == 'y':  
                       break
                    if input_2 == 'n':
                       break
                except ValueError:
                                  print('Error: {} is not a recognised choice'.format(input_2))

     #Training network
     from workspace_utils import active_session
     with active_session():
          if input_2 == 'y':
             model.to('cuda:0' if input_2 =='y' else 'cpu')
             epochs = int(input('Select the number of epochs.  We suggest an integer so that 2 < epochs < 20: '))
             print('The learn rate you specificed is {}'.format(lrate))
             print('Training of the network will start now...')
    
          elif input_2 =='n':
               model.to('cpu')
               epochs = int(input('Select the number of epochs.  We suggest an integer so that 2 < epochs < 20: '))
               print('The learn rate you specificed is {}'.format(lrate))
               print('Training of the network will start now...')
    
          epoch = epochs
          print_every = 10
          steps= 0 
          model.to('cuda:0' if input_2 =='y' else 'cpu')
        
          for e in range(epoch):
              model.train()
              running_loss = 0
       
              for ii, (inputs, labels) in enumerate(trainloader):
                  steps += 1
                  inputs, labels = inputs.to('cuda:0' if input_2 =='y' else 'cpu'), labels.to('cuda:0' if input_2 =='y' else 'cpu')
                  optimizer.zero_grad()
        
            #forward pass & backward pass
            
                  output = model.forward(inputs)
                  loss = criterion(output,labels)
                  loss.backward()
                  optimizer.step()
            
                  running_loss += loss.item()
            
                  if steps % print_every ==0:
                #turn dropout off by putting model in evaluaton mode! use this mode for inference + validation
                     model.eval()
                #run validation function
                #we don't worry about the weights during validaton
                     with torch.no_grad():
                          validation_loss, accuracy = validate(model,criterion,validloader)
                        
                          print('Epoch:{}/{}'.format(e+1,epoch),
                                'Train Loss:{:.3f}'.format(running_loss/print_every),
                                'Validation Loss: {:.3f}'.format(validation_loss/len(validloader)),
                                'Accuracy:{:.3f}'.format(accuracy/len(validloader)))
                
                          running_loss = 0
                
                     model.train()              #turn dropout back on to train
       
     test(testloader) #Run the test function 

#Save the checkpoint to file checkpoint.pth
#I got some help here: https://medium.com/@annebonner/load-that-checkpoint-tips-and-tricks-for-a-successful-udacity-project-checkpoint-load-dd12bea7c505
     model.class_to_idx = train_dataset.class_to_idx
     model.cpu()
     checkpoint = {'input_size': 25088,
                   'output_size': 102,
                   'epochs': epochs,
                   'batch_size': 64,
                   'model': model,
                   'classifier': model.classifier,
                   'optimizer': optimizer.state_dict(),
                   'state_dict': model.state_dict(),
                   'class_to_idx': model.class_to_idx
                  }
     torch.save(checkpoint, 'checkpoint.pth')

    #Turn off the GPU if the user activated it after saving the checkpoint is finished 
     #if model('cuda:0'):
       # model.to('cpu')
    #End of training
     print('Training is now complete and the model is saved to checkpoint.pth')