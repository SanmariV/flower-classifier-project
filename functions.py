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
    model, criterion = nn_type(model, lrate)
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



# Load a checkpoint and rebuild the model
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

#Process an image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model 
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

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


        