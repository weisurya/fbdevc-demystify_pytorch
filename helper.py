import torch
from torch import nn, optim
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop=0.5):
        '''
        Arguments
        ---------
        input_size : integer | size of input layer
        output_size : integer | size of output layer
        hidden_layers : list of integers | size of hidden layers
        '''
        super().__init__()
        
        # Set initial layer
        self.fc = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Prepare the combination for the hidden layers
        layer_size = zip(hidden_layers[:-1], hidden_layers[1:])
        
        # Set hidden layers
        self.fc.extend([nn.Linear(this_layer, next_layer) for this_layer, next_layer in layer_size])
        
        # Set the output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # Set dropout module
        self.dropout = nn.Dropout(p=drop)
    
    def forward(self, x):
        for each in self.fc:
            x = F.relu(each(x))
            x = self.dropout(x)
        
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    
def validate_model(model, testloader, criterion, device='cpu', isTransfer=False):
    accuracy = 0
    test_loss = 0
    model.to(device)
    
    for x, y in testloader:
        # Set into particular device type
        x, y = x.to(device), y.to(device)
            
        # Flatten image into 784-long vector if it's not for transfer learning
        if isTransfer == False:
            x.resize_(x.size()[0], 784)
        
        # Forward propagation
        y_hat = model.forward(x)
        
        # Calculate loss
        loss = criterion(y_hat, y)
        
        # Calculate the test loss
        test_loss += loss.item()
        
        # Calculate the accuracy
        # Because the activation function is using Log-softmax, we calculate the exponential to get the probabilities
        prob = torch.exp(y_hat)
        
        # Compare the highest probability of our prediction with the true label
        correctness = (y.data == prob.max(1)[1])
        
        # Calculate the accuracy by taking account all of the correct prediction and take the mean
        accuracy += correctness.type_as(torch.FloatTensor()).mean()
        
    return test_loss, accuracy
        
    
def train_model(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=10, device='cpu', isTransfer=False):
    step = 0
    train_loss = 0
    model.to(device)
    
    for e in range(epochs):
        # Set the model into training mode
        model.train()
        
        for x, y in trainloader:
            # Set into particular device type
            x, y = x.to(device), y.to(device)
            
            step += 1
            
            # Flatten image into 784-long vector if it's not for transfer learning
            if isTransfer == False:
                x.resize_(x.size()[0], 784)
            
            # Make sure the gradient is pristine
            optimizer.zero_grad()
            
            # Forward propagation
            y_hat = model.forward(x)
            
            # Calculate loss
            loss = criterion(y_hat, y)
            
            # Back propagation
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Calculate the training loss
            train_loss += loss.item()
            
            # Evaluate the performance every 'print_every' images
            if step % print_every == 0:
                # Set into evaluation mode; Turn off the dropout
                model.eval()
                
                # Turn off the autograd to improve the performance
                with torch.no_grad():
                    test_loss, accuracy = validate_model(model, testloader, criterion, device, isTransfer)
                
                print("Device: {}.. ".format(device),
                      "Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))