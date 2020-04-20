from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import argparse
from helpers.get_dataloaders import get_dataloaders

def train_model(data_dir, arch, hidden_units, learning_rate, \
    cuda, epochs):
    
    trainloader, validloader, _, train_data = get_dataloaders(data_dir)
    
    # Get model
    model = eval("models.{}(pretrained=True)".format(arch))
    
    # Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False
        
    # Classifier input size to be used in our new classifier
    if arch == 'vgg16':
        in_size = model.classifier[0].in_features
    else: # arch == densenet121
        in_size = model.classifier.in_features
        
    # Update the classifier in  our model to match flower classes
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 256)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Use GPU if it's available
    device = torch.device("cuda" if cuda else "cpu")
        
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 20

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)       
            model.to(device)
            
            optimizer.zero_grad()
            
            #Forward pass
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        log_ps = model.forward(images)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    # Save the checkpoint
    class_to_idx = train_data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    state = {'input_size': in_size,
                'hidden_units': hidden_units,
                'output_size': 102,
                'structure': arch,
                'learning_rate': learning_rate,
                'classifier': model.classifier,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class
    }
                
    torch.save(state, 'checkpoint.pth')
    print("Training complete!")
                
    return model


def main():
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Train a new network on a data set')

    parser.add_argument('data_dir', type=str, \
        help='Path of the Image Dataset (with train, valid and test folders)')
    parser.add_argument('--arch', type=str, \
        help='Models architeture. Choose between vgg16 and densenet121. Default is vgg16')
    parser.add_argument('--learning_rate', type=float, \
        help='Learning rate. Default is 0.01')
    parser.add_argument('--hidden_units', type=int, \
        help='Hidden units. Default is 2100')
    parser.add_argument('--epochs', type=int, \
        help='Number of epochs. Default is 3')
    parser.add_argument('--gpu', action='store_true', \
        help='Use GPU for inference if available')
    
    args, _ = parser.parse_known_args()

    data_dir = args.data_dir
        
    arch = 'vgg16'
    if args.arch:
        arch = args.arch

    learning_rate = 0.001
    if args.learning_rate:
        learning_rate = args.learning_rate

    
    hidden_units = 2100
    if args.hidden_units:
        hidden_units = args.hidden_units

    epochs = 3
    if args.epochs:
        epochs = args.epochs


    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("Warning! GPU flag was set however GPU is not available")


    train_model(data_dir, arch=arch, hidden_units=hidden_units,\
     learning_rate=learning_rate, cuda=cuda, epochs=epochs)



if __name__ == '__main__':
    main()
