import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, MyNet
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]

    torch.manual_seed(0x5EED)
    torch.cuda.manual_seed(0x5EED)
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=64)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'mynet':
        model = MyNet()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    # Run any number of epochs you want
    ep = 25
    train_acc = [0] * ep
    train_loss = [0] * ep
    test_acc = [0] * ep
    test_loss = [0] * ep
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))

        train_acc[epoch] = acc
        train_loss[epoch] = ave_loss

        ################
        ## Validation ##
        ################
        model.eval()
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        with torch.no_grad():
            for batch, (x, label) in enumerate(val_loader,1):
                if use_cuda:
                    x, label = x.cuda(), label.cuda()
                out = model(x)
                loss = criterion(out, label)
                total_loss += loss.item()
                _, pred_label = torch.max(out, 1)
                total_cnt += x.size(0)
                correct_cnt += (pred_label == label).sum().item()
            print ('test loss: {:.6f}, acc: {:.3f}'.format(total_loss / batch, correct_cnt / total_cnt))
            test_acc[epoch] = correct_cnt / total_cnt
            test_loss[epoch] = total_loss / batch

        model.train()
        lr_scheduler.step()

    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    # Plot Learning Curve
    # TODO
    
    plt.plot(train_acc)
    plt.xlabel('epoches')
    plt.ylabel('accurancy')
    plt.savefig('train_acc_{}.jpg'.format(model_type))
    plt.close()

    plt.plot(test_acc)
    plt.xlabel('epoches')
    plt.ylabel('accurancy')
    plt.savefig('test_acc_{}.jpg'.format(model_type))
    plt.close()

    plt.plot(train_loss)
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.savefig('train_loss_{}.jpg'.format(model_type))
    plt.close()
    
    plt.plot(test_loss)
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.savefig('test_loss_{}.jpg'.format(model_type))
    plt.close()