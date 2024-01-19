
""" Pretrain the model for later prune """

# external library
from __future__ import print_function
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# customize
from utils import *
from tqdm import tqdm  # Used to display the progress bar
import config

import argparse 

parser = argparse.ArgumentParser(description='jinlovespho simple pytorch pruning')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--data_loc', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float )

args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model
    checkpoint = 'resnet18'  # Name of the saved checkpoint file
    epochs = args.epochs
    lr = args.lr 
    lr_decay_ratio = 0.2
    weight_decay = 0.0005
    momentum = 0.9

    model = config.models(model_name)
    device = config.DEVICE
    model.to(device)

    train_loader, test_loader = get_cifar_loaders(config.DATA)
    optimizer = optim.SGD(
        [w for name, w in model.named_parameters() if not 'mask' in name],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-10)  # use SGDR method to adjust lr with epoch
    criterion = nn.CrossEntropyLoss()

    global error_history
    error_history = []
    for epoch in tqdm(range(epochs)):
        train(model, train_loader, criterion, optimizer)
        validate(model, epoch, test_loader, criterion, checkpoint=checkpoint if epoch != 2 else checkpoint + '_init')
        scheduler.step()


