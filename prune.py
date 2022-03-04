""" Pruning and fine-tuning (retraining) """

# external library
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

# customize
from utils import *
import config


if __name__ == "__main__":
    model_name= 'resnet18'
    checkpoint = 'resnet18'
    prune_checkpoint = ''  # used to set the checkpoint file name of the pruned model
    save_every = 5  # Every time a certain percentage of the weight is "cut-off", a verification is performed
    cutout = True  # Whether to cut holes in the original image

    # Set retraining (fine-tuning) parameters
    finetune_steps = 10   # the number of iterative pruning
    lr = 0.001
    weight_decay = 0.0005   # L2 regularization
    momentum = 0.9

    old_format = False
    if 'wrn' in model_name:
        old_format = True

    model = config.models(model_name)  # create a new empty model
    model, sd = load_model(model, checkpoint, old_format)  # load pretrained model (weights)
    device = config.DEVICE
    model.to(device)

    if prune_checkpoint == '':
        prune_checkpoint = checkpoint + '_l2_'
    else:
        prune_checkpoint = prune_checkpoint

    train_loader, test_loader = get_cifar_loaders(config.DATA, cutout=cutout)
    optimizer = optim.SGD(
        [w for name, w in model.named_parameters() if not 'mask' in name],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Bring the learning rate back to what it was at the end of the last pretraining
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-10)
    for epoch in range(sd['epoch']):
        scheduler.step()
    for group in optimizer.param_groups:
        group['lr'] = scheduler.get_lr()[0]

    for prune_rate in [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
        model = sparsify(model, prune_rate)  # prune
        finetune(model, train_loader, criterion, optimizer, finetune_steps)  # retraining (fine tuning)

        if prune_rate % save_every == 0:
            checkpoint = prune_checkpoint + str(prune_rate)
        else:
            checkpoint = None

        if checkpoint:
            validate(model, prune_rate, test_loader, criterion, checkpoint=checkpoint)
            print(checkpoint)
            print("=========================")
