from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

# customize
import config


device = config.DEVICE
global error_history
error_history = []


class  AverageMeter ( object ):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self . val  =  0
        self . avg  =  0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_cifar_loaders(data_loc=config.DATA, batch_size=256, cutout=True, n_holes=1, length=16):
    """Load cifar10 dataset
    Args:
        data_loc (str): The location of the cifar10 dataset.
        cutout (bool): If true, n_holes black squares will be randomly made for each training set image.
        n_holes (int): The number of black square areas, only takes effect when cutout is true.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms . ToTensor (),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    if cutout:
        transform_train.transforms.append(Cutout(n_holes=n_holes, length=length))

    transform_test = transforms.Compose([
        transforms . ToTensor (),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_set = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

    test_set = torchvision.datasets.CIFAR10(root=data_loc, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader


def load_model(model, sd, old_format=False):
    """Load pretrained model (weights).
    Args:
        sd (str): The name of the checkpoint file to be loaded, sd is the state dict.
        old_format (bool): Used to load checkpoint files in general format.
    Returns:
        model: The model loaded with the checkpoint file.
        sd: The content of the loaded checkpoint file.
    """
    sd = torch.load('checkpoints/%s.t7' % sd, map_location='cpu')
    new_sd = model.state_dict()
    if 'state_dict' in sd.keys():
        old_sd = sd['state_dict']
    else:
        old_sd = sd['net']

    # Copy the weights of the loaded model to the new model
    if old_format:
        # this means the sd we are trying to load does not have masks and/or is named incorrectly
        keys_without_masks = [k for k in new_sd.keys() if 'mask' not in k]
        for old_k, new_k in zip(old_sd.keys(), keys_without_masks):
            new_sd[new_k] = old_sd[old_k]
    else:
        new_names = [v for v in new_sd]
        old_names = [v for v in old_sd]
        for  i , j  in  enumerate ( new_names ):
            if not 'mask' in j:
                new_sd[j] = old_sd[old_names[i]]

    try:
        model.load_state_dict(new_sd)
    except:
        new_sd = model.state_dict()
        old_sd = sd['state_dict']
        k_new = [k for k in new_sd.keys() if 'mask' not in k]
        k_new = [k for k in k_new if 'num_batches_tracked' not in k]
        for o, n in zip(old_sd.keys(), k_new):
            new_sd [ n ] =  old_sd [ o ]

        model.load_state_dict(new_sd)
    return model, sd


def get_error(output, labels, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    max_k = max(topk)
    batch_size = labels.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred  =  pred . t ()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res  = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


def get_no_params(net):
    """Count the number of non-zero weights in all convolutional layers."""
    params = net
    total = 0
    for  p  in  params :
        num = torch.sum(params[p] != 0)
        if 'conv' in p:
            total  +=  num
    return total


def train(model, train_loader, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()


    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        
        loss = criterion(output, labels)
       
       # Calculate top1 and top5 errors
        err1, err5 = get_error(output.detach(), labels, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(err1.item(), images.size(0))
        top5.update(err5.item(), images.size(0))

        optimizer . zero_grad ()
        loss.backward()
        optimizer.step()
        

    print("Finished Training")


def validate(model, epoch, val_loader, criterion, checkpoint=None):
    """
    Args:
        checkpoint (str): The name of the checkpoint file to save; if not, it will not be saved; the default is not saved.
    """
    global error_history

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    """
    avg_loss = 0
    total_batch = len(val_loader)

    model.eval()
    for i, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        err1, err5 = get_error(output.detach(), labels, topk=(1, 5))

        losses.update(loss.item(), images.size(0))

        top1.update(err1.item(), images.size(0))
        top5.update(err5.item(), images.size(0))

    error_history.append(top1.avg)
    
    print('Test loss: ', loss)
    

    avg_loss += loss / 5
    print('Average of Test loss: ', avg_loss)
    """
  
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
      for data in val_loader:
              images, labels = data
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              loss = criterion(outputs, labels)
              
              test_loss += loss.item()
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
    print("correct: ", correct)
    print("total: ", total)
    

    print("Accuracy of the network: ", 100.*correct/total, "%")



    if checkpoint:
        state = {
            'net': model.state_dict(),
            'masks': [w for name, w in model.named_parameters() if 'mask' in name],
            'epoch': epoch,
            'error_history': error_history,
        }
        torch.save(state, config.CP + '/%s.t7' % checkpoint)


def finetune(model, train_loader, criterion, optimizer, steps=100):
    #model.train()     #To make the 'without retraining' after pruning
    data_iter = iter(train_loader)
    for  i  in  range ( steps ):
        try:
            images, labels = data_iter.next()
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = data_iter.next()

        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer . zero_grad ()
        loss.backward()
        optimizer.step()


def expand_model(model, layers=torch.Tensor()):

    for layer in model.children():
        if len(list(layer.children())) > 0:
            layers  =  expand_model ( layer , layers )   # recursively until the bottom
        else:
            if isinstance(layer, nn.Conv2d) and 'mask' not in layer._get_name():
                layers = torch.cat((layers.view(-1), layer.weight.view(-1)))
    return layers


def calculate_threshold(model, rate):
    """Calculate the weight threshold.
    Args:
        rate (float): 0~100, the ratio of pruning.
    Returns:
        float: Threshold, weights smaller than this value will be removed during pruning.
    """
    empty  =  torch .Tensor ()   # Create an empty Tensor to which target weights will be added
    if torch.cuda.is_available():
        empty = empty.cuda()
    pre_abs  =  expand_model ( model , empty )   # Get all uncropped weights and get a one-dimensional Tensor
    weights  =  torch . abs ( pre_abs )   # take the absolute value

    return  np . percentile ( weights . detach (). cpu (). numpy (), rate )   # Take the quantile of all weights (determined by the percentage rate) as the threshold


def sparsify(model, prune_rate):
    """Prunes according to the given ratio.
    Args:
        prune_rate (float): The ratio of prune.
    Returns:
        The pruned model.
    """
    threshold = calculate_threshold(model, prune_rate)
    try:
        model.__prune__(threshold)
    except:
        model.module.__prune__(threshold)
    return model


class Cutout(object):
    """A random number of black squares are created on an image.
    Args:
        n_holes (int): Number of square areas.
        length (int): The side length of the square.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): An image of type Tensor with dimensions (C, H, W).
        Returns:
            Tensor: image with n_holes square area, Tensor type
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y  =  np . random . randint ( h )
            x  =  np . random . randint ( w )

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
