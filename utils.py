from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 自定义
import config


device = config.DEVICE
global error_history
error_history = []


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_cifar_loaders(data_loc=config.DATA, batch_size=128, cutout=True, n_holes=1, length=16):
    """加载 cifar10数据集
    Args:
        data_loc (str): cifar10数据集的位置。
        cutout (bool): 如果为真，将对每张训练集图像都随机制造 n_holes个黑色正方形区域。
        n_holes (int): 黑色正方形区域的数量，只在 cutout为真时生效。
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    if cutout:
        transform_train.transforms.append(Cutout(n_holes=n_holes, length=length))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_set = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

    test_set = torchvision.datasets.CIFAR10(root=data_loc, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader


def load_model(model, sd, old_format=False):
    """加载预训练模型（权重）。
    Args:
        sd (str): 要加载的 checkpoint文件名，sd即 state dict。
        old_format (bool): 用于加载一般格式的 checkpoint文件。
    Returns:
        model: 加载过checkpoint文件的模型。
        sd: 所加载checkpoint文件的内容。
    """
    sd = torch.load('checkpoints/%s.t7' % sd, map_location='cpu')
    new_sd = model.state_dict()
    if 'state_dict' in sd.keys():
        old_sd = sd['state_dict']
    else:
        old_sd = sd['net']

    # 将所加载的模型的权重复制到新模型中
    if old_format:
        # this means the sd we are trying to load does not have masks and/or is named incorrectly
        keys_without_masks = [k for k in new_sd.keys() if 'mask' not in k]
        for old_k, new_k in zip(old_sd.keys(), keys_without_masks):
            new_sd[new_k] = old_sd[old_k]
    else:
        new_names = [v for v in new_sd]
        old_names = [v for v in old_sd]
        for i, j in enumerate(new_names):
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
            new_sd[n] = old_sd[o]

        model.load_state_dict(new_sd)
    return model, sd


def get_error(output, labels, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    max_k = max(topk)
    batch_size = labels.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


def get_no_params(net):
    """统计所有卷积层中非零权重的数量。"""
    params = net
    total = 0
    for p in params:
        num = torch.sum(params[p] != 0)
        if 'conv' in p:
            total += num
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

        # 计算 top1和 top5误差
        err1, err5 = get_error(output.detach(), labels, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(err1.item(), images.size(0))
        top5.update(err5.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, epoch, val_loader, criterion, checkpoint=None):
    """
    Args:
        checkpoint (str): 保存checkpoint文件的名称；如果没有，则不保存；默认不保存。
    """
    global error_history

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

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
    if checkpoint:
        state = {
            'net': model.state_dict(),
            'masks': [w for name, w in model.named_parameters() if 'mask' in name],
            'epoch': epoch,
            'error_history': error_history,
        }
        torch.save(state, config.CP + '/%s.t7' % checkpoint)


def finetune(model, train_loader, criterion, optimizer, steps=100):
    model.train()
    data_iter = iter(train_loader)
    for i in range(steps):
        try:
            images, labels = data_iter.next()
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = data_iter.next()

        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def expand_model(model, layers=torch.Tensor()):

    for layer in model.children():
        if len(list(layer.children())) > 0:
            layers = expand_model(layer, layers)  # 递归，直到最底层
        else:
            if isinstance(layer, nn.Conv2d) and 'mask' not in layer._get_name():
                layers = torch.cat((layers.view(-1), layer.weight.view(-1)))
    return layers


def calculate_threshold(model, rate):
    """计算权重阈值。
    Args:
        rate (float): 0~100, 剪枝的比例。
    Returns:
        float: 阈值，小于该值的权重在剪枝时将被去除。
    """
    empty = torch.Tensor()  # 创建一个空 Tensor，目标权重将被加到里面
    if torch.cuda.is_available():
        empty = empty.cuda()
    pre_abs = expand_model(model, empty)  # 获取所有未被裁剪的权重，得到一个一维 Tensor
    weights = torch.abs(pre_abs)  # 取绝对值

    return np.percentile(weights.detach().cpu().numpy(), rate)  # 取所有权重的分位数（由百分数 rate决定）作为阈值


def sparsify(model, prune_rate=50.):
    """按给定比例进行剪枝。
    Args:
        prune_rate (float): 剪枝的比例。
    Returns:
        剪枝后的模型。
    """
    threshold = calculate_threshold(model, prune_rate)
    try:
        model.__prune__(threshold)
    except:
        model.module.__prune__(threshold)
    return model


class Cutout(object):
    """在一张图像上，随机制造一定数量的黑色正方形区域。
    Args:
        n_holes (int): 正方形区域的数量。
        length (int): 正方形的边长。
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor类型的图像，尺寸为 (C, H, W)。
        Returns:
            Tensor: 带有 n_holes个正方形区域的图像，Tensor类型
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
