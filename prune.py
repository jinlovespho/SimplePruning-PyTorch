""" 剪枝(prune)及微调（再训练 retrain） """

# 外部库
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

# 自定义
from utils import *
import config


if __name__ == "__main__":
    model_name= 'resnet18'
    checkpoint = 'resnet18'
    prune_checkpoint = ''  # 用于设置剪枝后模型的checkpoint文件名
    save_every = 20  # 每 ”剪掉“一定百分比大小的权重，就进行一次验证
    cutout = True  # 是否在原图像上挖洞

    # 设置再训练（微调）参数
    finetune_steps = 1
    lr = 0.001
    weight_decay = 0.0005
    momentum = 0.9

    old_format = False
    if 'wrn' in model_name:
        old_format = True

    model = config.models(model_name)  # 新建空模型
    model, sd = load_model(model, checkpoint, old_format)  # 加载预训练模型（权重）
    device = config.DEVICE
    model.to(device)

    if prune_checkpoint == '':
        prune_checkpoint = checkpoint + '_l1_'
    else:
        prune_checkpoint = prune_checkpoint

    train_loader, test_loader = get_cifar_loaders(config.DATA, cutout=cutout)
    optimizer = optim.SGD(
        [w for name, w in model.named_parameters() if not 'mask' in name],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 使学习率恢复到上次预训练结束时的状态
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-10)
    for epoch in range(sd['epoch']):
        scheduler.step()
    for group in optimizer.param_groups:
        group['lr'] = scheduler.get_lr()[0]

    for prune_rate in tqdm(range(100)):
        model = sparsify(model, prune_rate)  # 剪枝
        finetune(model, train_loader, criterion, optimizer, finetune_steps)  # 再训练 (微调)

        if prune_rate % save_every == 0:
            checkpoint = prune_checkpoint + str(prune_rate)
        else:
            checkpoint = None

        if checkpoint:
            validate(model, prune_rate, test_loader, criterion, checkpoint=checkpoint)
