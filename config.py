from models import *

DATA = '/mnt/ssd2/dataset/CIFAR10'  # It is the folder that cifar10 will be saved.
CP = './checkpoints'  # Please make the new 'checkpoints' folder. (.t7) files will be saved in this folder.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def models(name):
    """create a new model
    Args:
        name (string): network type, choose from resnet9, resnet18, resnet34, resnet50, wrn_40_2, wrn_16_2, wrn_40_1
    """
    nets= {'resnet9'  : ResNet9(),
           'resnet18' : ResNet18(),
           'resnet34' : ResNet34(),
           'resnet50' : ResNet50(),
           'wrn_40_2' : WideResNet(40, 2),
           'wrn_16_2' : WideResNet(16, 2),
           'wrn_40_1' : WideResNet(40, 1)}

    return nets[name]
