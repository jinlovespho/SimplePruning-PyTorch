from models import *

DATA = './cifar10'  # cifar10数据集的路径
CP = './checkpoints'  # checkpoint文件（.t7）的保存路径
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def models(name):
    """创建一个新模型
    Args:
        name (string): 网络类型，从 resnet9、resnet18、resnet34、resnet50、wrn_40_2、wrn_16_2、wrn_40_1中选择
    """
    nets= {'resnet9'  : ResNet9(),
           'resnet18' : ResNet18(),
           'resnet34' : ResNet34(),
           'resnet50' : ResNet50(),
           'wrn_40_2' : WideResNet(40, 2),
           'wrn_16_2' : WideResNet(16, 2),
           'wrn_40_1' : WideResNet(40, 1)}

    return nets[name]