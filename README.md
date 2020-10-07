# [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

## 内容说明

这份代码是如题的这篇2015年论文的PyTorch版复现，论文内容是对CNN进行**网络剪枝**（network pruning），去除不重要的权重（weight），从而显著减小模型大小。论文作者在这篇论文的基础上加上量化（Quantization  ）和霍夫曼编码（Huffman Encoding ），在16年提出了为人熟知的三阶段方法——[Deep Compression](https://arxiv.org/abs/1510.00149)，前者是后者的初始版本。原仓库作者似乎把它们搞混了，为了和后者区分开来，我把仓库名改成了简单剪枝（Simple pruning）。

我这份代码是fork过来的，原仓库为[DeepCompression-PyTorch](https://github.com/jack-willturner/DeepCompression-PyTorch)。在跑通的过程中，我修改了一些细枝末节的东西，同时根据自己的理解对关键代码加了一些中文注释，希望对研究阅读原代码和研究论文的人有所帮助。

## 运行环境

- **Windows 10**


- 硬件：AMD R5 3600X + RTX2060 super（我用的单GPU，原仓库支持双GPU，为了精简代码我去掉了）


- **pytorch1.2 GPU版** （原仓库没有说明PyTorch的版本，我用的是1.2）

## 使用步骤

原仓库是通过命令行运行的，可能是用的Linux。我用的IDE是**pycharm**，为了方便调试我把命令行去掉了，在IDE中直接运行相应的.py文件。

#### 1. 准备cifar10数据集

在项目下新建文件夹**cifar10**和**checkpoints**，在cifar10中放入下载好的[cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)文件。（如果想换个位置，可以自行设置config.py中的DATA）

#### 2. 预训练（pretrain）

运行**train.py**，会在**checkpoints**文件夹下得到一个**.t7文件**，里面主要是预训练模型的权重。（注意：默认训练200个epoch，时间很长，可以把**epochs**改小点）

#### 3. 剪枝（prune）和再训练（retrain）

运行**prune.py**，会在**checkpoints**文件夹下得到多个**.t7文件**，分别是处于剪枝的不同阶段的模型权重。（同样，在运行之前可以把**finetune_steps**改小一点，先正常运行再说）

## 代码实现的细节

1. 原论文（15年的）针对的LeNet、AlexNet和VGG-16，而该复现针对的是**RestNet**和**宽ResNet**（见论文[Wide Residual Networks](https://arxiv.org/abs/1605.07146)）进行剪枝。
2. 理论上随着剪枝比例的增加，模型的权重应该越来越少，因此剪枝过程生成的 **.t7文件** 会越来越小，但实际上并不会减小。这是因为权重存储在**Tensor**张量中，剪枝的过程只是将Tensor中的部分元素置为0，但是仍然占据着空间。pytorch保存权重的时候将每个Tensor视为一个对象，**Tensor是稠密矩阵，只能整体保存，不能单独保存某几个元素的值**。如果想要真正实现模型压缩，需要考虑**稀疏矩阵**（参考**torch.sparse**）或者其他的方法。
3. **权重阈值**的选择：
   - 原论文的思路是设定一个权重的阈值，将低于这个阈值的权重被认为是不重要的，因此将被“剪掉”，但**原论文中并没有说明如何选择这个阈值**。
   - 该复现采用的方法——是获取剩余的（未被剪掉）全部权重，取其**分位数**。例如：如果剪枝比例是50%，就取全部权重的中位数。

4. 其他内容有时间再补。

