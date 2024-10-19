# Assignment 2 - Pix2Pix

本次实验的第二个部分要求用FCN网络实现Pix2Pix。

## 环境配置
本次实验使用解释器python3.10，以及库opencv-python 4.10.0.84, numpy 1.26.4, torch2.4.1, matplotlib3.9.2

## 模型训练
```basic
python train.py
```
---


## 网络参数与结构的调整过程
### FCN-8s(Net1)
首次尝试使用论文中提到的FCN-8s网络(相比其他网络效果更好)。

在此网络架构中，第一次卷积将3通道图像变为32通道并将大小缩减为原图像的1/2，随后依次经过4个卷积层得到图像大小分别为原图像的1/4,
1/8,1/16,1/32。再加两个卷积层(相当于CNN中全连接层的作用)得到512通道，大小为1/32的heatmap。

原论文中，直接将heatmap上采样即得到FCN-32s；将heatmap上采样2倍并跳跃连接与1/16大小的图像结合再上采样得到FCN-16s；
将上述与1/16图像结合的结果上采样2倍再与1/8图像结合再上采样得到FCN-8s。理论上还有FCN-4s,FCN-2s,但作者表明FCN-8s效果最好，
增加更多的跳跃连接不会带来提升。
![FCN](pics/FCN.png "FCN网络结构示意图")

此网络训练完成后，训练集误差很小，大约在1e-2至1e-3，然而验证集误差却维持在0.3以上。
出现这种情况的原因是过拟合，网络设置卷积层和参数很多，学习能力很强，而数据集却只有几百张图片，导致了过拟合，下面展示了此网络的训练集及验证集结果：

![FCNnet1_train](pics/net1_train.png "FCNnet1训练集结果")
![FCNnet1_val](pics/net1_val.png "FCNnet1验证集结果")

为了解决过拟合问题，我们修改上述FCN-8s，降低参数量得到下面的net2。

### FCN-8s(net2)
相比与net1，net2将初始步的通道数由32减为8，之后的通道数也依次减少，总参数量大幅下降。

然而在训练中，训练集误差稳定在0.12左右，验证集误差稳定在0.42左右。两种误差都较大，说明可能有欠拟合且学习中遇到瓶颈（也可能是学习率太大或网络结构有问题）。
如下图所示：

![FCNnet2_train](pics/net2_train.png "FCNnet2训练集结果")
![FCNnet2_val](pics/net2_val.png "FCNnet2验证集结果")

### FCN-8s(net3)
由上面的结果发现网络的结构不能太简单也不能太复杂，针对同一个数据集，如何平衡好过拟合与欠拟合的问题十分重要。
经过多次尝试与参数调整后，最终设置的FCN-8s结构见[FCN_network.py](Pix2Pix/FCN_network.py)。

另外还添加了脚本[plot_loss.py](Pix2Pix/plot_loss.py)用于绘制loss曲线。



---
## 总结与不足
从网络结构和调整过程来看，主要出现了以下问题：

1.输出层的激活函数选择不当导致训练loss或验证loss不收敛。

2.网络层数及通道数选择不当导致参数过多或过少，出现过拟合或欠拟合的现象。可以适当调整网络结构并引入正则项权重衰减。

3.训练误差或验证误差出现振荡或不下降的情况，需要适当调整batchsize或学习率以平衡振动和收敛缓慢的问题。

本次实验的一些不足之处：

没有使用更多的数据集来提高模型的泛化能力。其中主要是本人的设备性能有限，训练一次网络时间较长，再加上其它数据集占用内存很大，故没有做更进一步的尝试。

## Reference and Acknowledgement
>📋 Thanks for the algorithms proposed by [Paper: Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)
> 
>   [Paper: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

> 其他参考资料：DIP课程课件，Pytorch相关教程
> 
> [知乎相关文章1](https://zhuanlan.zhihu.com/p/401217834)
> 
> [知乎相关文章2](https://zhuanlan.zhihu.com/p/285601835)
> 
> [知乎相关文章3](https://zhuanlan.zhihu.com/p/622943295)
> 
> （包括如何处理欠拟合、过拟合等问题）
