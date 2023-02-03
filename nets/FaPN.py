# from torch import nn
# class FaPN(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(FaPN, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x + x * y.expand_as(x)

from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional  as F
class FaPN(nn.Module):
    """
    Squeeze:1.对输入的特征图做自适应全局平均池化
            2.然后在打平，选择最简单的全局平均池化，使其具有全局的感受野，使网络底层也能利用全局信息。
    Excitation:1.使用全连接层，对Squeeze的结果做非线性转化，它是类似于神经网络中门的机制，
                 通过参数来为每个特整层生成相应的权重，其中参数学习用来显示地建立通道间的，
                 相关性
    特征重标定：使用Excitation得到的结果为权重，乘以输入的特征图
    """
    def __init__(self, channel, reduction=16):

        super(FaPN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True))
        self.fc2 =nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        # (b, c, h, w) --> (b, c, 1, 1)
        y = self.avg_pool(x)
        # (b, c, 1, 1) --> (b, c*1*1)
        y = y.view(b, c)
        # 压缩
        y = self.fc1(y)
        # 扩张
        y = self.fc2(y)
        y = y.view(b, c, 1, 1)
        # 伸张维度
        return x * y.expand_as(x) + x
