import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import torch
from ..utils import ConvModule
from ..registry import NECKS

class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat

class mafa(nn.Module):
    def __init__(self,
                 in_chan,
                 out_chan,
                 num_outs,
                 pool_ratios=[0.1,0.2,0.3],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(mafa, self).__init__()
        self.FSM = FeatureSelectionModule(in_chan, out_chan)
        # self.AAP1 = torch.nn.AdaptiveAvgPool2d()
        # self.AAP2 = torch.nn.AdaptiveAvgPool2d()
        # self.AAP3 = torch.nn.AdaptiveAvgPool2d()
        # self.Upsample = torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)

    def forward(self, x):
        n, c, h, w = x.sizie()
        AAP1 = torch.nn.AdaptiveAvgPool2d(0.1 * h)
        AAP2 = torch.nn.AdaptiveAvgPool2d(0.2 * h)
        AAP3 = torch.nn.AdaptiveAvgPool2d(0.3 * h)
        x1 = AAP1(x)
        x2 = AAP2(x)
        x3 = AAP3(x)

        x1 = F.interpolate(x1, [h, w])
        x2 = F.interpolate(x2, [h, w])
        x3 = F.interpolate(x3, [h, w])
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)

        x = self.FSM(x)
