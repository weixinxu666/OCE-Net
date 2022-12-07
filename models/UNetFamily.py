"""
This part contains UNet series models, 
including UNet, R2UNet, Attention UNet, R2Attention UNet, DenseUNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os
# from models.self_attention_gate import Attention_block


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================Core Module================================
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_single(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_single, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):  # attention Gate
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

# ==================================================================
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, down_size = 4):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//down_size)
        self.Conv2 = conv_block(ch_in=64//down_size, ch_out=128//down_size)
        self.Conv3 = conv_block(ch_in=128//down_size, ch_out=256//down_size)
        self.Conv4 = conv_block(ch_in=256//down_size, ch_out=512//down_size)
        self.Conv5 = conv_block(ch_in=512//down_size, ch_out=1024//down_size)

        self.Up5 = up_conv(ch_in=1024//down_size, ch_out=512//down_size)
        self.Up_conv5 = conv_block(ch_in=1024//down_size, ch_out=512//down_size)

        self.Up4 = up_conv(ch_in=512//down_size, ch_out=256//down_size)
        self.Up_conv4 = conv_block(ch_in=512//down_size, ch_out=256//down_size)

        self.Up3 = up_conv(ch_in=256//down_size, ch_out=128//down_size)
        self.Up_conv3 = conv_block(ch_in=256//down_size, ch_out=128//down_size)

        self.Up2 = up_conv(ch_in=128//down_size, ch_out=64//down_size)
        self.Up_conv2 = conv_block(ch_in=128//down_size, ch_out=64//down_size)

        self.Conv_1x1 = nn.Conv2d(64//down_size, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)  # mine

        return d1



class U_Net_small(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, down_size = 4):
        super(U_Net_small, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_single(ch_in=img_ch, ch_out=64//down_size)
        self.Conv2 = conv_block_single(ch_in=64//down_size, ch_out=128//down_size)
        self.Conv3 = conv_block_single(ch_in=128//down_size, ch_out=256//down_size)
        self.Conv4 = conv_block_single(ch_in=256//down_size, ch_out=512//down_size)
        # self.Conv5 = conv_block_single(ch_in=512//down_size, ch_out=1024//down_size)
        #
        # self.Up5 = up_conv(ch_in=1024//down_size, ch_out=512//down_size)
        # self.Up_conv5 = conv_block_single(ch_in=1024//down_size, ch_out=512//down_size)

        self.Up4 = up_conv(ch_in=512//down_size, ch_out=256//down_size)
        self.Up_conv4 = conv_block_single(ch_in=512//down_size, ch_out=256//down_size)

        self.Up3 = up_conv(ch_in=256//down_size, ch_out=128//down_size)
        self.Up_conv3 = conv_block_single(ch_in=256//down_size, ch_out=128//down_size)

        self.Up2 = up_conv(ch_in=128//down_size, ch_out=64//down_size)
        self.Up_conv2 = conv_block_single(ch_in=128//down_size, ch_out=64//down_size)

        self.Conv_1x1 = nn.Conv2d(64//down_size, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)
        #
        # # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4, d5), dim=1)
        #
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)  # mine

        return d1


# ============================================================
class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2, downsize = 2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64//downsize, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64//downsize, ch_out=128//downsize, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128//downsize, ch_out=256//downsize, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256//downsize, ch_out=512//downsize, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512//downsize, ch_out=1024//downsize, t=t)

        self.Up5 = up_conv(ch_in=1024//downsize, ch_out=512//downsize)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024//downsize, ch_out=512//downsize, t=t)

        self.Up4 = up_conv(ch_in=512//downsize, ch_out=256//downsize)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512//downsize, ch_out=256//downsize, t=t)

        self.Up3 = up_conv(ch_in=256//downsize, ch_out=128//downsize)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256//downsize, ch_out=128//downsize, t=t)

        self.Up2 = up_conv(ch_in=128//downsize, ch_out=64//downsize)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128//downsize, ch_out=64//downsize, t=t)

        self.Conv_1x1 = nn.Conv2d(64//downsize, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)

        return d1

# ===========================================================
class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, downsize = 2):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//downsize)
        self.Conv2 = conv_block(ch_in=64//downsize, ch_out=128//downsize)
        self.Conv3 = conv_block(ch_in=128//downsize, ch_out=256//downsize)
        self.Conv4 = conv_block(ch_in=256//downsize, ch_out=512//downsize)
        self.Conv5 = conv_block(ch_in=512//downsize, ch_out=1024//downsize)

        self.Up5 = up_conv(ch_in=1024//downsize, ch_out=512//downsize)
        self.Att5 = Attention_block(F_g=512//downsize, F_l=512//downsize, F_int=256//downsize)
        self.Up_conv5 = conv_block(ch_in=1024//downsize, ch_out=512//downsize)

        self.Up4 = up_conv(ch_in=512//downsize, ch_out=256//downsize)
        self.Att4 = Attention_block(F_g=256//downsize, F_l=256//downsize, F_int=128//downsize)
        self.Up_conv4 = conv_block(ch_in=512//downsize, ch_out=256//downsize)

        self.Up3 = up_conv(ch_in=256//downsize, ch_out=128//downsize)
        self.Att3 = Attention_block(F_g=128//downsize, F_l=128//downsize, F_int=64//downsize)
        self.Up_conv3 = conv_block(ch_in=256//downsize, ch_out=128//downsize)

        self.Up2 = up_conv(ch_in=128//downsize, ch_out=64//downsize)
        self.Att2 = Attention_block(F_g=64//downsize, F_l=64//downsize, F_int=32//downsize)
        self.Up_conv2 = conv_block(ch_in=128//downsize, ch_out=64//downsize)

        self.Conv_1x1 = nn.Conv2d(64//downsize, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)
        return d1



class AttU_Net_small(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, downsize = 4):
        super(AttU_Net_small, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//downsize)
        self.Conv2 = conv_block(ch_in=64//downsize, ch_out=128//downsize)
        self.Conv3 = conv_block(ch_in=128//downsize, ch_out=256//downsize)
        self.Conv4 = conv_block(ch_in=256//downsize, ch_out=512//downsize)
        # self.Conv5 = conv_block(ch_in=512//downsize, ch_out=1024//downsize)
        #
        # self.Up5 = up_conv(ch_in=1024//downsize, ch_out=512//downsize)
        # self.Att5 = Attention_block(F_g=512//downsize, F_l=512//downsize, F_int=256//downsize)
        # self.Up_conv5 = conv_block(ch_in=1024//downsize, ch_out=512//downsize)

        self.Up4 = up_conv(ch_in=512//downsize, ch_out=256//downsize)
        self.Att4 = Attention_block(F_g=256//downsize, F_l=256//downsize, F_int=128//downsize)
        self.Up_conv4 = conv_block(ch_in=512//downsize, ch_out=256//downsize)

        self.Up3 = up_conv(ch_in=256//downsize, ch_out=128//downsize)
        self.Att3 = Attention_block(F_g=128//downsize, F_l=128//downsize, F_int=64//downsize)
        self.Up_conv3 = conv_block(ch_in=256//downsize, ch_out=128//downsize)

        self.Up2 = up_conv(ch_in=128//downsize, ch_out=64//downsize)
        self.Att2 = Attention_block(F_g=64//downsize, F_l=64//downsize, F_int=32//downsize)
        self.Up_conv2 = conv_block(ch_in=128//downsize, ch_out=64//downsize)

        self.Conv_1x1 = nn.Conv2d(64//downsize, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)
        #
        # # decoding + concat path
        # d5 = self.Up5(x5)
        # x4 = self.Att5(g=d5, x=x4)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)
        return d1



# ==============================================================
class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2, downsize=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64//downsize, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64//downsize, ch_out=128//downsize, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128//downsize, ch_out=256//downsize, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256//downsize, ch_out=512//downsize, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512//downsize, ch_out=1024//downsize, t=t)

        self.Up5 = up_conv(ch_in=1024//downsize, ch_out=512//downsize)
        self.Att5 = Attention_block(F_g=512//downsize, F_l=512//downsize, F_int=256//downsize)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024//downsize, ch_out=512//downsize, t=t)

        self.Up4 = up_conv(ch_in=512//downsize, ch_out=256//downsize)
        self.Att4 = Attention_block(F_g=256//downsize, F_l=256//downsize, F_int=128//downsize)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512//downsize, ch_out=256//downsize, t=t)

        self.Up3 = up_conv(ch_in=256//downsize, ch_out=128//downsize)
        self.Att3 = Attention_block(F_g=128//downsize, F_l=128//downsize, F_int=64//downsize)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256//downsize, ch_out=128//downsize, t=t)

        self.Up2 = up_conv(ch_in=128//downsize, ch_out=64//downsize)
        self.Att2 = Attention_block(F_g=64//downsize, F_l=64//downsize, F_int=32//downsize)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128//downsize, ch_out=64//downsize, t=t)

        self.Conv_1x1 = nn.Conv2d(64//downsize, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1, dim=1)

        return d1

#==================DenseUNet=====================================
class Single_level_densenet(nn.Module):
    def __init__(self, filters, num_conv=4):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters, filters, 3, padding=1))
            self.bn_list.append(nn.BatchNorm2d(filters))

    def forward(self, x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final


class Down_sample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Down_sample, self).__init__()
        self.down_sample_layer = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        y = self.down_sample_layer(x)
        return y, x


class Upsample_n_Concat(nn.Module):
    def __init__(self, filters):
        super(Upsample_n_Concat, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding=1, stride=2)
        self.conv = nn.Conv2d(2 * filters, filters, 3, padding=1)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self, x, y):
        x = self.upsample_layer(x)
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.bn(self.conv(x)))
        return x


class Dense_Unet(nn.Module):
    def __init__(self, in_chan=3,out_chan=2,filters=64, num_conv=4):

        super(Dense_Unet, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, filters, 1)
        self.d1 = Single_level_densenet(filters, num_conv)
        self.down1 = Down_sample()
        self.d2 = Single_level_densenet(filters, num_conv)
        self.down2 = Down_sample()
        self.d3 = Single_level_densenet(filters, num_conv)
        self.down3 = Down_sample()
        self.d4 = Single_level_densenet(filters, num_conv)
        self.down4 = Down_sample()
        self.bottom = Single_level_densenet(filters, num_conv)
        self.up4 = Upsample_n_Concat(filters)
        self.u4 = Single_level_densenet(filters, num_conv)
        self.up3 = Upsample_n_Concat(filters)
        self.u3 = Single_level_densenet(filters, num_conv)
        self.up2 = Upsample_n_Concat(filters)
        self.u2 = Single_level_densenet(filters, num_conv)
        self.up1 = Upsample_n_Concat(filters)
        self.u1 = Single_level_densenet(filters, num_conv)
        self.outconv = nn.Conv2d(filters, out_chan, 1)

    #         self.outconvp1 = nn.Conv2d(filters,out_chan, 1)
    #         self.outconvm1 = nn.Conv2d(filters,out_chan, 1)

    def forward(self, x):
        x = self.conv1(x)
        x, y1 = self.down1(self.d1(x))
        x, y2 = self.down1(self.d2(x))
        x, y3 = self.down1(self.d3(x))
        x, y4 = self.down1(self.d4(x))
        x = self.bottom(x)
        x = self.u4(self.up4(x, y4))
        x = self.u3(self.up3(x, y3))
        x = self.u2(self.up2(x, y2))
        x = self.u1(self.up1(x, y1))
        x1 = self.outconv(x)
        #         xm1 = self.outconvm1(x)
        #         xp1 = self.outconvp1(x)
        x1 = F.softmax(x1,dim=1)
        return x1
# =========================================================


class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=1, out_ch=2, downsize=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0]//downsize, filters[0]//downsize)
        self.conv1_0 = conv_block_nested(filters[0]//downsize, filters[1]//downsize, filters[1]//downsize)
        self.conv2_0 = conv_block_nested(filters[1]//downsize, filters[2]//downsize, filters[2]//downsize)
        self.conv3_0 = conv_block_nested(filters[2]//downsize, filters[3]//downsize, filters[3]//downsize)
        self.conv4_0 = conv_block_nested(filters[3]//downsize, filters[4]//downsize, filters[4]//downsize)

        self.conv0_1 = conv_block_nested((filters[0] + filters[1])//downsize, filters[0]//downsize, filters[0]//downsize)
        self.conv1_1 = conv_block_nested((filters[1] + filters[2])//downsize, filters[1]//downsize, filters[1]//downsize)
        self.conv2_1 = conv_block_nested((filters[2] + filters[3])//downsize, filters[2]//downsize, filters[2]//downsize)
        self.conv3_1 = conv_block_nested((filters[3] + filters[4])//downsize, filters[3]//downsize, filters[3]//downsize)

        self.conv0_2 = conv_block_nested((filters[0] * 2 + filters[1])//downsize, filters[0]//downsize, filters[0]//downsize)
        self.conv1_2 = conv_block_nested((filters[1] * 2 + filters[2])//downsize, filters[1]//downsize, filters[1]//downsize)
        self.conv2_2 = conv_block_nested((filters[2] * 2 + filters[3])//downsize, filters[2]//downsize, filters[2]//downsize)

        self.conv0_3 = conv_block_nested((filters[0] * 3 + filters[1])//downsize, filters[0]//downsize, filters[0]//downsize)
        self.conv1_3 = conv_block_nested((filters[1] * 3 + filters[2])//downsize, filters[1]//downsize, filters[1]//downsize)

        self.conv0_4 = conv_block_nested((filters[0] * 4 + filters[1])//downsize, filters[0]//downsize, filters[0]//downsize)

        self.final = nn.Conv2d(filters[0]//downsize, out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNetpp(nn.Module):
    def __init__(self,  input_channels=1, num_classes=2, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output




# if __name__ == '__main__':
#     net = Dense_Unet(3,1,128).cuda()
#     print(net)
#     in1 = torch.randn(4,3,224,224).cuda()
#     out = net(in1)
#     print(out.size())

if __name__ == '__main__':
    import models
    from thop import profile


    # net = models.UNetFamily.U_Net(1,2).to(device)
    # net = models.UNetFamily.U_Net_small(1,2)
    # net = models.UNetFamily.AttU_Net_small(1,2)
    # net = models.UNetFamily.Dense_Unet(1,2, 64)
    # net = models.DR_UNet.U_Net(1,2)

    # net = models.Dynamic_Gabor_UNet.U_Net_small(1,2)
    net = models.unet_apf_self_att.U_Net_apf_self_att_small(1,2)

    # net = models.deform_unet.U_Net_small(1,2)
    # net = models.SK_gabor_UNet.U_Net_all_fusion_small(1,2).cuda()
    # net = models.unet_apf_self_att.U_Net_apf_self_att_small(1,2).to(device)

    # net = U_Net_small(1,2)
    input = torch.randn(1, 1, 48, 48)
    flops, params = profile(net, inputs=(input,))
    print('flops:', flops)
    print('params', params)

    # test network forward
    # net = U_Net(1,2).cuda()

    # net = AttU_Net(1,2).cuda()
    # print(net)
    # in1 = torch.randn((2,1,48,48)).cuda()
    # out1 = net(in1)
    # print(out1.size())