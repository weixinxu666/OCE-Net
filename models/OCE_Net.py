"""
This part contains UNet series models,
including UNet, R2UNet, Attention UNet, R2Attention UNet, DenseUNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os

from models.sk_att_fusion import SKConv_my as SKConv
from models.DCOA_Conv import Dynamic_conv2d
from models.MSFM_parts import APF, APF_DNL, APF_DNL_cbam, APF_DNL_cbam_dc
from models.GLFM_parts import GLFM
from models.UARM import Refine_block, PALayer_shallow


os.environ["CUDA_VISIBLE_DEVICES"] = "0"




class encoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(encoder, self).__init__()
        self.conv = nn.Sequential(
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class conv_block_dcoa(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_dcoa, self).__init__()
        self.conv = nn.Sequential(
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            Dynamic_conv2d(ch_out, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_dcoa_single(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_dcoa_single, self).__init__()
        self.conv = nn.Sequential(
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # Dynamic_conv2d(ch_out, ch_out, kernel_size=3, ratio=0.25, padding=1),
            # nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv_dcoa(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_dcoa, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x



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



# ==========================Core Module================================
class conv_block_sk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_sk, self).__init__()
        self.conv = nn.Sequential(
            SKConv(ch_in, ch_in, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            SKConv(ch_out, ch_out, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv_sk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_sk, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            SKConv(ch_in, ch_in, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x






# Main Model ---- the proposed OCE-Net

class OCENet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, downsize = 2):
        super(OCENet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//downsize)    # plain conv
        self.Conv1_dc = conv_block_dcoa(ch_in=64//downsize, ch_out=64//downsize)    # Dynamic Complex Orientation Aware Convolution DCOA Conv
        self.Conv1_sk = SKConv(64 // downsize)    #  use SK Attention module to fuse the DCOA features (orientation features) and the plain features
        # self.Conv1_sk = nn.Conv2d(64//downsize, 64//downsize, kernel_size=1)   # the performance will drop if we use conv instead of SK_conv
        self.Fusion1 = conv_block(ch_in=(64 + 64) // downsize, ch_out=64 // downsize)


        self.Conv2 = conv_block(ch_in=64//downsize, ch_out=128//downsize)
        self.Conv2_dc = conv_block_dcoa(ch_in=128//downsize, ch_out=128//downsize)
        self.Conv2_sk = SKConv(128//downsize)
        # self.Conv2_sk = nn.Conv2d(128 // downsize, 128 // downsize, kernel_size=1)
        self.Fusion2 = conv_block(ch_in=(128 + 128) // downsize, ch_out=128 // downsize)

        self.Conv3 = conv_block(ch_in=128//downsize, ch_out=256//downsize)   #plain conv
        self.Conv3_dc = conv_block_dcoa(ch_in=256//downsize, ch_out=256//downsize)    # DCOA Conv
        self.Conv3_sk = SKConv(256 // downsize)
        # self.Conv3_sk = nn.Conv2d(256 // downsize, 256 // downsize, kernel_size=1)
        self.Fusion3 = conv_block(ch_in=(256 + 256) // downsize, ch_out=256 // downsize)

        self.Conv4 = conv_block(ch_in=256//downsize, ch_out=512//downsize)
        # self.Conv5 = conv_block(ch_in=512//downsize, ch_out=1024//downsize)
        #
        # self.Up5 = up_conv(ch_in=1024//downsize, ch_out=512//downsize)
        # self.Att5 = GLFM(F_g=512 // downsize, F_l=512 // downsize, F_int=256 // downsize)    # Global and Local Fusion Module (GLFM)
        # self.Up_conv5 = conv_block(ch_in=1024//downsize, ch_out=512//downsize)

        self.Up4 = up_conv(ch_in=512//downsize, ch_out=256//downsize)
        self.Att4 = GLFM(F_g=256 // downsize, F_l=256 // downsize, F_int=128 // downsize)   #   # Global and Local Fusion Module (GLFM)
        self.Up_conv4 = conv_block(ch_in=512//downsize, ch_out=256//downsize)

        self.Up3 = up_conv(ch_in=256//downsize, ch_out=128//downsize)
        self.Up3_dc = up_conv_dcoa(ch_in=256//downsize, ch_out=128//downsize)    # DCOA UP_CONV
        self.Up3_sk = SKConv(256//downsize)
        # self.Up3_sk = nn.Conv2d(256 // downsize, 256 // downsize, kernel_size=1)
        self.Att3 = GLFM(F_g=128 // downsize, F_l=128 // downsize, F_int=64 // downsize)
        self.Up_conv3 = conv_block_dcoa(ch_in=256//downsize, ch_out=128//downsize)

        self.Up2 = up_conv(ch_in=128//downsize, ch_out=64//downsize)
        self.Up2_dc = up_conv_dcoa(ch_in=128//downsize, ch_out=64//downsize)
        self.Up2_sk = SKConv(128//downsize)
        # self.Up2_sk = nn.Conv2d(128 // downsize, 128 // downsize, kernel_size=1)
        self.Att2 = GLFM(F_g=64 // downsize, F_l=64 // downsize, F_int=32 // downsize)
        self.Up_conv2 = conv_block_dcoa(ch_in=128//downsize, ch_out=64//downsize)

        self.refine = Refine_block(64 // downsize)    # UARM

        self.Conv_1x1 = nn.Conv2d(64//downsize, output_ch, kernel_size=1, stride=1, padding=0)

        self.conv_prob = nn.Conv2d(64 // downsize, 1, kernel_size=1, stride=1, padding=0)


        # self.OCE_NL = APF_DNL_cbam(in_ch1=64 // downsize, in_ch2=128 // downsize,
        #                         in_ch3=256 // downsize, out_ch=64 // downsize)    #  Orientation and Context Entangled Non-local (OCE-NL)

        self.OCE_NL = APF_DNL_cbam_dc(in_ch1=64 // downsize, in_ch2=128 // downsize,
                                in_ch3=256 // downsize, out_ch=64 // downsize)    #  Orientation and Context Entangled Non-local (OCE-NL)



    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1_dcoa = self.Conv1_dc(x1)
        x1 = self.Conv1_sk(x1, x1_dcoa)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2_dcoa = self.Conv2_dc(x2)
        x2 = self.Conv2_sk(x2, x2_dcoa)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3_dcoa = self.Conv3_dc(x3)
        x3 = self.Conv3_sk(x3, x3_dcoa)

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
        # d4 = torch.cat((x3_dc, d4), dim=1)  #  use plain concat to conduct fusion in the upsampling state (poor performance)
        d4_sk = self.Up3_sk(d4, x3_dcoa)    #  use sk attention to conduct fusion in the upsampling state (good performance)
        d4 = torch.cat((x3_dcoa, d4), dim=1)
        d4 = self.Fusion3(d4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4_sk), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # d3 = torch.cat((x2_dc, d3), dim=1)
        d3_sk = self.Up2_sk(d3, x2_dcoa)
        d3 = torch.cat((x2_dcoa, d3), dim=1)
        d3 = self.Fusion2(d3)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3_sk), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # d2 = torch.cat((x1_dc, d2), dim=1)
        d2_sk = self.Conv1_sk(d2, x1_dcoa)
        d2 = torch.cat((x1_dcoa, d2), dim=1)
        d2 = self.Fusion1(d2)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2_sk), dim=1)
        d2 = self.Up_conv2(d2)


        d2 = self.OCE_NL(d2, d3, d4, x1_dcoa, x2_dcoa, x3_dcoa)
        # d2 = self.apf(d2, d3, d4)

        prob_feat = self.conv_prob(d2)
        prob_map = F.softmax(prob_feat, dim=1)  # Batch 1  H   W

        d2 = self.refine(d2, prob_map)    # UARM refine

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)

        return d1






if __name__ == '__main__':

    net = OCENet(1,2).cuda()

    # print(net)
    in1 = torch.randn((4,1,48,48)).cuda()
    out1 = net(in1)
    print(out1.size())


