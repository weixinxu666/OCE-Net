import torch
import torch.nn as nn
from torch.nn import init
import math
from models.self_att_parts import *
from models.OCE_DNL import NonLocal2d_bn


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class CBAM_channel(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_channel, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out





#------------------------APF Conv-------------------
class APF_conv(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF_conv, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        # self.eca = nn.Conv2d(out_ch, out_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        out = self.fusion(x4)
        # out = self.eca(x4)
        out = self.gamma*out
        return out
#------------------------APF Conv-------------------






#------------------------APF Normal-------------------
class up(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up(x)


class eca(nn.Module):
    def __init__(self, k_size=3):
        super(eca, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        z = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(z).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)





class APF(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.eca = eca()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)
        out = self.eca(x4)
        out = self.gamma*out + x4
        return out
#------------------------APF Normal-------------------





# -------------------------------APF Self Attention--------------
class eca_self_att(nn.Module):
    def __init__(self, in_ch):
        super(eca_self_att, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2, bias=False)
        self.self_att = Self_Attn(in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # z = y.squeeze(-1).transpose(-1, -2)
        st_out, _ = self.self_att(y)
        y = st_out
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class APF_self_att(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF_self_att, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.eca = eca_self_att(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)
        out = self.eca(x4)
        out = self.gamma*out + x4
        return out
# -------------------------------APF Self Attention--------------







#------------------------APF DNL-------------------------
class eca_DNL(nn.Module):
    def __init__(self, in_ch):
        super(eca_DNL, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2, bias=False)
        self.DNL = NonLocal2d_bn(in_ch, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # z = y.squeeze(-1).transpose(-1, -2)
        st_out = self.DNL(x)
        y = st_out
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class APF_DNL(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF_DNL, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.eca = eca_DNL(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)
        out = self.eca(x4)
        out = self.gamma*out + x4
        return out
#------------------------APF DNL-------------------------









#------------------------APF DNL Channel Aware-------------------------
class NonLocalNd_bn_cbam(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out, out_bn, whiten_type, temperature,
                 with_gc, with_unary):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        # assert whiten_type in ['channel', 'spatial']
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(NonLocalNd_bn_cbam, self).__init__()
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
        if 'bn_affine' in whiten_type:
            self.key_bn_affine = nn.BatchNorm1d(planes)
            self.query_bn_affine = nn.BatchNorm1d(planes)
        if 'bn' in whiten_type:
            self.key_bn = nn.BatchNorm1d(planes, affine=False)
            self.query_bn = nn.BatchNorm1d(planes, affine=False)
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.temperature = temperature
        self.with_gc = with_gc
        self.with_unary = with_unary

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

        self.cbam_channel = ChannelGate(inplanes)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        # init.constant_(self.norm.weight, 0)
        # init.constant_(self.norm.bias, 0)
        # self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            # print('not change lr_mult')
            pass

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        # x_qk = self.cbam_channel(x)
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]

        query = self.conv_query(x)
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        # if 'channel' in self.whiten_type:
        #     key_mean = key.mean(2).unsqueeze(2)
        #     query_mean = query.mean(2).unsqueeze(2)
        #     key -= key_mean
        #     query -= query_mean
        if 'spatial' in self.whiten_type:
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type:
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)
        if 'ln_nostd' in self.whiten_type:
            key_mean = key.mean(1).mean(1).view(key.size(0), 1, 1)
            query_mean = query.mean(1).mean(1).view(query.size(0), 1, 1)
            key -= key_mean
            query -= query_mean

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map / self.scale
        sim_map = sim_map / self.temperature
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        # if self.norm is not None:
        #     out = self.norm(out)
        out_sim = self.gamma * out_sim

        if self.with_unary:
            if query_mean.shape[1] == 1:
                # query_mean = self.cbam_channel(query_mean)
                # key = self.cbam_channel(key)
                query_mean = query_mean.expand(-1, key.shape[1], -1)
            unary = torch.bmm(query_mean.transpose(1, 2), key)
            unary = self.softmax(unary)
            out_unary = torch.bmm(value, unary.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_unary

        # out = residual + out_sim

        if self.with_gc:
            # [N, 1, H', W']
            mask = self.conv_mask(input_x)
            # [N, 1, H'x W']
            mask = mask.view(mask.size(0), mask.size(1), -1)
            mask = self.softmax(mask)
            # [N, C', 1, 1]
            out_gc = torch.bmm(value, mask.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_gc

        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)

        out = out_sim + residual

        return out

class NonLocal2d_bn_cbam(NonLocalNd_bn_cbam):

    def __init__(self, inplanes, planes, downsample=True, use_gn=False, lr_mult=None, use_out=False, out_bn=False,
                 whiten_type=['channel'], temperature=1.0, with_gc=False, with_unary=False):
        super(NonLocal2d_bn_cbam, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample,
                                            use_gn=use_gn, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn,
                                            whiten_type=whiten_type, temperature=temperature, with_gc=with_gc,
                                            with_unary=with_unary)













class eca_DNL_cbam(nn.Module):
    def __init__(self, in_ch):
        super(eca_DNL_cbam, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2, bias=False)
        self.DNL = NonLocal2d_bn_cbam(in_ch, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # z = y.squeeze(-1).transpose(-1, -2)
        st_out = self.DNL(x)
        y = st_out
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class APF_DNL_cbam(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF_DNL_cbam, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.eca = eca_DNL_cbam(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)
        out = self.eca(x4)
        out = self.gamma*out + x4
        return out




class APF_DNL_cbam_fusion(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF_DNL_cbam_fusion, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up2_dc = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.up3_dc = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.eca = eca_DNL_cbam(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3, x1_dc, x2_dc, x3_dc):
        x2 = self.up2(x2)
        x2_dc = self.up2_dc(x2_dc)
        x3 = self.up3(x3)
        x3_dc = self.up3_dc(x3_dc)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)
        out = self.eca(x4)
        out = self.gamma*out + x4
        return out


#------------------------APF DNL Channel Aware-------------------------





#------------------------APF DNL Channel Aware Skip-------------------------
class NonLocalNd_bn_cbam_skip(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out, out_bn, whiten_type, temperature,
                 with_gc, with_unary):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        # assert whiten_type in ['channel', 'spatial']
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(NonLocalNd_bn_cbam_skip, self).__init__()
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
        if 'bn_affine' in whiten_type:
            self.key_bn_affine = nn.BatchNorm1d(planes)
            self.query_bn_affine = nn.BatchNorm1d(planes)
        if 'bn' in whiten_type:
            self.key_bn = nn.BatchNorm1d(planes, affine=False)
            self.query_bn = nn.BatchNorm1d(planes, affine=False)
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.temperature = temperature
        self.with_gc = with_gc
        self.with_unary = with_unary

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

        self.cbam_channel = ChannelGate(inplanes)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        # init.constant_(self.norm.weight, 0)
        # init.constant_(self.norm.bias, 0)
        # self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            # print('not change lr_mult')
            pass

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        x_qk = self.cbam_channel(x)
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]

        query = self.conv_query(x_qk)
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        if 'channel' in self.whiten_type:
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean
        if 'spatial' in self.whiten_type:
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type:
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)
        if 'ln_nostd' in self.whiten_type:
            key_mean = key.mean(1).mean(1).view(key.size(0), 1, 1)
            query_mean = query.mean(1).mean(1).view(query.size(0), 1, 1)
            key -= key_mean
            query -= query_mean

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map / self.scale
        sim_map = sim_map / self.temperature
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        # if self.norm is not None:
        #     out = self.norm(out)
        out_sim = self.gamma * out_sim

        if self.with_unary:
            if query_mean.shape[1] == 1:
                query_mean = self.cbam_channel(query_mean)
                key = self.cbam_channel(key)
                query_mean = query_mean.expand(-1, key.shape[1], -1)
            unary = torch.bmm(query_mean.transpose(1, 2), key)
            unary = self.softmax(unary)
            out_unary = torch.bmm(value, unary.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_unary

        # out = residual + out_sim

        if self.with_gc:
            # [N, 1, H', W']
            mask = self.conv_mask(input_x)
            # [N, 1, H'x W']
            mask = mask.view(mask.size(0), mask.size(1), -1)
            mask = self.softmax(mask)
            # [N, C', 1, 1]
            out_gc = torch.bmm(value, mask.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_gc

        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)

        out = out_sim + residual

        return out

class NonLocal2d_bn_cbam_skip(NonLocalNd_bn_cbam_skip):

    def __init__(self, inplanes, planes, downsample=True, use_gn=False, lr_mult=None, use_out=False, out_bn=False,
                 whiten_type=['channel'], temperature=1.0, with_gc=False, with_unary=True):
        super(NonLocal2d_bn_cbam_skip, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample,
                                            use_gn=use_gn, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn,
                                            whiten_type=whiten_type, temperature=temperature, with_gc=with_gc,
                                            with_unary=with_unary)



class eca_DNL_cbam_skip(nn.Module):
    def __init__(self, in_ch):
        super(eca_DNL_cbam_skip, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2, bias=False)
        self.DNL = NonLocal2d_bn_cbam_skip(in_ch, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # y = self.avg_pool(x)
        # z = y.squeeze(-1).transpose(-1, -2)
        st_out = self.DNL(x)
        y = st_out
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class APF_DNL_cbam_skip(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF_DNL_cbam_skip, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.eca = eca_DNL_cbam_skip(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)
        out = self.eca(x4)
        out = self.gamma*out + x4
        return out

#------------------------APF DNL Channel Aware Skip-------------------------














#------------------------APF DNL Channel Aware DC-------------------------
class NonLocalNd_bn_cbam_dc(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out, out_bn, whiten_type, temperature,
                 with_gc, with_unary):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        # assert whiten_type in ['channel', 'spatial']
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(NonLocalNd_bn_cbam_dc, self).__init__()

        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)

        self.conv_query_dc = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key_dc = conv_nd(inplanes, planes, kernel_size=1)

        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
        if 'bn_affine' in whiten_type:
            self.key_bn_affine = nn.BatchNorm1d(planes)
            self.query_bn_affine = nn.BatchNorm1d(planes)
        if 'bn' in whiten_type:
            self.key_bn = nn.BatchNorm1d(planes, affine=False)
            self.query_bn = nn.BatchNorm1d(planes, affine=False)

        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.temperature = temperature
        self.with_gc = with_gc
        self.with_unary = with_unary


        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

        self.cbam_channel = ChannelGate(inplanes)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        # init.constant_(self.norm.weight, 0)
        # init.constant_(self.norm.bias, 0)
        # self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            # print('not change lr_mult')
            pass

    def forward(self, x, x_dc):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        x_qk = self.cbam_channel(x)
        x_qk_dc = self.cbam_channel(x_dc)
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]

        query = self.conv_query(x_qk)
        # [N, C', T, H', W']
        key = self.conv_key(x_qk)

        query_dc = self.conv_query_dc(x_qk_dc)
        # [N, C', T, H', W']
        key_dc = self.conv_key_dc(x_qk_dc)

        value = self.conv_value(x_qk)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)

        # [N, C', H x W]
        query_dc = query_dc.view(query_dc.size(0), query_dc.size(1), -1)
        # [N, C', H' x W']
        key_dc = key_dc.view(key_dc.size(0), key_dc.size(1), -1)

        value = value.view(value.size(0), value.size(1), -1)

        if 'channel' in self.whiten_type:
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean

        if 'spatial' in self.whiten_type:
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type:
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)
        if 'ln_nostd' in self.whiten_type:
            key_mean = key.mean(1).mean(1).view(key.size(0), 1, 1)
            query_mean = query.mean(1).mean(1).view(query.size(0), 1, 1)
            key -= key_mean
            query -= query_mean

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map / self.scale
        sim_map = sim_map / self.temperature
        sim_map = self.softmax(sim_map)

        sim_map_dc = torch.bmm(query_dc.transpose(1, 2), key_dc)
        sim_map_dc = sim_map_dc / self.scale
        sim_map_dc = sim_map_dc / self.temperature
        sim_map_dc = self.softmax(sim_map_dc)


        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        out_sim = torch.bmm(sim_map_dc, out_sim)


        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        # if self.norm is not None:
        #     out = self.norm(out)
        out_sim = self.gamma * out_sim

        if self.with_unary:
            if query_mean.shape[1] == 1:
                query_mean = self.cbam_channel(query_mean)
                key = self.cbam_channel(key)
                query_mean = query_mean.expand(-1, key.shape[1], -1)
            unary = torch.bmm(query_mean.transpose(1, 2), key)
            unary = self.softmax(unary)


            out_unary = torch.bmm(value, unary.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_unary

        # out = residual + out_sim

        if self.with_gc:
            # [N, 1, H', W']
            mask = self.conv_mask(input_x)
            # [N, 1, H'x W']
            mask = mask.view(mask.size(0), mask.size(1), -1)
            mask = self.softmax(mask)
            # [N, C', 1, 1]
            out_gc = torch.bmm(value, mask.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_gc

        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)

        out = out_sim + residual

        return out

class NonLocal2d_bn_cbam_dc(NonLocalNd_bn_cbam_dc):

    def __init__(self, inplanes, planes, downsample=True, use_gn=False, lr_mult=None, use_out=False, out_bn=False,
                 whiten_type=['channel'], temperature=1.0, with_gc=False, with_unary=True):
        super(NonLocal2d_bn_cbam_dc, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample,
                                            use_gn=use_gn, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn,
                                            whiten_type=whiten_type, temperature=temperature, with_gc=with_gc,
                                            with_unary=with_unary)



class eca_DNL_cbam_dc(nn.Module):
    def __init__(self, in_ch):
        super(eca_DNL_cbam_dc, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2, bias=False)
        self.DNL = NonLocal2d_bn_cbam_dc(in_ch, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_dc):
        # y = self.avg_pool(x)
        # z = y.squeeze(-1).transpose(-1, -2)
        st_out = self.DNL(x, x_dc)
        y = st_out
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class APF_DNL_cbam_dc(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF_DNL_cbam_dc, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        # self.fusion3 = nn.Conv2d(out_ch * 2, out_ch, 1)
        self.eca = eca_DNL_cbam_dc(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3, x1_dc, x2_dc, x3_dc):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)

        x2_dc = self.up2(x2_dc)
        x3_dc = self.up3(x3_dc)
        x4_dc = torch.cat([x1_dc, x2_dc, x3_dc], dim=1)
        x4_dc = self.fusion(x4_dc)

        out = self.eca(x4, x4_dc)
        out = self.gamma*out + x4
        return out

#------------------------APF DNL Channel Aware DC-------------------------







#------------------------APF DNL Channel Aware DC2-------------------------
class NonLocalNd_bn_cbam_dc2(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out, out_bn, whiten_type, temperature,
                 with_gc, with_unary):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        # assert whiten_type in ['channel', 'spatial']
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(NonLocalNd_bn_cbam_dc2, self).__init__()

        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)

        self.conv_query_dc = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key_dc = conv_nd(inplanes, planes, kernel_size=1)

        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
        if 'bn_affine' in whiten_type:
            self.key_bn_affine = nn.BatchNorm1d(planes)
            self.query_bn_affine = nn.BatchNorm1d(planes)
        if 'bn' in whiten_type:
            self.key_bn = nn.BatchNorm1d(planes, affine=False)
            self.query_bn = nn.BatchNorm1d(planes, affine=False)

        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.temperature = temperature
        self.with_gc = with_gc
        self.with_unary = with_unary


        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

        # self.cbam_channel = ChannelGate(inplanes)
        # self.cbam_channel2 = ChannelGate(inplanes)

        # self.fusion = nn.Conv2d(F_l*4, F_int*2, kernel_size=1)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        # init.constant_(self.norm.weight, 0)
        # init.constant_(self.norm.bias, 0)
        # self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            # print('not change lr_mult')
            pass

    def forward(self, x, x_dc):

        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        # x_qk = self.cbam_channel(x)
        # x_qk_dc = self.cbam_channel2(x_dc)

        x_qk = x
        x_qk_dc = x_dc

        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]

        query = self.conv_query(x_qk)
        # [N, C', T, H', W']
        key = self.conv_key(x_qk)

        query_dc = self.conv_query(x_qk_dc)
        # [N, C', T, H', W']
        key_dc = self.conv_key(x_qk_dc)

        value = self.conv_value(x_qk)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)

        # [N, C', H x W]
        query_dc = query_dc.view(query_dc.size(0), query_dc.size(1), -1)
        # [N, C', H' x W']
        key_dc = key_dc.view(key_dc.size(0), key_dc.size(1), -1)

        value = value.view(value.size(0), value.size(1), -1)

        # if 'channel' in self.whiten_type:
        #     key_mean = key.mean(2).unsqueeze(2)
        #     query_mean = query.mean(2).unsqueeze(2)
        #     key -= key_mean
        #     query -= query_mean
        #
        #     key_mean_dc = key_dc.mean(2).unsqueeze(2)
        #     query_mean_dc = query_dc.mean(2).unsqueeze(2)
        #     key_dc -= key_mean_dc
        #     query_dc -= query_mean_dc


        if 'spatial' in self.whiten_type:
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type:
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)
        if 'ln_nostd' in self.whiten_type:
            key_mean = key.mean(1).mean(1).view(key.size(0), 1, 1)
            query_mean = query.mean(1).mean(1).view(query.size(0), 1, 1)
            key -= key_mean
            query -= query_mean

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map / self.scale
        sim_map = sim_map / self.temperature
        # sim_map = self.softmax(sim_map)

        # sim_map_dc = torch.bmm(query_dc.transpose(1, 2), key_dc)
        # sim_map_dc = sim_map_dc / self.scale
        # sim_map_dc = sim_map_dc / self.temperature

        # sim_map_dc = sim_map_dc * sim_map
        # sim_map_dc = self.softmax(sim_map_dc)

        sim_map_dc1 = torch.bmm(query_dc.transpose(1, 2), key)
        sim_map_dc1 = sim_map_dc1 / self.scale
        sim_map_dc1 = sim_map_dc1 / self.temperature
        sim_map_dc1 = self.softmax(sim_map_dc1)
        #
        sim_map_dc2 = torch.bmm(query.transpose(1, 2), key_dc)
        sim_map_dc2 = sim_map_dc2 / self.scale
        sim_map_dc2 = sim_map_dc2 / self.temperature
        sim_map_dc2 = self.softmax(sim_map_dc2)


        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        # out_sim = torch.bmm(sim_map_dc, out_sim)
        out_sim = torch.bmm(sim_map_dc1, out_sim)
        out_sim = torch.bmm(sim_map_dc2, out_sim)

        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        # if self.norm is not None:
        #     out = self.norm(out)
        out_sim = self.gamma * out_sim

        if self.with_unary:
            if query_mean.shape[1] == 1:
                # query_mean = self.cbam_channel(query_mean)
                # key = self.cbam_channel(key)
                query_mean = query_mean.expand(-1, key.shape[1], -1)
                query_mean_dc = query_mean_dc.expand(-1, key_dc.shape[1], -1)

            unary = torch.bmm(query_mean.transpose(1, 2), key)
            unary_dc = torch.bmm(query_mean_dc.transpose(1, 2), key_dc)
            unary_all = unary * unary_dc
            unary = self.softmax(unary_all)

            out_unary = torch.bmm(value, unary.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_unary

        # out = residual + out_sim

        if self.with_gc:
            # [N, 1, H', W']
            mask = self.conv_mask(input_x)
            # [N, 1, H'x W']
            mask = mask.view(mask.size(0), mask.size(1), -1)
            mask = self.softmax(mask)
            # [N, C', 1, 1]
            out_gc = torch.bmm(value, mask.permute(0, 2, 1)).unsqueeze(-1)
            out_sim = out_sim + out_gc

        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)

        out = out_sim + residual

        return out

class NonLocal2d_bn_cbam_dc2(NonLocalNd_bn_cbam_dc2):

    def __init__(self, inplanes, planes, downsample=True, use_gn=False, lr_mult=None, use_out=False, out_bn=False,
                 whiten_type=['channel'], temperature=1.0, with_gc=False, with_unary=False):
        super(NonLocal2d_bn_cbam_dc2, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample,
                                            use_gn=use_gn, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn,
                                            whiten_type=whiten_type, temperature=temperature, with_gc=with_gc,
                                            with_unary=with_unary)



class eca_DNL_cbam_dc2(nn.Module):
    def __init__(self, in_ch):
        super(eca_DNL_cbam_dc2, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2, bias=False)
        self.DNL = NonLocal2d_bn_cbam_dc2(in_ch, in_ch)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, x_dc):
        # y = self.avg_pool(x)
        # z = y.squeeze(-1).transpose(-1, -2)
        st_out = self.DNL(x, x_dc)
        y = st_out
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class APF_DNL_cbam_dc2(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(APF_DNL_cbam_dc2, self).__init__()
        self.out_ch = out_ch
        self.up2 = up(in_ch2, out_ch, scale_factor=2)
        self.up3 = up(in_ch3, out_ch, scale_factor=4)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.conv1x1 = nn.Conv2d(out_ch * 2, out_ch, 1)
        # self.fusion3 = nn.Conv2d(out_ch * 2, out_ch, 1)
        # self.eca = eca_DNL_cbam_dc2(out_ch)
        self.eca = eca_DNL(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3, x1_dc, x2_dc, x3_dc):
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.fusion(x4)

        x2_dc = self.up2(x2_dc)
        x3_dc = self.up3(x3_dc)
        x4_dc = torch.cat([x1_dc, x2_dc, x3_dc], dim=1)
        x4_dc = self.fusion(x4_dc)


        # out = self.eca(x4 + x4_dc)
        out = self.eca(self.conv1x1(torch.cat([x4, x4_dc], dim=1)))
        # out = self.eca(x4, x4_dc)
        out = self.gamma*out + x4
        return out

#------------------------APF DNL Channel Aware dc2-------------------------





if __name__ == '__main__':
    img1 = torch.randn(3, 64, 64, 3)
    img2 = torch.randn(3, 32, 32, 32)
    img3 = torch.randn(3, 16, 16, 16)
    img1 = img1.cuda()
    img2 = img2.cuda()
    img3 = img3.cuda()
    img1 = img1.permute(0,3,1,2)
    img2 = img2.permute(0,3,1,2)
    img3 = img3.permute(0,3,1,2)
    model = APF_DNL_cbam_dc2(in_ch1=64, in_ch2=32, in_ch3=16, out_ch=3)
    # model = APF_conv(in_ch1=64, in_ch2=32, in_ch3=16, out_ch=3)
    # model = APF(in_ch1=64, in_ch2=32, in_ch3=16, out_ch=3)
    model = model.cuda()
    out = model(img1, img2, img3, img1, img2, img3)
    print(out.shape)
    print('ok')

#
# if __name__ == '__main__':
#     img1 = torch.randn(3, 64, 64, 3)