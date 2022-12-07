import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Function


os.environ["CUDA_VISIBLE_DEVICES"] = "2"





class sign_(nn.Module):

    def __init__(self, *kargs, **kwargs):
        super(sign_, self).__init__(*kargs, **kwargs)
        self.r = sign_f.apply  ### <-----注意此处

    def forward(self, inputs):
        outs = self.r(inputs)
        return outs


class sign_f(Function):
    @staticmethod
    def forward(ctx, inputs):
        output = inputs.new(inputs.size())
        output_all = inputs.new(inputs.size())
        output_all[inputs >= 0] = 1
        output[inputs >= 0.7] = 1
        output[inputs < 0.] = 1
        output_inter = output_all - output
        output_inter[inputs>=0] = 1
        ctx.save_for_backward(inputs)
        return output_inter

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_output[input_ > 1.] = 0
        grad_output[input_ < 0] = 0
        return grad_output



class Selector(nn.Module):

    def __init__(self, ):
        super(Selector, self).__init__()
        self.sign = sign_()

    def forward(self, feat, prob):
        prob_map = self.sign(prob)

        out_thick = feat * (1 - prob_map)
        out_thin= feat * prob_map
        return out_thin, out_thick




# def selector(feat, prob, thres=0.8):
#     out_thin = feat * (1 - prob)
#     out_thick= feat * prob
#     return out_thin, out_thick


class PALayer_shallow(nn.Module):
    def __init__(self, channel):
        super(PALayer_shallow, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y



class PALayer_deep(nn.Module):
    def __init__(self, channel):
        super(PALayer_deep, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y



class Refine_block(nn.Module):
    def __init__(self, channel):
        super(Refine_block, self).__init__()
        self.pa_shallow = PALayer_shallow(channel)
        self.pa_deep = PALayer_deep(channel)

        self.selector = Selector()

    def forward(self, x, prob):
        thin, thick = self.selector(x, prob)
        thin_att = self.pa_deep(thin)
        thick_att = self.pa_shallow(thick)
        feat_out = thin_att + thick_att
        return feat_out







if __name__ == '__main__':
    feat = torch.randn((2,32,48,48)).cuda()
    prob = torch.randn((2,1,48,48)).cuda()

    model = Refine_block(32)
    model = model.cuda()

    feat_out = model(feat, prob)

    print('ok')
