"""
This part is the available loss function
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).cuda()
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)  # Not include softmax
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
        # self.nll_loss = nn.CrossEntropyLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)  



class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets): # 包含了log_softmax函数，调用时网络输出层不需要加log_softmax
        return self.nll_loss((1 - F.softmax(inputs,1)) ** self.gamma * F.log_softmax(inputs,1), targets)



class DICELoss(nn.Module):
    def __init__(self, weight=None, dimention=2, epsilon = 1e-5):
        super(DICELoss, self).__init__()
        self.weight = weight
        self.dimention = dimention
        self.epsilon = epsilon


    def forward(self, input, target):

        input = torch.sigmoid(input)
        input = input.view(-1)
        target = target.view(-1)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = (input * target).sum()
        if self.weight is not None:
            intersect = self.weight * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        if self.dimention == 2:
            denominator = (input * input).sum() + (target * target).sum()
        else:
            denominator = (input + target).sum()

        return 1. - 2 * (intersect / denominator.clamp(min=self.epsilon))


