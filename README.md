# OCE-Net - PyTorch
Official pytorch codes and models for paper:

## [Orientation and Context Entangled Network for Retinal Vessel Segmentation](https://www.sciencedirect.com/science/article/pii/S0957417422024629) 
Expert Systems with Applications (ESWA)  (Top Journal, JCR Q1, IF=8.665)\
**Xinxu Wei**, Kaifu Yang, Danilo Bzdok, and Yongjie Li




# Datasets
All the datasets involved in the paper have been provided. \
You can get access to the datasets by clicking the links below!

[Retinal Vessel Datasets](https://pan.baidu.com/s/1yQBjDa4omzSwTY6RUnkeGg) (password: abcd)





# Training
You can train the model by runing **train.py**, the modelcan be trained and the checkpoint can be saved in the folder **experiments**.



# Testing
You can test the pre-trained models on the provided datasets or your own data by runing **test.py**. Please change the dir path of data before the testing.



# Metrics
You can run the **calculate_metrics.py** to get the metrics of segmentation (Acc, F1 Score, ect.).



# Model architecture
![Model](/pic/model.png)



# Dynamic Complex Orientation Aware Convolution (DCOA Conv)
![DCOA_conv](/pic/DCOA.png)

```python
def getGaborFilterBank(h, w):
    nScale = 1
    M = 8
    Kmax = math.pi / 2
    f = math.sqrt(2)
    k = Kmax / f ** (nScale - 1)
    sigma = math.pi
    sqsigma = sigma ** 2
    postmean = math.exp(-sqsigma / 2)
    if h != 1:
        gfilter_real = torch.zeros(M, h, w)
        for i in range(M):
            theta = i / M * math.pi
            kcos = k * math.cos(theta)
            ksin = k * math.sin(theta)
            for y in range(h):
                for x in range(w):
                    y1 = y + 1 - ((h + 1) / 2)
                    x1 = x + 1 - ((w + 1) / 2)
                    tmp1 = math.exp(-(k * k * (x1 * x1 + y1 * y1) / (2 * sqsigma)))
                    tmp2 = math.cos(kcos * x1 + ksin * y1) - postmean  # For real part
                    gfilter_real[i][y][x] = k * k * tmp1 * tmp2 / sqsigma
            xymax = torch.max(gfilter_real[i])
            xymin = torch.min(gfilter_real[i])
            gfilter_real[i] = (gfilter_real[i] - xymin) / (xymax - xymin)
    else:
        gfilter_real = torch.ones(M, h, w)
    Gfilter_real = torch.zeros(M, 1, h, w)
    Gfilter_real = torch.unsqueeze(gfilter_real, 1)
    return Gfilter_real


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=8, temperature=34, init_weight=True, is_cuda = True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.is_cuda = is_cuda
        self.attention = attention2d(in_planes, ratio, K, temperature)
        # self.gabor_bank = getGaborFilterBank(*(3,3))
        self.bn = nn.BatchNorm2d(1)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        # self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()


    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight
        weight_gb = getGaborFilterBank(*(3,3))   # K 1 3 3
        weight_gb = weight_gb.cuda()
        weight_gb = self.bn(weight_gb)
        weight_gb = weight_gb.unsqueeze(1)   # K 1 1 3 3
        weight_all = weight * weight_gb
        weight_all = weight_all.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight_all).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
```







# Unbalanced Attention Refining Module (UARM)
![UARM](/pic/UARM.png)

```python
class sign_(nn.Module):

    def __init__(self, *kargs, **kwargs):
        super(sign_, self).__init__(*kargs, **kwargs)
        self.r = sign_f.apply  

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
        output[inputs < 0.4] = 1
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
```







# Experiments
## Qualitative
![ex1](/pic/visual_ex_1.png)

## Quantitative
![ex2](/pic/visual_ex_2.png)

![ex3](/pic/visual_ex_abla.png)



## Quantitative
![ex4](/pic/ex_drive.png)

![ex5](/pic/ex_chase.png)

![ex6](/pic/ex_stare.png)




# Requirements

````
torch==1.5+cuda101
torchvision==0.6.0
tensorboardX==2.4.1
numpy==1.19.5
opencv-python-headless==4.5.5.92
tqdm==4.62.2
scikit-learn==0.21.3
joblib==1.1.0
````


# Citation

```
@article{wei2022orientation,
  title={Orientation and context entangled network for retinal vessel segmentation},
  author={Wei, Xinxu and Yang, Kaifu and Bzdok, Danilo and Li, Yongjie},
  journal={Expert Systems with Applications},
  pages={119443},
  year={2022},
  publisher={Elsevier}
}
```
