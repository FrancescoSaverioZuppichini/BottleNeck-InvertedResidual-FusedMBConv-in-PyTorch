# Residual, BottleNeck, Inverted Residual, Linear BottleNeck, MBConv Explained 
## What the hell are those + implementation in PyTorch

Keeping track of names in modern Deep Learning is hard. Today we'll see different blocks used in modern CNN architecture such as ResNet, MobileNet, EfficientNet, and their implementation in PyTorch! 


**All these blocks have been implemented in my library [glasses](https://github.com/FrancescoSaverioZuppichini/glasses)**

Before we do anything, let's create a general conv - norm - act layer


```python
from functools import partial
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)
```


```python
import torch

x = torch.randn((1, 32, 56, 56))

Conv1X1BnReLU(32, 64)(x).shape
```




    torch.Size([1, 64, 56, 56])



## Residual Connections

Residual connections were used in ResNet proposed in [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) and let's also cite Schmidhuber lab's work on [*Highway networks*](https://arxiv.org/abs/1505.00387). The idea is to add your input to your output, `output = layer(input) + input`. The following image may help you visualize it. But, I mean it is just a `+` operator. The residual operation improves the ability of a gradient to propagate across multiplier layers permitting to effectively train networks with more than a hundred layers.

![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch/4c7cfa65c71641d0b86768ee3722d85634e05b5e/images/Residual.svg)

In PyTorch, we can easily create a `ResidualAdd` Layer


```python
from torch import nn
from torch import Tensor

class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x += res
        return x

    
ResidualAdd(
    nn.Conv2d(32, 32, kernel_size=1)
)(x).shape
```




    torch.Size([1, 32, 56, 56])



### Shortcut
Sometimes your residual hasn't the same output's dimension, so we cannot add them. We can project the input using a conv in the shortcut (the black arrow with the +) to match your output's feature


```python
from typing import Optional

class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x

ResidualAdd(
    nn.Conv2d(32, 64, kernel_size=1),
    shortcut=nn.Conv2d(32, 64, kernel_size=1)
)(x).shape
```




    torch.Size([1, 64, 56, 56])



## BottleNeck Blocks
Bottlenecks blocks were also introduced in [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385). A BottleNeck block takes an input of size `BxCxHxW`, it first reduces it to `BxC/rxHxW` using an inexpensive `1x1 conv`, then applies a `3x3 conv` and finally remaps the output to the same feature dimension as the input, `BxCxHxW` using again a `1x1 conv`. This is faster than using three `3x3 convs`. Since the input is reduced first, this is why we called it "BottleNeck". The following figure visualizes the block, we used `r=4` as in the original implementation 

![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch/4c7cfa65c71641d0b86768ee3722d85634e05b5e/images/BottleNeck.svg)

The first two convs are followed by batchnorm and a non-linear activation, while the last non-linearity is applied after the addition. In PyTorch


```python
from torch import nn

class BottleNeck(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, reduction: int = 4):
        reduced_features = out_features // reduction
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        Conv1X1BnReLU(in_features, reduced_features),
                        # narrow -> narrow
                        Conv3X3BnReLU(reduced_features, reduced_features),
                        # narrow -> wide
                        Conv1X1BnReLU(reduced_features, out_features, act=nn.Identity),
                    ),
                    shortcut=Conv1X1BnReLU(in_features, out_features)
                    if in_features != out_features
                    else None,
                ),
                nn.ReLU(),
            )
        )
        
BottleNeck(32, 64)(x).shape
```




    torch.Size([1, 64, 56, 56])



Notice that we apply `shortcut` only if the input and output features are different.

In practice a `stride=2` is used in the middle convolution when we wish to reduce the spatial dimension. 

## Linear BottleNecks

Linear BottleNecks were introduced in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). A Linear BottleNeck Block is a BottleNeck Block without the last activation. In the paper, section 3.2 they go into details about why having non-linearity before the output hurt performance. In a nutshell, the non-linearity function, line ReLU that sets everything < 0 to 0, destroys information. They have empirically shown that this is true when the input's channels are less than the output's. So, remove the `nn.ReLU` in the BottleNeck and you have it.

## Inverted Residual

Inverted Residuals were introduced, again, in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). Inverted Residual blocks are inverted BottleNeck layers. They expand features with the first conv instead of reducing them. The following image should make this clear


![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch/4c7cfa65c71641d0b86768ee3722d85634e05b5e/images/InvertedResidual.svg)

So we go from `BxCxHxW` to ->`BxCexHxW` -> `BxCexHxW` -> `BxCxHxW`, where `e` is the *expansion ratio* and it is set to `4`. Instead of going wide -> narrow -> wide as in normal bottleneck block, they do the opposite, narrow -> wide -> narrow. In PyTorch this is trivial


```python
class InvertedResidual(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # narrow -> wide
                        Conv1X1BnReLU(in_features, expanded_features),
                        # wide -> wide
                        Conv3X3BnReLU(expanded_features, expanded_features),
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                    shortcut=Conv1X1BnReLU(in_features, out_features)
                    if in_features != out_features
                    else None,
                ),
                nn.ReLU(),
            )
        )
        
InvertedResidual(32, 64)(x).shape
```




    torch.Size([1, 64, 56, 56])



In `MobileNet`residual connections are only applied when the input and output features match, don't ask me why, if you know it please comment :) So you should do something like


```python
class MobileNetLikeBlock(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        # use ResidualAdd if features match, otherwise a normal Sequential
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        # narrow -> wide
                        Conv1X1BnReLU(in_features, expanded_features),
                        # wide -> wide
                        Conv3X3BnReLU(expanded_features, expanded_features),
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )
        
MobileNetLikeBlock(32, 64)(x).shape
MobileNetLikeBlock(32, 32)(x).shape
```




    torch.Size([1, 32, 56, 56])



## MBConv

So after MobileNetV2, its building blocks were referred as `MBConv`. A `MBConv` is a Inverted Linear BottleNeck layer with Depth-Wise Separable Convolution. 

### Depth-Wise Separable Convolution
Depth-Wise Separable Convolutions adopt a trick to splint a normal 3x3 conv in two convs to reduce the number of parameters. The first one applies a single 3x3 filter to each input's channels, the other applies a 1x1 filter to all the channels. If you do your match, this is the same thing as doing a normal 3x3 conv but you save parameters. 

This is also kind of stupid because it is way slower than a normal 3x3 on the current hardware we have. 

The following picture shows the idea

![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch/4c7cfa65c71641d0b86768ee3722d85634e05b5e/images/DepthwiseSeparableConv.svg)

The different colours in the channels represent one individual filter applied per channel

In PyTorch:


```python
class DepthWiseSeparableConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(in_features, in_features, kernel_size=3, groups=in_features),
            nn.Conv2d(in_features, out_features, kernel_size=1)
        )
        
DepthWiseSeparableConv(32, 64)(x).shape
```




    torch.Size([1, 64, 54, 54])



The first convolution is usually called `depth` while the second `point`. Let's count the parameters


```python
sum(p.numel() for p in DepthWiseSeparableConv(32, 64).parameters() if p.requires_grad)
```




    2432



Let's see a normal Conv2d


```python
sum(p.numel() for p in nn.Conv2d(32, 64, kernel_size=3).parameters() if p.requires_grad)
```




    18496



That's a big difference

### Getting the MBConv
So, let's create a full MBConv. There are a couple of MBConv's important details, normalization is applied to both depth and point convolution and non-linearity only in the depth convolution (remember linear bottlenecks). The applied ReLU6 non-linearity. Putting everything together


```python
class MBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        # narrow -> wide
                        Conv1X1BnReLU(in_features, 
                                      expanded_features,
                                      act=nn.ReLU6
                                     ),
                        # wide -> wide
                        Conv3X3BnReLU(expanded_features, 
                                      expanded_features, 
                                      groups=expanded_features,
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )
        
MBConv(32, 64)(x).shape
```




    torch.Size([1, 64, 56, 56])



A slighly modified version of this block, with [Squeeze and Excitation](https://arxiv.org/abs/1709.01507) is used in [EfficientNet](https://arxiv.org/abs/1905.11946).

## Fused Inverted Residual (Fused MBConv)

Fused Inverted Residuals were introduced in [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) to make MBConv faster. So basically, since Depthwise convolutions are slow, they fused the first and second conv in a single 3x3 conv (section 3.2). 

![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch/4c7cfa65c71641d0b86768ee3722d85634e05b5e/images/FusedInvertedResidual.svg)


```python
class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        Conv3X3BnReLU(in_features, 
                                      expanded_features, 
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )
        
MBConv(32, 64)(x).shape
```




    torch.Size([1, 64, 56, 56])



## Conclusion

Now you should know the difference between all these blocks and the reasoning behind them! I highly reccomand reading the paper realted to them, you can't go wrong with that. 

For a more detailed guide to ResNet, check out [Residual Networks: Implementing ResNet in Pytorch](https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278)
