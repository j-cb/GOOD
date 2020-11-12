import torch
import torch.nn as nn
import numpy as np

class Add_ParamI(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = x + self.bias
        return out
        
    def ibp_forward(self, l, u):
        l_ = l + self.bias
        u_ = u + self.bias
        return l_, u_
                
class Scale_By_ParamI(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = x * self.scalar
        return out
    
    def ibp_forward(self, l, u):
        if self.scalar >= 0:
            l_ = l * self.scalar
            u_ = u * self.scalar
        else:
            u_ = l * self.scalar
            l_ = u * self.scalar
        return l_, u_

class FlattenI(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    def ibp_forward(self, l, u):
        l_ = self.forward(l)
        u_ = self.forward(u)
        return l_, u_
    
class Conv2dI(nn.Conv2d):    
    def ibp_forward(self, l, u):
        l_ = (nn.functional.conv2d(l, self.weight.clamp(min=0), bias=None, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups) +
              nn.functional.conv2d(u, self.weight.clamp(max=0), bias=None, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups)
             )
        u_ = (nn.functional.conv2d(u, self.weight.clamp(min=0), bias=None, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups) +
              nn.functional.conv2d(l, self.weight.clamp(max=0), bias=None, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups)
             )
        if self.bias is not None:
            l_ += self.bias[None,:,None,None]
            u_ += self.bias[None,:,None,None]
        return l_, u_

class ReLUI(nn.ReLU):
    def ibp_forward(self, l, u):
        l_ = l.clamp(min=0)
        u_ = u.clamp(min=0)
        return l_, u_
    
class AvgPool2dI(nn.AvgPool2d):
    def ibp_forward(self, l, u):
        return self.forward(l), self.forward(u)
    
class AdaptiveAvgPool2dI(nn.AdaptiveAvgPool2d):
    def ibp_forward(self, l, u):
        return self.forward(l), self.forward(u)
    

class LinearI(nn.Linear):
    def ibp_forward(self, l, u):
        l_ = (self.weight.clamp(min=0) @ l.t() + self.weight.clamp(max=0) @ u.t() 
              + self.bias[:,None]).t()
        u_ = (self.weight.clamp(min=0) @ u.t() + self.weight.clamp(max=0) @ l.t() 
              + self.bias[:,None]).t()
        return l_, u_
    
def conv3x3I(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2dI(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
class SequentialI(nn.Sequential):
    def ibp_forward(self, l, u):
        for module in self:
            l,u = module.ibp_forward(l, u)
        return l,u