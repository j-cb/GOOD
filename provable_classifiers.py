import torch
import torch.nn as nn
import copy
from torch.nn import functional as F
import modules_ibp
    
class CNN_IBP(nn.Module):
    def __init__(self, dset_in_name='MNIST', size='L'):
        super().__init__()
        if dset_in_name == 'MNIST':
            self.color_channels = 1
            self.hw = 28
            self.num_classes = 10
        elif dset_in_name == 'CIFAR10' or dset_in_name == 'SVHN':
            self.color_channels = 3
            self.hw = 32
            self.num_classes = 10
        else:
            raise ValueError(f'str(dset_in_name) dataset not supported.')
        self.size = size
        
        if size == 'L':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 64, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(64, 64, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(64, 128, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(128*(self.hw//2)**2, 512)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = modules_ibp.LinearI(512, self.num_classes)

            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                          )

            self.__name__ = 'CNN_L_' + dset_in_name

        elif size == 'XL':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(128, 256, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(256*(self.hw//2)**2, 512)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = modules_ibp.LinearI(512, 512)
            self.A7 = modules_ibp.ReLUI()
            self.L8 = modules_ibp.LinearI(512, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                           self.A7,
                           self.L8,
                          )

            self.__name__ = 'CNN_XL_' + dset_in_name
            
        else:
            raise ValueError(str(size) + 'is not a valid size.')
        
    def forward(self, x):
        x = x.type(torch.get_default_dtype())
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def ibp_forward(self, l, u):
        l = l.type(torch.get_default_dtype())
        u = u.type(torch.get_default_dtype())
        for layer in self.layers:
            l, u = layer.ibp_forward(l, u)
        return l, u
    
    def ibp_elision_forward(self, l, u, num_classes=10):
        l = l.type(torch.get_default_dtype())
        u = u.type(torch.get_default_dtype())
        for layer in self.layers[:-1]:
            l, u = layer.ibp_forward(l, u)
        
        layer = self.layers[-1]
        assert isinstance(layer, modules_ibp.LinearI)
        W = layer.weight
        Wd = W.unsqueeze(dim=1).expand((num_classes,num_classes,-1)) - W.unsqueeze(dim=0).expand((num_classes,num_classes,-1))
        bd = layer.bias.unsqueeze(dim=1).expand((num_classes,num_classes)) -  layer.bias.unsqueeze(dim=0).expand((num_classes,num_classes))
        ud = torch.einsum('abc,nc->nab', Wd.clamp(min=0), u) + torch.einsum('abc,nc->nab', Wd.clamp(max=0), l)
        ud += bd.unsqueeze(0)
        
        l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() + layer.bias[:,None]).t()
        u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() + layer.bias[:,None]).t()
        l,u = l_, u_
        return l, u, ud