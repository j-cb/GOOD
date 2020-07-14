import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import paths_config
import tiny_utils.tinyimages_80mn_loader
import noisefunctions
from auto_augment import AutoAugment
import numpy as np
import warnings

class Noise_Dataset(torch.utils.data.dataset.Dataset):
    """A dataset that is built from a ground dataset and a noise function, returning noisy images of the same shape as the ground data.
       noise_fn should be a function accepting a ground data sample with its label and returning the noisy sample and label with the same shape as the input
    """
    
    def __init__(self, ground_ds, noise_fn, label_fn=None, transform=None):
        self.ground_ds = ground_ds
        self.noise_fn = noise_fn
        self.label_fn = label_fn
        self.transform = transform
        #self.__name__ = noise.__name__ + '_on_' + ground_ds.__name__

    def __getitem__(self, index):
        noisy = self.noise_fn(self.ground_ds[index])
        inp = noisy[0]
        if self.transform is not None:
            inp = transforms.ToPILImage()(inp)
            inp = self.transform(inp)
            
        if self.label_fn == None:
            lbl = noisy[1]
        else:
            lbl = self.label_fn(noisy[1])
        return inp, lbl

    def __len__(self):
        return len(self.ground_ds)
    
    _repr_indent = 4
    def __repr__(self):
        head = "Dataset " + self.__class__.__name__ + '_' + self.noise_fn.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


def getloader_MNIST(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif list(augmentation) == ['crop']:
        transform = transforms.Compose([
            transforms.RandomCrop(28, augmentation['crop']),
            transforms.ToTensor(),
        ])
    else:
        raise KeyError(f'Only crop augmentation supported for MNIST. Got {augmentation}.')
    dset = datasets.MNIST(paths_config.location_dict['MNIST'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_FashionMNIST(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif list(augmentation) == ['crop']:
        transform = transforms.Compose([
            transforms.RandomCrop(28, augmentation['crop']),
            transforms.ToTensor(),
        ])
    else:
        raise KeyError(f'Only crop augmentation supported. Got {augmentation}.')
        
    dset = datasets.FashionMNIST(paths_config.location_dict['FashionMNIST'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_EMNIST_Letters(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            lambda x: np.array(x).T,
            transforms.ToTensor(),
        ])
    elif list(augmentation) == ['crop']:
        transform = transforms.Compose([
            lambda x: np.array(x).T,
            transforms.RandomCrop(28, augmentation['crop']),
            transforms.ToTensor(),
        ])
    else:
        raise KeyError(f'Only crop augmentation supported. Got {augmentation}.')
        
    dset = datasets.EMNIST(paths_config.location_dict['EMNIST'], split='letters', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_Omniglot(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
            lambda x: 1-x,
        ])
    elif list(augmentation) == ['crop']:
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.RandomCrop(28, augmentation['crop']),
            transforms.ToTensor(),
            lambda x: 1-x,
        ])
    else:
        raise KeyError(f'Only crop augmentation supported. Got {augmentation}.')
        
    dset = datasets.omniglot.Omniglot(paths_config.location_dict['Omniglot'], download=True, background=train, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_NotMNIST(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x[0].view(1, 28, 28) ,
        ])
    elif list(augmentation) == ['crop']:
        if augmentation['crop'] == 0:
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: x[0].view(1, 28, 28) ,
            ])
        else:
            raise ValueError(f'Crop not supported for NotMNIST')
    else:
        raise KeyError(f'Augmentation not supported. Got {augmentation}.')
        
    if train:
        raise ValueError(f'ImageNet- only existes for the validation data of ImgeNet exists. train is set to {train}, which is not allowed.')
        
    dset = datasets.ImageFolder('../datasets/notMNIST_small', transform=transform) # deleted corrupted files ['A'] = {'RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png'}, ['F'] = {'Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png'}
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader
    

def getloader_TINY(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
        
    tiny_dset = tiny_utils.tinyimages_80mn_loader.TinyImages(transform=transform, exclude_cifar=['H'] + train*['CEDA11'])
    if train:
        dset = torch.utils.data.Subset(tiny_dset, range(100000, 50000000))
        dset.__repr__ = tiny_dset.__repr__
    else:
        dset = torch.utils.data.Subset(tiny_dset, range(40000))
        dset.__repr__ = tiny_dset.__repr__
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader
    
def getloader_CIFAR10(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )

    dset = datasets.CIFAR10(paths_config.location_dict['CIFAR10'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader
    
def getloader_CIFAR100(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )

    dset = datasets.CIFAR100(paths_config.location_dict['CIFAR100'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_SVHN(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        if augmentation.get('HFlip'):
            warnings.warn(f'Random horizontal flip augmentation selected for SVHN, which usually is not done. Augementations: {augmentation}')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if train:
        split = 'train'
    else:
        split = 'test'
    dset = datasets.SVHN(paths_config.location_dict['SVHN'], split=split, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noise_fn):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if augmentation.get('28g'):
        ground_ds = datasets.MNIST(paths_config.location_dict['MNIST'], train=train, download=True, transform=transform)
    else:
        ground_ds = datasets.CIFAR10(paths_config.location_dict['CIFAR10'], train=train, download=True, transform=transform)
    
    dset = Noise_Dataset(ground_ds, noise_fn, label_fn=lambda x: 0, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_Uniform(train, batch_size, augmentation, dataloader_kwargs):
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.noise_uniform)

def getloader_Smooth(train, batch_size, augmentation, dataloader_kwargs):
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.noise_low_freq)

def getloader_LSUN_CR(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Resize(size=(28, 28)),
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                [transforms.Resize(size=(32, 32)),]
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if train:
        raise ValueError(f'Only the validation split of LSUN Classroom is available. train is set to {train}, which is not allowed.')
    else:
        classes = ['classroom_val']
    dset = datasets.LSUN(paths_config.location_dict['LSUN'], classes=classes, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_ImageNetM(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if train:
        raise ValueError(f'ImageNet- only existes for the validation data of ImgeNet exists. train is set to {train}, which is not allowed.')
    dset = datasets.ImageFolder(paths_config.location_dict['ImageNet-'], transform=transform)
    dset.__repr__ = lambda: "Dataset ImageNet-"
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

    
    
datasets_dict = {'MNIST':             getloader_MNIST,
                 'FashionMNIST':      getloader_FashionMNIST,
                 'EMNIST_Letters':    getloader_EMNIST_Letters,
                 'Omniglot':          getloader_Omniglot,
                 'NotMNIST':          getloader_NotMNIST,
                 'TINY':              getloader_TINY,
                 'CIFAR10':           getloader_CIFAR10,
                 'CIFAR100':          getloader_CIFAR100,
                 'SVHN':              getloader_SVHN,
                 'Uniform':           getloader_Uniform,
                 'Smooth':            getloader_Smooth,
                 'LSUN_CR':           getloader_LSUN_CR,
                 'ImageNet-':         getloader_ImageNetM
                 }

val_loader_out_dicts = dict([])
val_loader_out_dicts['MNIST'] = {
        'FashionMNIST': dict([]),
        'EMNIST_Letters': dict([]),
        'CIFAR10': {'28g': True},
        'Uniform': {'28g': True},
        'TINY': {'28g': True},
        'Omniglot': dict([]),
        'NotMNIST': dict([]),
}
val_loader_out_dicts['CIFAR10'] = {
        'CIFAR100': dict([]),
        'SVHN': dict([]),
        'LSUN_CR': dict([]),
        'Uniform':  dict([]),
        'TINY':  dict([]),
        'ImageNet-':  dict([]),
        'Smooth':  dict([]),
}
val_loader_out_dicts['SVHN'] = {
        'CIFAR100': dict([]),
        'CIFAR10': dict([]),
        'LSUN_CR': dict([]),
        'Uniform':  dict([]),
        'TINY':  dict([]),
        'ImageNet-':  dict([]),
        'Smooth':  dict([]),
}

test_loader_out_dicts = dict([])
test_loader_out_dicts['MNIST'] = {
        'FashionMNIST': dict([]),
        'EMNIST_Letters': dict([]),
        'CIFAR10': {'28g': True},
        'Uniform': {'28g': True},
        'TINY': {'28g': True},
        'Omniglot': dict([]),
        'NotMNIST': dict([]),
}
test_loader_out_dicts['CIFAR10'] = {
        'CIFAR100': dict([]),
        'SVHN': dict([]),
        'LSUN_CR': dict([]),
        'Uniform':  dict([]),
        'TINY':  dict([]),
        'ImageNet-':  dict([]),
        'Smooth':  dict([]),
}
test_loader_out_dicts['SVHN'] = {
        'CIFAR100': dict([]),
        'CIFAR10': dict([]),
        'LSUN_CR': dict([]),
        'Uniform':  dict([]),
        'TINY':  dict([]),
        'ImageNet-':  dict([]),
        'Smooth':  dict([]),
}

def get_val_out_loaders(dset_in_name, batch_size, dataloader_kwargs):
    return [datasets_dict[name](train=False, batch_size=batch_size, augmentation=augm, dataloader_kwargs=dataloader_kwargs) for name, augm in val_loader_out_dicts[dset_in_name].items()]
def get_test_out_loaders(dset_in_name, batch_size, dataloader_kwargs):
    return [datasets_dict[name](train=False, batch_size=batch_size, augmentation=augm, dataloader_kwargs=dataloader_kwargs) for name, augm in test_loader_out_dicts[dset_in_name].items()]
