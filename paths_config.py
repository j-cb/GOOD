location_dict = dict([])
location_dict['MNIST'] = '../data'
location_dict['FashionMNIST'] = '../data'
location_dict['EMNIST'] = '../data'
location_dict['Omniglot'] = '../data'
location_dict['NotMNIST'] = '../datasets/notMNIST_small'
location_dict['CIFAR10'] = '../data'
location_dict['SVHN'] = '../data'
location_dict['CIFAR100'] = '../data'
location_dict['LSUN'] = '../datasets/lsun'
location_dict['ImageNet-'] = '../datasets/imagenet_minus_cifar10/imagenet/val_folder/'

pretrained_dict = dict([])
pretrained_dict['CIFAR10_CEDA'] = 'models/CIFAR10_CEDA_200.pt'
pretrained_dict['SVHN_CEDA'] = 'models/SVHN_CEDA_300.pt'