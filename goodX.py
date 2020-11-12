import argparse
import sys
import os
import datetime
from types import MethodType

import torch
import torch.utils.data
from torch import nn, optim

import dataloading
import provable_classifiers
#import provable_classifiers_mod
import good_training
import evaluation
import tiny_utils.tinyimages_80mn_loader
import dataloading
import paths_config
from auto_augment import AutoAugment

#from continuepath_config import continue_dict_path, continue_dict_name


#Parameters that can be set when calling python ...X.py --parameter value
parser = argparse.ArgumentParser(description='GOOD training')
#for all modes
parser.add_argument('--mode', type=str, default='train',
                    help='set to eval for evaluation')
parser.add_argument('--dset_in_name', type=str, default='MNIST',
                    help='in dataset.')
parser.add_argument('--gpu',  type=int, default=0,
                    help='Choose GPU, -1 for CPU.')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')

#for mode=='train'
parser.add_argument('--arch', type=str, default='L',
                    help='model architecture.')
parser.add_argument('--expfolder', type=str, default=None,
                    help='folder under which the models will be saved')
parser.add_argument('--tb_folder', type=str, default='default',
                    help='folder in which the tb logs will be saved')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--start_lr', type=float, default=None,
                    help='start lr; will be divided by batch size since we use sum reduction. (default: 0.005)')
parser.add_argument('--epochs', type=int, default=None,
                    help='number of epochs to train')
parser.add_argument('--method', type=str, default='plain',
                    help='mode options are plain, lc_ibp')
parser.add_argument('--acet', type=str, default='',
                    help='Use ACET in addition to the rest. lc or ce')#acet is added on top, so for standard acet use --method 'plain'.
parser.add_argument('--optimizer', type=str, default=None,
                    help='SGDM or ADAM.')

#training augmentation
parser.add_argument("--traincrop", type=int, default=4, 
                    help="randomly crop.")
parser.add_argument("--autoaugment", default=False, action="store_true" , 
                    help="Use autoaugment.")
parser.add_argument('--pretrained', type=str, default=None,
                    help='Initialize with a specified model.')

#for certain out-losses
parser.add_argument('--kappa', type=float, default=None,
                    help='final factor of out loss')
parser.add_argument('--eps', type=float, default=None,
                    help='final input bound, used for all inter-epoch evaluations')
parser.add_argument('--dset_out_name', type=str, default='TINY',
                    help='out dataset.')
parser.add_argument('--good_quantile', type=float, default=1.0,
                    help='For quantile_square_logit_spread loss.')

#for mode=='eval'
parser.add_argument("--n_pgd_eval", type=int, default=0, 
                    help="steps of eval pgd.")
parser.add_argument('--eval_n_samples', type=int, default=30000,
                    help='max number of test samples per dataset for eval')
parser.add_argument("--save_eval_plots", default=False, action="store_true" , 
                    help="Save histograms etc. during evaluation.")

args = parser.parse_args()
print(f'Parsed arguments: {args.__dict__}')

if args.expfolder:
    expfolder = 'experiments/' + args.expfolder
else:
    expfolder = 'experiments/' + 'GOOD_' + args.dset_in_name
    
os.makedirs(expfolder, exist_ok=True)

torch.manual_seed(args.seed)
if args.gpu > -1:
    device = torch.device('cuda:'+str(args.gpu))
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')
torch.set_printoptions(edgeitems=10)

dataloader_kwargs = {'num_workers': 1, 'pin_memory': True}

#Load datasets and define dataloaders TODO: only load those that are (will be) used.

if args.mode == 'train':
    if args.dset_in_name == 'MNIST':
        augmentation_train_in = {'crop': args.traincrop}
        augmentation_train_out = {'28g': True, 'crop': args.traincrop}
        augmentation_val_in = dict([])
    if args.dset_in_name == 'CIFAR10':
        augmentation_train_in = {'crop': args.traincrop, 'HFlip': True, 'autoaugment': args.autoaugment}
        augmentation_train_out = {'crop': args.traincrop, 'HFlip': True, 'autoaugment': args.autoaugment}
        augmentation_val_in = dict([])
    if args.dset_in_name == 'SVHN':
        augmentation_train_in = {'crop': args.traincrop, 'HFlip': False, 'autoaugment': args.autoaugment}
        augmentation_train_out = {'crop': args.traincrop, 'HFlip': False, 'autoaugment': args.autoaugment}
        augmentation_val_in = dict([])
    
    train_model = provable_classifiers.CNN_IBP(dset_in_name=args.dset_in_name, size=args.arch).to(device)
    
    train_loader_in = dataloading.datasets_dict[args.dset_in_name](train=True, batch_size=args.batch_size, augmentation=augmentation_train_in, dataloader_kwargs=dataloader_kwargs)
    train_loader_out = dataloading.datasets_dict[args.dset_out_name](train=True, batch_size=args.batch_size, augmentation=augmentation_train_out, dataloader_kwargs=dataloader_kwargs) #train_loader_out should usually be at least as long as train_loader_in
    val_loader_in = dataloading.datasets_dict[args.dset_in_name](train=False, batch_size=args.batch_size, augmentation=augmentation_val_in, dataloader_kwargs=dataloader_kwargs)
    val_loader_out_list = dataloading.get_val_out_loaders(args.dset_in_name, args.batch_size, dataloader_kwargs)
    if args.pretrained:
        train_model.load_state_dict(torch.load(paths_config.pretrained_dict[args.pretrained], map_location='cpu'))
        evaluation.evaluate_ibp_lc(train_model, val_loader_in, val_loader_out_list, eps=0.01, conf_th=0.3, device=device, n_pgd=0, n_samples=1000)
    model_folder = good_training.train_good(
        model=train_model,
        train_loader_in=train_loader_in,
        train_loader_out=train_loader_out,
        val_loader_in=val_loader_in,
        val_loader_out_list=val_loader_out_list,
        device=device,
        expfolder=expfolder,
        args=args
    )
    
    print('Trained model saved under {}'.format(model_folder))

elif args.mode == 'eval':
    import eval_paths
    val_th = 0.3
    model_paths = eval_paths.eval_paths[args.dset_in_name]
    test_loader_in = dataloading.datasets_dict[args.dset_in_name](train=False, batch_size=args.batch_size, augmentation=dict([]), dataloader_kwargs=dataloader_kwargs)
    test_loader_out_list = dataloading.get_test_out_loaders(args.dset_in_name, args.batch_size, dataloader_kwargs)
    for path in model_paths:
        try:
            arch = eval_paths.get_arch(path)
        except ValueError as e:
            print(e)
            continue
        model = provable_classifiers.CNN_IBP(dset_in_name=args.dset_in_name, size=arch).to(device)
        print('\n***********************************************************Evaluating ' + path + ' ***********************************************************')
        model.load_state_dict(torch.load(path, map_location='cpu'))
        short_name = eval_paths.get_shortname(path)
        print(f'The short name of this model is {short_name}')
        eval_dict = evaluation.evaluate_ibp_lc(model, test_loader_in, test_loader_out_list, short_name=short_name, eps=args.eps, conf_th=val_th,  device=device, n_pgd=args.n_pgd_eval, model_path=path, n_samples=args.eval_n_samples, do_accuracy_above=True, save_plots=args.save_eval_plots)

else:
    raise ValueError(f'{args.train} must be train or eval')