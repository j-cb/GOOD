import datetime
import sys
import os

import torch
from torch.nn import functional as F
from torch import nn, optim
import torchvision.utils as vutils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import evaluation
import lossfunctions
import attacks
import schedules
import names

import copy
import math

ce_loss = nn.CrossEntropyLoss(reduction='sum')
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train_good(model, train_loader_in, train_loader_out, val_loader_in, val_loader_out_list, device, expfolder, args):
    train_out_name = train_loader_out.dataset.__repr__().split()[1]
    train_in_name = train_loader_in.dataset.__repr__().split()[1]
    print(model.layers)
    starttime = datetime.datetime.utcnow()
    
    
    schedule = schedules.schedule(args)
    
    print(f'schedule: {schedule}')
    model_folder = names.model_folder_name(expfolder, starttime, args, schedule)
    for subfolder in ['state_dicts','sample_images','logs','batch_images']:
        os.makedirs(f'{model_folder}/{subfolder}/', exist_ok=True)
    tb_subfolder = f'tb_logs/{args.tb_folder}/{model_folder}'
    os.makedirs(tb_subfolder, exist_ok=True)
    writer = SummaryWriter(tb_subfolder)
    print(f'model folder: {model_folder}')
    print(f'tb_subfolder: {tb_subfolder}')
    
    trainstart_message = f'Training {model.__name__} for {schedule["epochs"]} epochs of {2*min(len(train_loader_in.dataset), len(train_loader_out.dataset))} samples.'
    print(trainstart_message)
    
    if schedule['optimizer'] == 'SGDM':
        optimizer = optim.SGD(model.parameters(), lr=schedule['start_lr'], weight_decay=0.05, momentum=0.9, nesterov=True)
    elif schedule['optimizer'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=schedule['start_lr'], weight_decay=0.005)
    else: 
        raise ValueError(f'Optimizer {schedule["optimizer"]} not supported. Must be SGDM or ADAM.')
    print(f'Optimizer settings: {optimizer}')
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, schedule['lr_decay_epochs'], gamma=schedule['lr_decay_factor'], last_epoch=-1)

    num_classes = model.num_classes

    for epoch in range(schedule['epochs']):
        #initialize epoch summary
        in_samples_this_epoch = 0
        ce_losses_in_epoch = []
        log_confs_in_epoch = []
        corrects_in_this_epoch = 0
        
        out_samples_this_epoch = 0
        above_quantile_losses_out_epoch, below_quantile_losses_out_epoch, full_good_losses_out_epoch, zero_good_losses_out_epoch, good_losses_out_epoch = [], [], [], [], []
        oe_losses_out_epoch, ceda_losses_out_epoch = [], []
        losses_out_epoch, kappa_losses_out_epoch = [], []
        log_confs_out_epoch, ub_log_confs_out_epoch = [], []
        acet_losses_out_epoch = []
        
        losses_epoch = []
            
        #hyperparameters for this epoch
        if schedule['kappa_epoch_ramp'] == 0:
            kappa_epoch = schedule['kappa'] * (epoch >= schedule['out_start_epoch'])
        else:
            kappa_epoch = schedule['kappa'] * min(max(epoch-schedule['out_start_epoch'], 0)/schedule['kappa_epoch_ramp'], 1)
        if schedule['eps_epoch_ramp'] == 0:
            eps_epoch = schedule['eps'] * (epoch >= schedule['eps_start_epoch'])
        else:
            eps_epoch = schedule['eps'] * min((max(epoch-schedule['eps_start_epoch'], 0)/schedule['eps_epoch_ramp']), 1)
        
        #if acet is turned on, it will be used in addition to the args.method   
        if args.acet:
            if args.acet == 'ce':
                acet_lossfn = lossfunctions.CrossEntropyLossDistr
            elif args.acet == 'lc':
                acet_lossfn = lossfunctions.LogConf
            pgd = attacks.LinfPGDAttack(epsilon=eps_epoch, n=schedule['acet_n'], loss_fn=acet_lossfn, random_start=False, device=device)
        do_acet_epoch = args.acet and kappa_epoch > 0
        
        model.train()
        for batch_number, data in enumerate(zip(train_loader_in, train_loader_out),0):
            img_batch_parts, lbl_batch_parts = [d[0].to(device) for d in data], [d[1].to(device) for d in data]
            img_batch_in = img_batch_parts[0].to(device)
            img_batch_out = img_batch_parts[1].to(device)
            lbl_batch_in = lbl_batch_parts[0].to(device)
            lbl_batch_in_1hot = F.one_hot(lbl_batch_in, num_classes).float()
            lbl_batch_out = 1/num_classes *  torch.ones(lbl_batch_parts[1].size() + (num_classes,), dtype=lbl_batch_parts[1].dtype).to(device) #set uniform label as it represents optimum
                        
            batch_size_in = len(img_batch_in)
            batch_size_out = len(img_batch_out)
            
            in_samples_this_epoch += batch_size_in
            out_samples_this_epoch += batch_size_out
            
            #save example batch
            if epoch == 0 and batch_number == 0:
                vutils.save_image(img_batch_in, model_folder + '/batch_images/in_batch0.png')
                vutils.save_image(img_batch_out, model_folder + '/batch_images/out_batch0.png')
                
            optimizer.zero_grad() #resets the calculated gradients
            
            logit_batch_in = model(img_batch_in)
            
            ce_loss_in = ce_loss(logit_batch_in, lbl_batch_in)
            ce_losses_in_epoch.append(ce_loss_in.detach().cpu().numpy()) #tracking
            p_in = logit_batch_in.softmax(dim=-1) #tracking
            _, predicted_class_in = logit_batch_in.max(dim=-1) #tracking
            corrects_in_this_epoch += predicted_class_in.eq(lbl_batch_in).sum().item() #tracking
    
            do_acet_epoch = args.acet and kappa_epoch > 0
            if do_acet_epoch:
                if eps_epoch > 0:
                    adv_batch_out, _  = pgd.perturbt(img_batch_out, lbl_batch_out, model)
                    model.train() #to make sure it isn't set to eval after the attack
                else:
                    adv_batch_out = img_batch_out
                logit_adv_batch_out = model(adv_batch_out)
                acet_losses_indiv = acet_lossfn(logit_adv_batch_out, lbl_batch_out)
                log_conf_adv_out = logit_adv_batch_out.log_softmax(dim=-1).max(dim=-1)[0]
                acet_loss_out = acet_losses_indiv.sum()
                acet_losses_out_epoch.append(acet_loss_out.detach().cpu().numpy()) #tracking
                
            #calculate losses on the OOD inputs
            if args.method in {'OE', 'CEDA'}:
                logit_batch_out = model(img_batch_out)
                log_conf_out_batch = logit_batch_out.log_softmax(dim=-1).max(dim=-1)[0]
                
                ceda_loss_out = log_conf_out_batch.sum()
                log_pred_out = logit_batch_out.log_softmax(dim=-1)
                oe_loss_out = -(log_pred_out/num_classes).sum()
                ceda_losses_out_epoch.append(ceda_loss_out.detach().cpu().numpy()) #tracking
                oe_losses_out_epoch.append(oe_loss_out.detach().cpu().numpy()) #tracking
                log_confs_out_epoch.append(log_conf_out_batch.detach().cpu().numpy())
                
            if args.method == 'GOOD':
                l_logits_batch_out, u_logits_batch_out, ud_logit_out_batch = model.ibp_elision_forward(img_batch_out - eps_epoch, img_batch_out + eps_epoch, num_classes)
                ub_log_conf_out_batch = ud_logit_out_batch.max(dim=-1)[0].max(dim=-1)[0]
                ub_conf_out_batch = ub_log_conf_out_batch.exp()/num_classes
                logit_batch_out = model(img_batch_out)
                logit_diff = logit_batch_out.max(dim=-1)[0] - logit_batch_out.min(dim=-1)[0] #equals ud_logit_out_batch.max(dim=-1)[0].max(dim=-1)[0] for eps=0, but only needs 1 pass.
                l = math.floor(batch_size_out*args.good_quantile)
                h = batch_size_out - l
                above_quantile_indices = ub_log_conf_out_batch.topk(h, largest=True)[1] #above or exactly at quantile, i.e. 'not below'.
                below_quantile_indices = ub_log_conf_out_batch.topk(l, largest=False)[1] 
                
                above_quantile_loss_out = ((logit_diff[above_quantile_indices])**2/2).log1p().sum()
                below_quantile_loss_out = ((ub_log_conf_out_batch[below_quantile_indices])**2/2).log1p().sum()
                good_loss_out = above_quantile_loss_out + below_quantile_loss_out
                
                #for tracking only
                zero_good_loss_out = (logit_diff**2/2).log1p().sum()
                full_good_loss_out = (ub_log_conf_out_batch**2/2).log1p().sum()
                log_conf_out_batch = logit_batch_out.log_softmax(dim=-1).max(dim=-1)[0]
                ceda_loss_out = log_conf_out_batch.sum()
                log_pred_out = logit_batch_out.log_softmax(dim=-1)
                oe_loss_out = -(log_pred_out/num_classes).sum()
                
                above_quantile_losses_out_epoch.append(above_quantile_loss_out.detach().cpu().numpy())
                below_quantile_losses_out_epoch.append(below_quantile_loss_out.detach().cpu().numpy())
                good_losses_out_epoch.append(good_loss_out.detach().cpu().numpy())
                
                zero_good_losses_out_epoch.append(zero_good_loss_out.detach().cpu().numpy())
                full_good_losses_out_epoch.append(full_good_loss_out.detach().cpu().numpy())
                ceda_losses_out_epoch.append(ceda_loss_out.detach().cpu().numpy())
                oe_losses_out_epoch.append(oe_loss_out.detach().cpu().numpy())
                log_confs_out_epoch.append(log_conf_out_batch.detach().cpu().numpy())
                ub_log_confs_out_epoch.append(ub_log_conf_out_batch.detach().cpu().numpy())
                
                #save example out batch splits
                if epoch % 10 == 0 and batch_number == 0:
                    if len(above_quantile_indices) > 0:
                        vutils.save_image(img_batch_out[above_quantile_indices], model_folder + f'/batch_images/{epoch:3d}batch0_above_quantile.png')            
                    if len(below_quantile_indices) > 0:
                        vutils.save_image(img_batch_out[below_quantile_indices], model_folder + f'/batch_images/{epoch:3d}batch0_below_quantile.png')
            
            if args.method == 'plain' or epoch < schedule['out_start_epoch']:
                loss_batch = ce_loss_in.clone() #clone so adding acet to it cannot change ce_loss_in
                loss_name = 'in_ce'
                losses_out_epoch.append(0)
                kappa_losses_out_epoch.append(0)
            elif args.method == 'OE':
                loss_batch = ce_loss_in + kappa_epoch*oe_loss_out
                loss_name = f'in_ce+{kappa_epoch}*oe_loss_out'
                losses_out_epoch.append(oe_loss_out.detach().cpu().numpy())
                kappa_losses_out_epoch.append(kappa_epoch*oe_loss_out.detach().cpu().numpy())
            elif args.method == 'CEDA':
                loss_batch = ce_loss_in + kappa_epoch*ceda_loss_out
                loss_name = f'in_ce+{kappa_epoch}*ceda_loss_out'
                losses_out_epoch.append(ceda_loss_out.detach().cpu().numpy())
                kappa_losses_out_epoch.append(kappa_epoch*ceda_loss_out.detach().cpu().numpy())
            elif args.method == 'GOOD':
                loss_batch = ce_loss_in + kappa_epoch*good_loss_out
                loss_name = f'in_ce + {kappa_epoch}*(above_quantile_loss_out + eps{eps_epoch}below_quantile_loss_out)'
                losses_out_epoch.append(good_loss_out.detach().cpu().numpy())
                kappa_losses_out_epoch.append(kappa_epoch*good_loss_out.detach().cpu().numpy())
                
            #acet is added on top
            if do_acet_epoch:
                loss_batch += kappa_epoch*acet_loss_out
                loss_name += f'+{kappa_epoch}*out_{eps_epoch}_adv_conf'  
                
            losses_epoch.append(loss_batch.detach().cpu().numpy()) #tracking
            
            loss_batch.backward()# backpropagation of the loss. between here and optimizer.step() there should be no computations; only for saving the gradients it makes sense to have code between the two commands.
            optimizer.step()# updates the parameters of the model

        
        ce_loss_in_epoch = np.sum(ce_losses_in_epoch)/in_samples_this_epoch
        accuracy_epoch = corrects_in_this_epoch/in_samples_this_epoch
        log_conf_in_epoch = np.sum(log_confs_in_epoch)/in_samples_this_epoch
        loss_epoch = np.sum(losses_epoch)/in_samples_this_epoch #per in sample!
        
        loss_out_epoch = np.sum(losses_out_epoch)/out_samples_this_epoch
        kappa_loss_out_epoch = np.sum(kappa_losses_out_epoch)/out_samples_this_epoch   
        
        if args.acet and kappa_epoch > 0:
            acet_loss_out_epoch = np.sum(acet_losses_out_epoch)/out_samples_this_epoch
        
        if args.method in {'OE', 'CEDA'}:
            oe_loss_out_epoch = np.sum(oe_losses_out_epoch)/out_samples_this_epoch
            ceda_loss_out_epoch = np.sum(ceda_losses_out_epoch)/out_samples_this_epoch
            log_conf_out_epoch = np.sum( np.concatenate(log_confs_out_epoch))/out_samples_this_epoch
            median_log_conf_out_epoch = np.median(np.concatenate(log_confs_out_epoch))/out_samples_this_epoch
            
        if args.method == 'GOOD':
            above_quantile_loss_out_epoch = np.sum(above_quantile_losses_out_epoch)/out_samples_this_epoch
            below_quantile_loss_out_epoch = np.sum(below_quantile_losses_out_epoch)/out_samples_this_epoch
            full_good_loss_out_epoch = np.sum(full_good_losses_out_epoch)/out_samples_this_epoch
            zero_good_loss_out_epoch = np.sum(zero_good_losses_out_epoch)/out_samples_this_epoch
            oe_loss_out_epoch = np.sum(oe_losses_out_epoch)/out_samples_this_epoch
            ceda_loss_out_epoch = np.sum(ceda_losses_out_epoch)/out_samples_this_epoch
            log_conf_out_epoch = np.sum(np.concatenate(log_confs_out_epoch))/out_samples_this_epoch
            ub_log_conf_out_epoch = np.sum(np.concatenate(ub_log_confs_out_epoch))/out_samples_this_epoch
            median_log_conf_out_epoch = np.median(np.concatenate(log_confs_out_epoch))/out_samples_this_epoch
            median_ub_log_conf_out_epoch = np.median(np.concatenate(ub_log_confs_out_epoch))/out_samples_this_epoch
        s_0 = f'Epoch {epoch} (lr={get_lr(optimizer)}) complete with mean training loss {loss_epoch} (ce_loss_in: {ce_loss_in_epoch}, loss_out:{loss_out_epoch}, used loss:{loss_name}).'
        s_1 = 'Time since start of training: {0}.\n'.format(datetime.datetime.utcnow() - starttime)
        print(s_0)
        print(s_1)

        writer.add_scalar('TrainIn/loss_total_per_in', loss_epoch, epoch)
        writer.add_scalar('TrainIn/ce_loss_in', ce_loss_in_epoch, epoch)
        writer.add_scalar('TrainIn/accuracy', accuracy_epoch, epoch)
        
        writer.add_scalar('TrainOut/loss_out', loss_out_epoch, epoch)
        writer.add_scalar('TrainOut/kappa_loss_out', kappa_loss_out_epoch, epoch)
        if args.acet and kappa_epoch > 0:
             writer.add_scalar('TrainOut/acet_loss_out', acet_loss_out_epoch, epoch)
        if args.method in {'OE', 'CEDA'}:
            writer.add_scalar('TrainOut/oe_loss_out', oe_loss_out_epoch, epoch)
            writer.add_scalar('TrainOut/ceda_loss_out', ceda_loss_out_epoch, epoch)
            writer.add_scalar('TrainOut/log_conf_out', log_conf_out_epoch, epoch)
            writer.add_scalar('TrainOut/median_log_conf_out', median_log_conf_out_epoch, epoch)            
            writer.add_histogram('Train_log_conf_out', np.concatenate(log_confs_out_epoch), epoch)
        if args.method == 'GOOD':
            writer.add_scalar('TrainOut/above_quantile_loss_out', above_quantile_loss_out_epoch, epoch)
            writer.add_scalar('TrainOut/below_quantile_loss_out', below_quantile_loss_out_epoch, epoch)
            writer.add_scalar('TrainOut/full_good_loss_out', full_good_loss_out_epoch, epoch)
            writer.add_scalar('TrainOut/zero_good_loss_out', zero_good_loss_out_epoch, epoch)
            writer.add_scalar('TrainOut/oe_loss_out', oe_loss_out_epoch, epoch)
            writer.add_scalar('TrainOut/ceda_loss_out', ceda_loss_out_epoch, epoch)
            writer.add_scalar('TrainOut/log_conf_out', log_conf_out_epoch, epoch)
            writer.add_scalar('TrainOut/ub_log_conf_out', ub_log_conf_out_epoch, epoch) 
            writer.add_scalar('TrainOut/median_log_conf_out', median_log_conf_out_epoch, epoch)
            writer.add_scalar('TrainOut/median_ub_log_conf_out', median_ub_log_conf_out_epoch, epoch)
            writer.add_histogram('Train_log_conf_out', np.concatenate(log_confs_out_epoch), epoch)
            writer.add_histogram('Train_ub_log_conf_out', np.concatenate(ub_log_confs_out_epoch), epoch)
        writer.add_scalar('TrainHyPa/eps', eps_epoch, epoch)
        writer.add_scalar('TrainHyPa/kappa', kappa_epoch, epoch)
        writer.add_scalar('TrainHyPa/learning_rate', get_lr(optimizer), epoch)
        
        do_valuation = True #the whole evaluation only takes a few seconds.
        if do_valuation:
            val_th = 0.3 #for evaluating how many samples get conf values > 30%.
            if train_in_name == 'MNIST':
                val_eps = 0.3 #smaller values can be useful if no guarantees are given for this
            if train_in_name == 'SVHN' or train_in_name == 'CIFAR10':
                val_eps = 0.01
                
            eval_result_dict = evaluation.evaluate_ibp_lc(model, val_loader_in, val_loader_out_list, eps=val_eps, conf_th=val_th, device=device, n_pgd=0, n_samples=1000)
            
            in_accuracy, pred_in_confidences, pred_in_mean_confidence, pred_in_above_th, number_of_in_datapoints = eval_result_dict[val_loader_out_list[0].dataset.__repr__().split()[1]][:5]
            writer.add_scalar('Val/in_accuracy', in_accuracy, epoch)
            writer.add_scalar('Val/mean_confidence', pred_in_mean_confidence, epoch)
            writer.add_scalar('Val/confidences_above_{0:.2f}'.format(val_th), pred_in_above_th/number_of_in_datapoints, epoch)
            writer.add_scalar('Val/eps', val_eps, epoch)
            
            writer.add_histogram('Val/pred_in_confidences', pred_in_confidences, epoch)
            
            for val_loader_out in val_loader_out_list:
                out_name = val_loader_out.dataset.__repr__().split()[1]
                in_accuracy, pred_in_confidences, pred_in_mean_confidence, pred_in_above_th, number_of_in_datapoints, pred_out_confidences, pred_out_mean_confidence, pred_out_above_th, number_of_out_datapoints, ub_el_out_confidences, ub_elision_mean_out_confidence, ub_elision_median_out_confidence, ub_elision_out_below_th, auroc_from_predictions, auroc_out_guaranteed_softmax_elision, auroc_from_predictions_conservative, auroc_out_guaranteed_softmax_elision_conservative, pred_adv_out_confidences, adversarial_pred_out_mean_confidence, adversarial_pred_out_median_confidence, adversarial_pred_out_above_th = eval_result_dict[out_name]
            
                writer.add_scalar('Val{0}/mean_confidence'.format(out_name), pred_out_mean_confidence, epoch)
                writer.add_scalar('Val{0}/mean_ub_confidence'.format(out_name), ub_elision_mean_out_confidence, epoch)
                writer.add_scalar('Val{0}/median_ub_confidence'.format(out_name), ub_elision_median_out_confidence, epoch)
                writer.add_scalar('Val{0}/confidences_above_{1:.2f}'.format(out_name, val_th), pred_out_above_th/number_of_out_datapoints, epoch)
                writer.add_scalar('Val{0}/ub_confidences_below_{1:.2f}'.format(out_name, val_th), ub_elision_out_below_th/number_of_out_datapoints, epoch)
                writer.add_scalar('Val{0}/AUC'.format(out_name), auroc_from_predictions, epoch)
                writer.add_scalar('Val{0}/GAUC'.format(out_name), auroc_out_guaranteed_softmax_elision, epoch)
                writer.add_scalar('Val{0}/cAUC'.format(out_name), auroc_from_predictions_conservative, epoch)
                writer.add_scalar('Val{0}/cGAUC'.format(out_name), auroc_out_guaranteed_softmax_elision_conservative, epoch)
                
                writer.add_histogram('Val{0}confidences'.format(out_name), pred_out_confidences, epoch)
                writer.add_histogram('Val{0}/ub_confidences'.format(out_name), ub_el_out_confidences, epoch)

        lr_scheduler.step()
        if epoch%50 == 0 or epoch == 103 or epoch == 105:
            save_filename = model_folder + '/state_dicts/{0:03d}.pt'.format(epoch)
            torch.save(model.state_dict(), save_filename)
        torch.cuda.empty_cache()
        del data, img_batch_parts, lbl_batch_parts
        if 'reopen_data_file' in dir(train_loader_out): #loading 80M Tiny Images and thus training becomes much slower from epoch 2 if we do not do this.
            train_loader_out.reopen_data_file()
    stoptime = datetime.datetime.utcnow()
    dt = stoptime - starttime
    save_filename =  model_folder + '/state_dicts/' + str(epoch) + 'fin.pt'
    torch.save(model.state_dict(), save_filename)
    print('Training finished after {0} seconds'.format(dt))
    writer.close()
    return model_folder
