import numpy as np
np.seterr(all='ignore')

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torchvision
import attacks
import lossfunctions
import plotting
import scipy.special
import matplotlib.pyplot as plt
import matplotlib
import sklearn.metrics
import datetime
import os

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def auroc_conservative(values_in, values_out):
    s = 0
    for i in range(len(values_in)):
        s += (values_out < values_in[i]).sum() #+ 0.5*(values_out == values_in[i]).sum()
    s /= len(values_in)*len(values_out)
    return s

def auroc(values_in, values_out):
    y_true = len(values_in)*[1] + len(values_out)*[0]
    y_score = np.concatenate([values_in, values_out])
    return sklearn.metrics.roc_auc_score(y_true, y_score)

def accuracy(P, L):
    """Mean euclidean distance between two N✕2 numpy arrays"""
    C = (P == L)
    corr = np.sum(C)
    acc = corr/len(C)
    return acc

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def logsumexp(x, axis=-1):
    return scipy.special.logsumexp(x, axis=axis)

def log_confs_from_logits(logits):
    logits_normalized = logits - logits.max(axis=-1, keepdims=True)
    log_confidences = -logsumexp(logits_normalized, axis=-1)
    return log_confidences

def right_and_wrong_confidences_from_logits(logits, labels):
    #logits_normalized_by_label = logits - logits[:,labels]
    probabilities = softmax(logits, axis=-1)  
    right_confidences = np.copy(probabilities[range(probabilities.shape[0]), labels])
    probabilities[range(probabilities.shape[0]), labels] = 0
    wrong_confidences = probabilities.max(axis=-1)
    return right_confidences, wrong_confidences

def ub_log_confs_from_ud_logits(ud_logits, force_diag_0=False): #upper bound differences matrix
    if force_diag_0: #with elision, this is already given
        for i in range(ud_logits.shape[-1]): 
            ud_logits[:, i, i] = 0
    ub_log_probs = -logsumexp(-ud_logits, axis=-1)
    ub_log_confs = np.amax(ub_log_probs, axis=-1)
    return ub_log_confs
    
def conf_stats_from_log_confs(log_confs, th, k):
    confidences = np.exp(np.nan_to_num(log_confs))
    confidence_mean = np.mean(confidences)
    confidence_median = np.median(confidences)
    confidences_below_th = np.sum(confidences < th)
    confidences_above_th = np.sum(confidences > th)
    lowest_conf_indices = confidences.argsort()[:k]
    highest_conf_indices = (-confidences).argsort()[:k]
    return confidences, confidence_mean, confidence_median, confidences_below_th, confidences_above_th, lowest_conf_indices, highest_conf_indices 
        
def accuracy_above(logits, labels, th):
    log_confs = log_confs_from_logits(logits)
    confidences = np.exp(np.nan_to_num(log_confs))
    above = confidences > th
    pred_classes = np.argmax(logits, axis=1)
    if np.sum(above) == 0:
        return None
    acc_above = accuracy(pred_classes[above], labels[above])
    return acc_above  

def frac_above(logits, labels, th):
    log_confs = log_confs_from_logits(logits)
    confidences = np.exp(np.nan_to_num(log_confs))
    above = confidences > th
    f_above = sum(above) / len(logits)
    return f_above
    
def evaluate_ibp_lc(model, test_loader_in, test_loader_out_list, eps, conf_th, device, short_name=None, n_pgd=0, model_path=None, n_samples=30000, do_accuracy_above=False, save_plots=False): #pass model_path to save evaluation graphs in the model directory
    starttime_eval = datetime.datetime.utcnow()
    torch.set_printoptions(profile="full")
    in_name = str(test_loader_in.dataset.__repr__()).split()[1]
    num_classes = model.num_classes
    print('Evaluating {0} for accuracy and confidence-based OOD detection.'.format(model_path))
    model.eval()
    k = min(32, n_samples)  #number of worst case images to be saved
    if n_pgd > 0:
        pgd_attack_out = attacks.LinfPGDAttack(epsilon=eps, n=n_pgd, loss_fn=lossfunctions.LogConf,
        random_start=False, device=device)
        pgd_attack_in = attacks.LinfPGDAttack(epsilon=eps, n=n_pgd, loss_fn=nn.LogConf,
        random_start=False, device=device)
    
    #prepare LaTeX-usable outputs
    table_str_0 = f'Model '
    table_str_1 = f'{model_path} '

    print('\nIn dataset: {0}'.format(str(test_loader_in.dataset.__repr__())))
    n_in_samples = min(n_samples, len(test_loader_in.dataset))
    number_of_in_datapoints = 0
    labels_in = np.zeros(n_in_samples, dtype=int)
    logits_in = np.zeros((n_in_samples, num_classes))
    pred_scores_in = np.zeros((n_in_samples, num_classes))
    adversarial_logits_in = np.zeros((n_in_samples, num_classes))
    adversarial_pred_scores_in = np.zeros((n_in_samples, num_classes))
    ud_logits_in = np.zeros((n_in_samples, num_classes, num_classes))
    for batch, (img_torch, lbl) in enumerate(test_loader_in):
        bs = len(lbl)
        number_of_in_datapoints += bs
        if number_of_in_datapoints > n_in_samples: #adjust length of last batch
            assert number_of_in_datapoints - n_in_samples < bs
            bs = bs - (number_of_in_datapoints - n_in_samples)
            img_torch = img_torch[:bs]
            lbl = lbl[:bs]
        labels_in[test_loader_in.batch_size*batch:test_loader_in.batch_size*batch+bs] = lbl.numpy()
        img = img_torch.to(device)
        lbl_in_batch = lbl.to(device)        
        
        #get clean, adversarial and guaranteed predictions
        logit_in_batch = model(img)
        pred_score_in_batch = logit_in_batch.softmax(dim=-1)
        if n_pgd > 0:
            #Run an adversarial attack trying to make prediction wrong.
            adversarial_img, _ = pgd_attack_in.perturbt(img, lbl, model)
            model.eval()
            adversarial_logit_in_batch = model(adversarial_img)
            adversarial_pred_score_in_batch = adversarial_logit_in_batch.softmax(dim=-1)
        else:
            adversarial_img = img
            adversarial_logit_in_batch = model(adversarial_img)
            adversarial_pred_score_in_batch = adversarial_logit_in_batch.softmax(dim=-1)    
        l_logit_in_batch, u_logit_in_batch, ud_logit_in_batch = model.ibp_elision_forward(torch.clamp(img - eps, 0, 1), torch.clamp(img + eps, 0, 1), num_classes)   
        
        labels_in[test_loader_in.batch_size*batch:test_loader_in.batch_size*batch+bs] = lbl_in_batch.detach().cpu().numpy()
        logits_in[test_loader_in.batch_size*batch:test_loader_in.batch_size*batch+bs] = logit_in_batch.detach().cpu().numpy()
        pred_scores_in[test_loader_in.batch_size*batch:test_loader_in.batch_size*batch+bs] = pred_score_in_batch.detach().cpu().numpy()      
        adversarial_logits_in[test_loader_in.batch_size*batch:test_loader_in.batch_size*batch+bs] = adversarial_logit_in_batch.detach().cpu().numpy()
        adversarial_pred_scores_in[test_loader_in.batch_size*batch:test_loader_in.batch_size*batch+bs] = adversarial_pred_score_in_batch.detach().cpu().numpy()
        ud_logits_in[test_loader_in.batch_size*batch:test_loader_in.batch_size*batch+bs] = ud_logit_in_batch.detach().cpu().numpy()
        
        #save the first batch of in-distribution images and prepare folders
        if model_path and batch == 0:
            model_folder = 'evals/' +''.join(model_path.split('/')[:-2]) + 'eval_' + starttime_eval.strftime("%m-%d-%H-%M-%S") + 'e=' + str(eps)
            print(rf'Evaluation outputs saved in {model_folder}')
            os.makedirs(model_folder + '/sample_images/', exist_ok=True)
            os.makedirs(model_folder + '/values/', exist_ok=True)
            save_path = model_folder + '/sample_images/{1}_{0}first_batch_val_in'.format(str(test_loader_in.dataset).split()[1], short_name)
            vutils.save_image(img.detach(), save_path + '.png', normalize=False)
        
        #stop loader iteration if specified number of samples is reached
        if number_of_in_datapoints >= n_in_samples:
            break
    
    #analysis of the in-distribution results
    pred_in_classes = np.argmax(logits_in, axis=1)
    pred_adversarial_in_classes = np.argmax(adversarial_logits_in, axis=1)
    in_accuracy = accuracy(pred_in_classes, labels_in)
    adv_in_accuracy = accuracy(pred_adversarial_in_classes, labels_in)
        
    pred_in_log_confidences = log_confs_from_logits(logits_in)
    pred_in_confidences, pred_in_confidence_mean, pred_in_confidence_median, pred_in_confidences_below_th, pred_in_confidences_above_th, pred_in_lowest_conf_indices, pred_in_highest_conf_indices = conf_stats_from_log_confs(pred_in_log_confidences, conf_th, k)
    
    pred_in_right_confidences, pred_in_wrong_confidences = right_and_wrong_confidences_from_logits(logits_in, labels_in)
    pred_in_worst_conf_indices = (-pred_in_wrong_confidences).argsort()[:k]
    pred_in_worst_ce_indices = pred_in_right_confidences.argsort()[:k]
    
    #analyze if and by how much the accuracy improves if low confidence predictions are discarded
    if do_accuracy_above:
        #conf_thresholds = [0.0, 0.1, 0.11, 0.12, 0.13, 0.14, 0.16, 0.18, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
        #for conf in conf_thresholds:
        #    print(f'Accuracy wherever the confidence is above {conf:.2f}: {accuracy_above(logits_in, labels_in, conf):.2%}')
        t = np.linspace(0, 1, 1000)
        accuracy_above_vectorized = np.vectorize(accuracy_above, excluded=[0,1])
        plt.plot(t, accuracy_above_vectorized(logits_in, labels_in, t), c='#44DD44')
        plt.box(False)
        save_path = model_folder + f'/sample_images/{short_name}_accuracy_above_threshold.png'
        plt.axvline(x=1.0, linestyle='--', c='#BBBBBB')
        plt.axvline(x=0.1, linestyle='--', c='#BBBBBB')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(which='major', axis='y')
        plt.yticks([0.9,0.92,0.94, 0.96, 0.98, 1.0])
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        t = np.linspace(0, 1, 1000)
        frac_above_vectorized = np.vectorize(frac_above, excluded=[0,1])
        plt.plot(t, frac_above_vectorized(logits_in, labels_in, t), c='#4444DD')
        plt.box(False)
        save_path = model_folder + f'/sample_images/{short_name}_frac_above_threshold.png'
        plt.axvline(x=1.0, linestyle='--', c='#BBBBBB')
        plt.axvline(x=0.1, linestyle='--', c='#BBBBBB')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(which='major', axis='y')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    pred_adv_in_log_confidences = log_confs_from_logits(adversarial_logits_in)
    pred_adv_in_confidences, pred_adv_in_confidence_mean, pred_adv_in_confidence_median, pred_adv_in_confidences_below_th, pred_adv_in_confidences_above_th, pred_adv_in_lowest_conf_indices, pred_adv_in_highest_conf_indices = conf_stats_from_log_confs(pred_adv_in_log_confidences, conf_th, k)
        
    ub_elision_in_log_probs = -logsumexp(-ud_logits_in, axis=-1)
    ub_elision_in_log_probs[:,labels_in] = -1
    ub_elision_in_log_confs = np.amax(ub_elision_in_log_probs, axis=-1)
    ub_elision_in_confidences = np.exp(ub_elision_in_log_confs)
    ub_elision_mean_in_confidence = np.mean(ub_elision_in_confidences)
    ub_elision_median_in_confidence = np.median(ub_elision_in_confidences)
    ub_elision_in_below_th = np.sum(ub_elision_in_confidences < conf_th)
        
    auroc_vs_adversarials = auroc(pred_in_confidences, pred_adv_in_confidences)
    
    print('The accuracy of the predictions of the {0} model on the in-distribution is {1:.2f}%. '
          .format(model.__name__, in_accuracy*100),
          'Mean confidence: {0:.4f}. Median confidence: {1:.4f}. Above {2}: {3}/{4}.,'
          .format(pred_in_confidence_mean, pred_in_confidence_median, conf_th, pred_in_confidences_above_th, n_in_samples)
         )
    table_str_0 += f'& Acc. '
    table_str_1 += f'& {100*in_accuracy:03.1f} '
    if n_pgd > 0:
        print('Adversarial in samples under {0}: Accuracy: {1}'
              .format(pgd_attack_in.__name__,  adv_in_accuracy),
              'Mean confidence: {0:.4f}. Median confidence: {1:.4f}. Above {2}: {3}/{4}.'
              .format(pred_adv_in_confidence_mean, pred_adv_in_confidence_median, conf_th, pred_adv_in_confidences_above_th, n_in_samples),
              'AUROC in confidences vs adversarial in confidences: {0:.4f}'
              .format(auroc_vs_adversarials)
             )
        
    if model_path:
        if save_plots:
            plotting.save_images(k, test_loader_in, pred_in_lowest_conf_indices, pred_in_confidences, pred_scores_in, in_name, model_folder, descr='pred_lowest', index_selection='lowest', short_name=short_name)
            plotting.save_images(k, test_loader_in, pred_in_worst_conf_indices, pred_in_wrong_confidences, pred_scores_in, in_name, model_folder, descr='pred_worst_wrong', index_selection='worst_wrong_prob', short_name=short_name)
            plotting.save_images(k, test_loader_in, pred_in_worst_ce_indices, pred_in_right_confidences, pred_scores_in, in_name, model_folder, descr='pred_worst_ce', index_selection='worst_right_prob', short_name=short_name)
            plt.hist(pred_in_confidences, color='g', bins=50, range=(0.1, 1.0), log=False, label= f'{in_name}_in_pred_confidences')
            plt.ylim(ymin=0, ymax=len(pred_in_confidences)/1.5)
            plt.xlim(xmin=0.09, xmax=1.01)
            plt.axvline(pred_in_confidences.mean(), color='k', linestyle='dashed', linewidth=2)
            save_path = model_folder + f'/sample_images/{short_name}{in_name}_in_pred_confidences_hist'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        np.savetxt(model_folder + f'/values/{short_name}_{in_name}_pred_in_confidences.txt', pred_in_confidences)
            
    returns = {}
    for test_loader_out in test_loader_out_list:
        starttime_eval_out = datetime.datetime.utcnow()
        out_name = str(test_loader_out.dataset.__repr__()).split()[1]
        print('\nOut dataset: {0}'.format(test_loader_out.dataset.__repr__()))
        
        n_out_samples = min(n_samples, len(test_loader_out.dataset))
        print(f'{n_out_samples} out samples.')
        number_of_out_datapoints = 0
        #initialize numpy arrays for result values
        logits_out = np.zeros((n_out_samples, num_classes))
        l_logits_out = np.zeros((n_out_samples, num_classes))
        u_logits_out = np.zeros((n_out_samples, num_classes))
        ud_logits_out = np.zeros((n_out_samples, num_classes, num_classes))
        pred_scores_out = np.zeros((n_out_samples, num_classes))
        adversarial_logits_out = np.zeros((n_out_samples, num_classes))
        adversarial_pred_scores_out = np.zeros((n_out_samples, num_classes))
        th_eps_out = np.zeros(n_out_samples)

        for batch, (img_torch, lbl) in enumerate(test_loader_out):
            bs = len(lbl)
            number_of_out_datapoints += bs
            if number_of_out_datapoints > n_out_samples: #as above, reduce the length of the last batch if it overflows the total number of samples.
                assert number_of_out_datapoints - n_out_samples < bs
                bs = bs - (number_of_out_datapoints - n_out_samples)
                img_torch = img_torch[:bs]
                lbl = 0*lbl[:bs]
            logit_out_batch = np.zeros((bs, num_classes))
            img_out = img_torch.to(device)
            lbl_out_batch = lbl.to(device)
            
            #get the model outputs
            logit_out_batch = model(img_out)
            pred_score_out_batch = logit_out_batch.softmax(dim=-1)
            l_logit_out_batch, u_logit_out_batch, ud_logit_out_batch = model.ibp_elision_forward(torch.clamp(img_out - eps, 0, 1), torch.clamp(img_out + eps, 0, 1), num_classes)
            if n_pgd > 0:
                adversarial_out_img, _ = pgd_attack_out.perturbt(img_out, lbl, model)
                adversarial_logit_out_batch = model(adversarial_out_img)
                adversarial_pred_score_out_batch = adversarial_logit_out_batch.softmax(dim=-1)
            else:
                adversarial_out_img = img_out
                adversarial_logit_out_batch = model(adversarial_out_img)
                adversarial_pred_score_out_batch = adversarial_logit_out_batch.softmax(dim=-1)
                            
            logits_out[test_loader_out.batch_size*batch:test_loader_out.batch_size*batch+bs] = logit_out_batch.detach().cpu().numpy()
            pred_scores_out[test_loader_out.batch_size*batch:test_loader_out.batch_size*batch+bs] = pred_score_out_batch.detach().cpu().numpy() 

            l_logits_out[test_loader_out.batch_size*batch:test_loader_out.batch_size*batch+bs] = l_logit_out_batch.detach().cpu().numpy()
            u_logits_out[test_loader_out.batch_size*batch:test_loader_out.batch_size*batch+bs] = u_logit_out_batch.detach().cpu().numpy()
            ud_logits_out[test_loader_out.batch_size*batch:test_loader_out.batch_size*batch+bs] = ud_logit_out_batch.detach().cpu().numpy()
            
            adversarial_logits_out[test_loader_out.batch_size*batch:test_loader_out.batch_size*batch+bs] = adversarial_logit_out_batch.detach().cpu().numpy()
            adversarial_pred_scores_out[test_loader_out.batch_size*batch:test_loader_out.batch_size*batch+bs] = adversarial_pred_score_out_batch.detach().cpu().numpy() 
                
            if model_path and batch == 0: #save some example images that the evaluation is run on
                save_path = model_folder + f'/sample_images/{short_name}_{out_name}first_batch_val'
                vutils.save_image(img_out.detach(), save_path + '.png', normalize=False)
                save_path = model_folder + f'/sample_images/{short_name}_{out_name}first_batch_val_pgd'
                vutils.save_image(adversarial_out_img.detach(), save_path + '.png', normalize=False)
            save_adversarials = False #set this to True to save computed adversarial images.
            if model_path and n_pgd > 0 and save_adversarials:
                os.makedirs(model_folder + f'/sample_images/{short_name}_{out_name}_adv/', exist_ok=True)
                save_path = model_folder + f'/sample_images/{short_name}_{out_name}_adv/_batch{batch:3d}'
                for i in range(len(adversarial_out_img)):
                    vutils.save_image(img_out[i].detach(), f'{save_path}_out{i}.png', normalize=False)
                    vutils.save_image(adversarial_out_img[i].detach(), f'{save_path}_adv{i}.png', normalize=False)
            if number_of_out_datapoints >= n_out_samples:
                break
      
        pred_out_log_confidences = log_confs_from_logits(logits_out)
        pred_out_confidences, pred_out_confidence_mean, pred_out_confidence_median, pred_out_confidences_below_th, pred_out_confidences_above_th, pred_out_lowest_conf_indices, pred_out_highest_conf_indices = conf_stats_from_log_confs(pred_out_log_confidences, conf_th, k)
        
        #we calculate the bounds based on ibp with and without elision so we can judge the difference if we need to
        out_logit_spreads = u_logits_out.max(axis=-1) - l_logits_out.min(axis=-1)
        ub_spread_out_log_confidences = out_logit_spreads - np.log(num_classes)
        ub_spread_out_confidences, ub_spread_out_confidence_mean, ub_spread_out_confidence_median, ub_spread_out_confidences_below_th, ub_spread_out_confidences_above_th, ub_spread_out_lowest_conf_indices, ub_spread_out_highest_conf_indices = conf_stats_from_log_confs(ub_spread_out_log_confidences, conf_th, k)

        ub_out_logit_differences = u_logits_out[:,:,np.newaxis] - l_logits_out[:,np.newaxis,:]
        ub_out_log_confidences = ub_log_confs_from_ud_logits(ub_out_logit_differences, force_diag_0=True)
        ub_out_confidences, ub_out_confidence_mean, ub_out_confidence_median, ub_out_confidences_below_th, ub_out_confidences_above_th, ub_out_lowest_conf_indices, ub_out_highest_conf_indices = conf_stats_from_log_confs(ub_out_log_confidences, conf_th, k)
        
        ub_el_out_log_confidences = ub_log_confs_from_ud_logits(ud_logits_out, force_diag_0=False)
        ub_el_out_confidences, ub_el_out_confidence_mean, ub_el_out_confidence_median, ub_el_out_confidences_below_th, ub_el_out_confidences_above_th, ub_el_out_lowest_conf_indices, ub_el_out_highest_conf_indices = conf_stats_from_log_confs(ub_el_out_log_confidences, conf_th, k)

        pred_adv_out_log_confidences = log_confs_from_logits(adversarial_logits_out)
        pred_adv_out_confidences, pred_adv_out_confidence_mean, pred_adv_out_confidence_median, pred_adv_out_confidences_below_th, pred_adv_out_confidences_above_th, pred_adv_out_lowest_conf_indices, pred_adv_out_highest_conf_indices = conf_stats_from_log_confs(pred_adv_out_log_confidences, conf_th, k)
        
        
        auroc_from_predictions = auroc(pred_in_confidences, pred_out_confidences)
        auroc_out_guaranteed_spread = auroc(pred_in_confidences, np.nan_to_num(ub_spread_out_confidences))
        auroc_out_guaranteed_softmax = auroc(pred_in_confidences, np.nan_to_num(ub_out_confidences))
        auroc_out_guaranteed_softmax_elision = auroc(pred_in_confidences, np.nan_to_num(ub_el_out_confidences))
        auroc_out_adversarial = auroc(pred_in_confidences, pred_adv_out_confidences)

        auroc_from_predictions_conservative = auroc_conservative(pred_in_confidences, pred_out_confidences)
        auroc_out_guaranteed_spread_conservative = auroc_conservative(pred_in_confidences, ub_spread_out_confidences)
        auroc_out_guaranteed_softmax_conservative = auroc_conservative(pred_in_confidences, ub_out_confidences)
        auroc_out_guaranteed_softmax_elision_conservative = auroc_conservative(pred_in_confidences, ub_el_out_confidences)
        auroc_out_adversarial_conservative = auroc_conservative(pred_in_confidences, pred_adv_out_confidences)

        if model_path:
            if save_plots:
                plotting.save_images(k, test_loader_out, pred_out_highest_conf_indices, pred_out_confidences, pred_scores_out, out_name, model_folder, descr='pred', index_selection='highest', short_name=short_name)
                plt.hist(th_eps_out, bins=100, log=False, label= 'eps with guarantee <{0:.2f}'.format(conf_th))
                save_path = model_folder + f'/sample_images/{short_name}{out_name}eps_hist'            
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                plotting.save_images(k, test_loader_out, pred_adv_out_highest_conf_indices, pred_adv_out_confidences, adversarial_pred_scores_out, out_name, model_folder, descr='adv', index_selection='highest', short_name=short_name)
                plotting.save_images(k, test_loader_out, ub_el_out_highest_conf_indices, ub_el_out_confidences, pred_scores_out, out_name, model_folder, descr='ub_el', index_selection='highest', short_name=short_name)
                bins = np.linspace(0.1, 0.4, 200)
                plt.hist(pred_out_confidences, bins=bins, log=False, label= 'out_pred')
                plt.hist(ub_el_out_confidences, bins=bins, log=False, label= 'out_ub')
                plt.hist(pred_in_confidences, bins=bins, log=False, label= 'in_pred')
                plt.legend()
                save_path = model_folder + f'/sample_images/{short_name}{out_name}compare_hist'
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

            np.savetxt(model_folder + f'/values/{short_name}{out_name}_pred_out_confidences.txt', pred_out_confidences)
            np.savetxt(model_folder + f'/values/{short_name}{out_name}_pred_adv_out_confidences.txt', pred_adv_out_confidences)
            np.savetxt(model_folder + f'/values/{short_name}{out_name}_ub_el_out_confidences.txt', ub_el_out_confidences)
            

        #The non-elision values are mainly for comparing the methods and diagnosis but give less tight results, so you might want to comment them out.
        print('The mean confidence of the predictions of the {0} model is {1:.4f} on the in-distribution and {2:.4f} on the out-distribution.'.format(model.__name__, pred_in_confidence_mean, pred_out_confidence_mean))
        print('For epsilon={0} on the out-distribution, the guaranteed confidence upper bound from logit spread has a mean of {1} and a median of {2}.'.format(eps, ub_spread_out_confidence_mean, ub_spread_out_confidence_median))
        print('For epsilon={0} on the out-distribution, the guaranteed confidence upper bound from softmax has a mean of {1} and a median of {2}.'.format(eps, ub_out_confidence_mean, ub_out_confidence_median))
        print('For epsilon={0} on the out-distribution, the guaranteed confidence upper bound from softmax with elision has a mean of {1} and a median of {2}.'.format(eps, ub_el_out_confidence_mean, ub_el_out_confidence_median))
        print('In-samples confidence above {0}: {1}/{2} = {3}. Out-samples predicted confidence above {0}: {4}/{5} = {6}. Out-samples confidence {9}-guaranteed from spread below {0}: {7}/{5} = {8}. Out-samples confidence {9}-guaranteed from softmax below {0}: {10}/{5} = {11}. Out-samples confidence {9}-guaranteed from softmax with elision below {0}: {12}/{5} = {13}.'.format(
            conf_th, pred_in_confidences_above_th, n_in_samples,  pred_in_confidences_above_th/n_in_samples,  pred_out_confidences_above_th, n_out_samples, pred_out_confidences_above_th/n_out_samples,
            ub_spread_out_confidences_below_th, ub_spread_out_confidences_below_th/n_out_samples, eps,
            ub_out_confidences_below_th, ub_out_confidences_below_th/n_out_samples,
            ub_el_out_confidences_below_th, ub_el_out_confidences_below_th/n_out_samples)
        )
        print('Prediction AUROC: {0}. {1}-guaranteed AUROC: {2} (logit spread), {3} (softmax), {4} (elision softmax).'.format(auroc_from_predictions, eps, auroc_out_guaranteed_spread, auroc_out_guaranteed_softmax, auroc_out_guaranteed_softmax_elision))
        print('Conservative Prediction AUROC: {0}. {1}-guaranteed AUROC: {2} (logit spread), {3} (softmax), {4} (elision softmax).'.format(auroc_from_predictions_conservative, eps, auroc_out_guaranteed_spread_conservative, auroc_out_guaranteed_softmax_conservative, auroc_out_guaranteed_softmax_elision_conservative))
        if n_pgd > 0:
            print('Adversarial outs under {0}: Mean confidence: {1:.4f}. Median confidence: {2:.4f}. Confidence above {3}: {4}/{5} = {6:.4f}. AUROC: {7}.\n'.format(pgd_attack_out.__name__, pred_adv_out_confidence_mean, pred_adv_out_confidence_median, conf_th, pred_adv_out_confidences_above_th, len(pred_adv_out_confidences), pred_adv_out_confidences_above_th / len(pred_adv_out_confidences), auroc_out_adversarial))
        latex_str = 'Prediction/Adversarial/Guaranteed AUROC: {0:03.1f} & {1:03.1f} & {2:03.1f}'.format(100*auroc_from_predictions, 100*auroc_out_adversarial, 100*auroc_out_guaranteed_softmax_elision)
        latex_str_conservative = 'Conservative Prediction/Adversarial/Guaranteed AUROC: {0:03.1f} & {1:03.1f} & {2:03.1f}'.format(100*auroc_from_predictions_conservative, 100*auroc_out_adversarial_conservative, 100*auroc_out_guaranteed_softmax_elision_conservative)
        print(latex_str)
        print(latex_str_conservative)
        table_str_0 += f'& {out_name} P & A & G ' 
        table_str_1 += f' & {auroc_from_predictions_conservative*100:03.1f} & {auroc_out_adversarial_conservative*100:3.1f} & {auroc_out_guaranteed_softmax_elision_conservative*100:3.1f} '
        returns[out_name] = (
            in_accuracy, pred_in_confidences, pred_in_confidence_mean, pred_in_confidences_above_th, n_in_samples, 
            pred_out_confidences, pred_out_confidence_mean,  pred_out_confidences_above_th, n_out_samples,
            ub_el_out_confidences, ub_el_out_confidence_mean, ub_el_out_confidence_median, ub_el_out_confidences_below_th,
            auroc_from_predictions, auroc_out_guaranteed_softmax_elision, 
            auroc_from_predictions_conservative, auroc_out_guaranteed_softmax_elision_conservative, 
            pred_adv_out_confidences, pred_adv_out_confidence_mean, pred_adv_out_confidence_median, pred_adv_out_confidences_above_th
        )
        print('Eval for this out dataset: {0}.\n'.format(datetime.datetime.utcnow() - starttime_eval_out))
    table_str_0 += f' \\\\'
    table_str_1 += f' \\\\' #Note that the adversarial evaluations are sub-optimal since a dedicated attack evaluation is needed to get strong results.
    print(table_str_0) 
    print(table_str_1)
    print('Total time for eval: {0}.\n'.format(datetime.datetime.utcnow() - starttime_eval))
    return returns

#def eps_reduction_for_th(img_out, model, eps_start, )