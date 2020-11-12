import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import copy
import itertools
import datetime
import random
import lossfunctions

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# I-FGSM with projection
def ifgsm_attack():
    pass

#https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/adversarialbox/attacks.py
class LinfPGDAttack(object):
    def __init__(self, epsilon=0.1, n=20, loss_fn=lossfunctions.CrossEntropyLossDistr,
        random_start=False, device=torch.device('cpu'), a=None):
        """
        Attack parameter initialization. The attack performs n steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        if a == None:
            a = 2*epsilon / (n+1)
        self.device=device
        self.epsilon = epsilon
        self.n = n
        self.a = a
        self.rand = random_start
        self.loss_fn = loss_fn
        
        self.__name__ = 'LinfPGDAttack_epsilon={0:.3f}_n={1}_a={2:.3f}'.format(epsilon, n, a)

    
    def perturbt(self, X_in, y, model):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        model.eval()
        if self.n == 0:
            return X_in, y
        
        y_var = torch.autograd.Variable(y, requires_grad=False).to(self.device)
        
        Xt = X_in.clone().to(self.device)
        scores_orig = model(Xt)
        loss_orig = self.loss_fn(scores_orig, y_var)
        best_Xt = Xt.clone()
        loss_best = loss_orig
        
        if self.rand:
            Xt = Xt + (torch.rand_like(Xt)*2*self.epsilon - self.epsilon)
            Xt = torch.clamp(Xt, 0, 1)
        for i in range(self.n):
            model.zero_grad()
            X_var = torch.autograd.Variable(Xt, requires_grad=True).to(self.device)
            scores = model(X_var)
            loss_individual = self.loss_fn(scores, y_var)
            loss = loss_individual.sum()
            best_Xt = best_Xt*(loss_best>loss_individual).view(-1,1,1,1) + Xt*(loss_best<=loss_individual).view(-1,1,1,1)
            loss_best = torch.max(loss_individual, loss_best)
            loss.backward()
            grad = X_var.grad

            Xt += self.a * grad.sign()
            Xt = torch.min(torch.max(Xt, X_in - self.epsilon), X_in + self.epsilon)
            Xt = torch.clamp(Xt, 0, 1) # ensure valid pixel range
        loss_individual = self.loss_fn(scores, y_var)
        best_Xt = best_Xt*(loss_best>loss_individual).view(-1,1,1,1) + Xt*(loss_best<=loss_individual).view(-1,1,1,1)
        loss_best = torch.max(loss_individual, loss_best)
        del Xt, X_var, loss, scores
        model.zero_grad()
        return best_Xt.detach(), y 