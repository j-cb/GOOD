import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

cross_entropy = torch.nn.CrossEntropyLoss()

def CrossEntropyLossDistr(scores, labels):
    #batch_size = outputs.size()[0]            # batch_size
    logprobs = F.log_softmax(scores, dim=-1)
    losses = -torch.sum(labels*logprobs, dim=-1)
    return losses

def MultiBinaryCrossEntropyLossDistr(scores, labels):
    #batch_size = outputs.size()[0]            # batch_size
    scores_true = (labels*scores).sum(dim=-1)
    score_differences = scores - scores_true.view(-1,1)
    losses_resp_class = (torch.exp(score_differences) + 1).log()
    losses = losses_resp_class.sum(dim=-1)
    return losses

def CrossEntropyLossNumbers(scores, labels):
    #batch_size = outputs.size()[0]            # batch_size
    logprobs = F.log_softmax(scores, dim=-1)
    losses = -logprobs[labels]
    return losses

def LogConf(scores, labels=None):
    logprobs = F.log_softmax(scores, dim=-1)
    losses = logprobs.max(dim=-1)[0]
    return losses

def LogitConf0(scores, labels=None):
    losses = scores[:,0] - scores.mean(dim=-1)
    return losses

def LogConfTarget(scores, labels):
    logprobs = F.log_softmax(scores, dim=-1)
    losses = logprobs[:,labels]
    return losses

def LogitConfTarget(scores, labels):
    losses = scores[:,labels] - scores.mean(dim=-1)
    return losses

def UbLogConf(ub_logit_differences, labels=None):
    # ub_score_differences should have a shape N x K x K
    log_reciprocal_p = (-ub_logit_differences).logsumexp(dim=-1)
    ub_log_conf = -(log_reciprocal_p.min(dim=-1)[0])
    return ub_log_conf

def LbLogUnConf(ub_logit_differences, labels=None):
    # ub_score_differences should have a shape N x K x K
    log_reciprocal_p = ((-ub_logit_differences).exp().sum(dim=-1)-0.999).log() #0.999 to avoid log(0)
    lb_log_unconf = (log_reciprocal_p.max(dim=-1)[0])
    return lb_log_unconf
