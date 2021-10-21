import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.cuda.amp import autocast


class CrossEntropyWLogits(torch.nn.Module):
    def __init__(self, reduction='mean'):
        # can support different kinds of reductions if needed
        super(CrossEntropyWLogits, self).__init__()
        assert reduction == 'mean' or reduction == 'none', \
            'utils.loss.CrossEntropyWLogits: reduction not recognized'
        self.reduction = reduction

    def forward(self, logits, targets):
        # shape of targets needs to match that of preds
        log_preds = torch.log_softmax(logits, dim=1)
        if self.reduction == 'mean':
            return torch.mean(torch.sum(-targets*log_preds, dim=1), dim=0)
        else:
            return torch.sum(-targets*log_preds, dim=1)


def vat_loss(args, G, F1, criterion, im_data, cls_out):
    """
    criterion is cross entropy with logits
    """
    # source domain
    rand_noise = torch.randn(im_data.shape).cuda(non_blocking=True)
    # normalize random noise to unit l_2 norm
    # The 1e-6 is the \xi in eq 12 of the VAT PAMI paper
    eps = 1e-6 * nn.functional.normalize(
        rand_noise.flatten(1)).reshape(im_data.shape)
    eps.requires_grad = True
    # NOTE : when y in KL-Div(x, y) has no gradients, it is the same as
    # minimizing cross-entropy(x, y)
    xent = criterion(F1(G(im_data + eps)), cls_out)
    with autocast(enabled=False):
        radv = torch.autograd.grad(xent, eps)[0]
    radv = args.vat_radius * nn.functional.normalize(
        radv.flatten(1)).reshape(im_data.shape).detach()

    vat_loss = criterion(
        torch.log_softmax(F1(G(im_data + radv)), dim=1), cls_out)

    return vat_loss

def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=False, eta=eta)
    out_t1 = F.softmax(out_t1, dim=1)
    loss_ent = -lamda * torch.mean(torch.sum(
        out_t1 * (torch.log(out_t1 + 1e-5)), 1)) # mean over examples in minibatch
    return loss_ent

def adentropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1, dim=1)
    loss_adent = lamda * torch.mean(torch.sum(
        out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent