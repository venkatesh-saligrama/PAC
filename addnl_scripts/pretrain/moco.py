import os
import sys
from collections import OrderedDict

sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
import torch.optim as optim

from model.basenet import AlexNetBase, VGGBase, Predictor
from model.resnet import resnet34
from utils.ioutils import FormattedLogItem
from utils.ioutils import get_log_str
from utils.ioutils import parse_args
from utils.ioutils import rm_format
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset_pretrain
from utils.misc import AverageMeter
from utils.ioutils import WandbWrapper
from utils.moco_utils import momentum_update, dequeue_and_enqueue
import shutil
import numpy as np
import random


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main(args, wandb):
    source_loader, target_loader, class_list = return_dataset_pretrain(
        args, pt_type='moco')
    if args.fs_ss:
        print('Setting sharing strategy to file_system')
        torch.multiprocessing.set_sharing_strategy('file_system')

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(args.max_num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.net == 'resnet34':
        G = resnet34()
        Gm = resnet34()
        # num input channels/input dim
        inc = 512
    elif args.net == 'alexnet':
        G = AlexNetBase()
        Gm = AlexNetBase()
        inc = 4096
    elif args.net == 'vgg':
        G = VGGBase()
        Gm = VGGBase()
        inc = 4096
    else:
        raise ValueError('Model cannot be recognized.')

    for param, param_m in zip(G.parameters(), Gm.parameters()):
        param_m.data.copy_(param.data)
        param_m.requires_grad = False # updated not using gradients

    G.cuda()
    Gm.cuda()
    G.train()
    Gm.train()

    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad:
            if 'classifier' in key:
                params += [{'params': [value], 'lr': 10 * args.lr,
                            'weight_decay': 0.0005}]
            else:
                params += [{'params': [value], 'lr': args.lr,
                            'weight_decay': 0.0005}]

    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    # initialize queue used by MoCo
    queue = torch.randn(
        (inc, args.queue_len), requires_grad=False).cuda(non_blocking=True)
    queue = nn.functional.normalize(queue, dim=0)
    queue_ptr = 0

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss()

    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_step = checkpoint['train_step']
            optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
            G.load_state_dict(checkpoint['G_state_dict'])
            Gm.load_state_dict(checkpoint['Gm_state_dict'])
            queue = checkpoint['queue']
            queue_ptr = checkpoint['queue_ptr']
        else:
            raise Exception('Resume file not found : {}'.format(args.resume))

    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    for step in range(start_step, start_step + args.steps):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step)

        if step % len(target_loader) == 0:
            data_iter_t = iter(target_loader)
        if step % len(source_loader) == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)

        # query and key. query data goes through encoder G and key data goes
        # through momentum encoder Gm
        im_data_q = torch.cat((data_s[0][0], data_t[0][0]),
                              dim=0).cuda(non_blocking=True)
        im_data_k = torch.cat((data_s[0][1], data_t[0][1]),
                              dim=0).cuda(non_blocking=True)

        # normalizing explicitly -- not part of the network
        feat_q = nn.functional.normalize(G(im_data_q))
        # Momentum update key encoder
        momentum_update(G, Gm, m=args.momentum)
        feat_k = nn.functional.normalize(Gm(im_data_k))

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [feat_q, feat_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [feat_q, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= args.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = criterion(logits, labels)

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        # dequeue old keys and enqueue new keys.
        # returns incremented queue_ptr
        with torch.no_grad():
            queue_ptr = dequeue_and_enqueue(queue, queue_ptr, feat_k)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        loss_meter.update(loss.item(), len(logits))
        acc1_meter.update(acc1.item(), len(logits))
        acc5_meter.update(acc5.item(), len(logits))

        if step % args.log_interval == 0 and step > 0:
            log_info = OrderedDict({
                'Train Step' : step,
                'Loss' : FormattedLogItem(loss_meter.avg, '{:.6f}'),
                'Top 1 Accuracy' : FormattedLogItem(acc1_meter.avg, '{:.2f}'),
                'Top 5 Accuracy' : FormattedLogItem(acc5_meter.avg, '{:.2f}')
            })

            wandb.log(rm_format(log_info))
            print(get_log_str(args, log_info, title='Pretraining Log'))
            loss_meter.reset()
            acc1_meter.reset()
            acc5_meter.reset()

            save_dict = {
                'train_step' : step,
                'G_state_dict' : G.state_dict(),
                'Gm_state_dict' : Gm.state_dict(),
                'optimizer_state_dict' : optimizer_g.state_dict(),
                'queue' : queue,
                'queue_ptr' : queue_ptr,
                'avg_loss' : loss_meter.avg,
                'avg_acc1' : acc1_meter.avg,
                'avg_acc5' : acc5_meter.avg,
            }
            if step % args.save_interval == 0:
                torch.save(save_dict, os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            if step % (args.save_interval * args.ckpt_freq) == 0:
                shutil.copyfile(
                    os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                    os.path.join(
                        args.save_dir, 'checkpoint_{}.pth.tar'.format(step)))

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    wandb = WandbWrapper(args.debug)
    if not args.project:
        args.project = 'ssda_mme-addnl_scripts'
    wandb.init(name=args.expt_name, dir=args.save_dir,
               config=args, reinit=True, project=args.project)
    main(args, wandb)

    wandb.join()