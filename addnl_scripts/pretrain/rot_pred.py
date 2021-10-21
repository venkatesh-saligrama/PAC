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
import shutil
import numpy as np
import random
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


def validate(G, F2, loader_s, loader_t):
    G.eval()
    F2.eval()
    acc_s = AverageMeter()
    acc_t = AverageMeter()
    accs = [acc_s, acc_t]
    loaders = [loader_s, loader_t]

    torch.backends.cudnn.benchmark = False
    for lidx, loader in enumerate(loaders):
        for i, data in enumerate(loader):
            imgs = data[0].reshape((-1,) + data[0].shape[2:]).cuda()
            rot_labels = data[2].reshape((-1,) + data[2].shape[2:]).cuda()
            preds = F2(G(imgs))
            preds = preds.argmax(dim=1)
            accs[lidx].update(
                (preds == rot_labels).sum()/float(len(imgs)), len(imgs))
    torch.backends.cudnn.benchmark = True
    return acc_s.avg, acc_t.avg

def main(args, wandb):
    if args.fs_ss:
        print('Setting sharing strategy to file_system')
        torch.multiprocessing.set_sharing_strategy('file_system')

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(args.max_num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    source_loader, target_loader, class_list = return_dataset_pretrain(args)
    torch.manual_seed(args.seed)

    # Training settings
    if args.net == 'resnet34':
        G = resnet34()
        # num input channels/input dim
        inc = 512
    elif args.net == 'alexnet':
        G = AlexNetBase()
        inc = 4096
    elif args.net == 'vgg':
        G = VGGBase()
        inc = 4096
    else:
        raise ValueError('Model cannot be recognized.')

    G.cuda()
    G.train()

    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad:
            if 'classifier' in key:
                # last layer learning rate is 10 times that of the rest of the
                # backbone (the is last layer)
                params += [{'params': [value], 'lr': 10*args.lr,
                            'weight_decay': 0.0005}]
            else:
                params += [{'params': [value], 'lr': args.lr,
                            'weight_decay': 0.0005}]

    # Predictor to predict image rotation
    F2 = Predictor(num_class=4, inc=inc, temp=args.T, hidden=args.cls_layers,
                   normalize=args.cls_normalize, cls_bias=args.cls_bias)
    F2.cuda()
    F2.train()
    scaler = GradScaler()

    # The classifiers have 10 times the learning rate of the backbone
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=args.wd, nesterov=True)
    # learning rates
    optimizer_f = optim.SGD(list(F2.parameters()), lr=10*args.lr, momentum=0.9,
                            weight_decay=args.wd, nesterov=True)

    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            G.load_state_dict(checkpoint['G_state_dict'])
            F2.load_state_dict(checkpoint['F2_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_f.load_state_dict(checkpoint['optimizer_f_state_dict'])
            start_step = checkpoint['Train Step'] + 1
            print('Resuming from train step {}'.format(start_step))
        else:
            raise Exception('No file found at {}'.format(args.resume))
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    for step in range(start_step, all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step)

        if step % len(target_loader) == 0:
            data_iter_t = iter(target_loader)
        if step % len(source_loader) == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)

        im_data_s = data_s[0].reshape(
            (-1,) + data_s[0].shape[2:]).cuda(non_blocking=True)
        rot_labels_s = data_s[2].reshape(
            (-1,) + data_s[2].shape[2:]).cuda(non_blocking=True)
        im_data_t = data_t[0].reshape(
            (-1,) + data_t[0].shape[2:]).cuda(non_blocking=True)
        rot_labels_t = data_t[2].reshape(
            (-1,) + data_t[2].shape[2:]).cuda(non_blocking=True)

        with autocast():
            all_imgs = torch.cat((im_data_s, im_data_t), dim=0)
            all_labels = torch.cat(
                (rot_labels_s, rot_labels_t), dim=0)
            all_feats = G(all_imgs)
            preds = F2(all_feats)
            loss = criterion(preds, all_labels)

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_g)
        scaler.step(optimizer_f)
        scaler.update()

        if step % args.log_interval == 0:
            log_info = OrderedDict({
                'Train Step': step,
                'Loss': FormattedLogItem(loss.item(), '{:.6f}'),
            })
            wandb.log(rm_format(log_info))
            log_str = get_log_str(args, log_info, title='Training Log')
            print(log_str)

        if step % args.save_interval == 0 and step > 0:
            with torch.no_grad():
                acc_s, acc_t = validate(
                    G, F2, source_loader, target_loader)
            log_info = OrderedDict({
                'Train Step': step,
                'Source Acc': FormattedLogItem(100. * acc_s, '{:.2f}'),
                'Target Acc': FormattedLogItem(100. * acc_t, '{:.2f}')
            })
            wandb.log(rm_format(log_info))
            log_str = get_log_str(args, log_info, title='Validation Log')
            print(log_str)
            G.train()
            F2.train()

            torch.save({
                'Train Step' : step,
                'G_state_dict' : G.state_dict(),
                'F2_state_dict' : F2.state_dict(),
                'optimizer_g_state_dict' : optimizer_g.state_dict(),
                'optimizer_f_state_dict' : optimizer_f.state_dict(),
            }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))

            if step % (args.save_interval * args.ckpt_freq) == 0:
                shutil.copyfile(
                    os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                    os.path.join(
                        args.save_dir, 'checkpoint_{}.pth.tar'.format(step)))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    wandb = WandbWrapper(~args.use_wandb)
    if not args.project:
        args.project = 'ssda_mme-addnl_scripts'
    wandb.init(name=args.expt_name, dir=args.save_dir,
               config=args, reinit=True, project=args.project)
    main(args, wandb)

    wandb.join()