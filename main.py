import os
import random
import shutil
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from model.basenet import AlexNetBase, VGGBase, Predictor
from model.resnet import resnet34
from utils.eval import test
from utils.ioutils import FormattedLogItem
from utils.ioutils import WandbWrapper
from utils.ioutils import gen_unique_name
from utils.ioutils import get_log_str
from utils.ioutils import parse_args
from utils.ioutils import rm_format
from utils.loss import CrossEntropyWLogits
from utils.loss import entropy, adentropy
from utils.loss import vat_loss
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset


def main(args, wandb):
    if args.fs_ss:
        print('Setting sharing strategy to file_system')
        torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(args.max_num_threads)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    source_loader, target_loader, target_loader_unl, target_loader_val, \
        target_loader_test, class_list = return_dataset(args)

    # Feature extractor
    if args.net == 'resnet34':
        G = resnet34()
        inc = 512 # feat dim
    elif args.net == 'alexnet':
        G = AlexNetBase()
        inc = 4096
    elif args.net == 'vgg':
        G = VGGBase()
        inc = 4096
    else:
        raise ValueError('Model cannot be recognized.')

    if args.backbone_path:
        if os.path.isfile(args.backbone_path):
            checkpoint = torch.load(args.backbone_path)
            G.load_state_dict(checkpoint['G_state_dict'])
        else:
            raise Exception(
                'Path for backbone {} not found'.format(args.backbone_path))

    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad:
            if 'classifier' in key:
                # last layer learning rate is 10 times that of the rest of the
                # backbone (the is last layer)
                params += [{'params': [value], 'lr': 10*args.lr,
                            'weight_decay': args.wd}]
            else:
                params += [{'params': [value], 'lr': args.lr,
                            'weight_decay': args.wd}]

    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T,
                   hidden=args.cls_layers, normalize=args.cls_normalize,
                   cls_bias=args.cls_bias)

    G.cuda()
    F1.cuda()

    G.train()
    F1.train()

    scaler = GradScaler() # amp

    f_params = list(F1.parameters())
    # The classifiers have 10 times the learning rate of the backbone
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=args.wd, nesterov=True)
    optimizer_f = optim.SGD(f_params, lr=10*args.lr, momentum=0.9,
                            weight_decay=args.wd, nesterov=True)

    # To resume from a checkpoint
    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            G.load_state_dict(checkpoint['G_state_dict'])
            F1.load_state_dict(checkpoint['F1_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_f.load_state_dict(checkpoint['optimizer_f_state_dict'])
            start_step = checkpoint['step'] + 1
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
    criterion1 = CrossEntropyWLogits(reduction='none')

    # Iterators for data loaders
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)

    best_acc = 0 # best validation accuracy
    for step in range(start_step, args.steps):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step)

        if step % len(target_loader) == 0 and step > 0:
            data_iter_t = iter(target_loader)
        if step % len(target_loader_unl) == 0 and step > 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len(source_loader) == 0 and step > 0:
            data_iter_s = iter(source_loader)

        data_s = next(data_iter_s)
        im_data_s = data_s[0].cuda(non_blocking=True)
        gt_labels_s = data_s[1].cuda(non_blocking=True)

        data_t = next(data_iter_t)
        im_data_t = data_t[0].cuda(non_blocking=True)
        gt_labels_t = data_t[1].cuda(non_blocking=True)

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        with autocast():
            feats_s = G(im_data_s)
            feats_t = G(im_data_t)

            loss = 0
            cls_loss = 0
            out_s = F1(feats_s)
            cls_loss += criterion(out_s, gt_labels_s)
            out_t = F1(feats_t)
            cls_loss += criterion(out_t, gt_labels_t)
            loss += cls_loss

            data_t_unl = next(data_iter_t_unl)
            if args.cons_wt > 0: # using consistency regularization
                # NOTE : for consistency regularization, the second transform is
                # supposed to be the mellow one which is used for
                # computing pseudo-labels : im_data_tu2
                im_data_tu = data_t_unl[0][0].cuda(non_blocking=True)
                im_data_tu2 = data_t_unl[0][1].cuda(non_blocking=True)

                # compute unlabelled data feats
                feats_unl = G(im_data_tu)

                # NOTE : currently cons_aug_level which is the more intense
                # augmentation is applied on im_data_tu2
                pls = torch.softmax(F1(feats_unl).detach(), dim=1)
                confs, _ = torch.max(pls, dim=1)
                pl_mask = (confs > args.cons_threshold).float()
                loss_cons = (criterion1(F1(G(im_data_tu2)), pls) * pl_mask).mean()
                loss += args.cons_wt * loss_cons

            # NOTE : this is currently not meant to be used with the
            # consistency regularization term above. Would need a small change
            # in im_data_tu below for this.
            if args.vat_tw > 0:
                im_data_tu = data_t_unl[0].cuda(non_blocking=True)
                feats_unl = G(im_data_tu)
                vat_loss_t = vat_loss(
                    args, G, F1, criterion1,
                    torch.cat((im_data_t, im_data_tu), dim=0),
                    torch.softmax(
                        F1(torch.cat((feats_t, feats_unl), dim=0)), dim=1))
                loss += args.vat_tw * vat_loss_t

            # if some entropy based method is used
            if args.ent_method:
                if args.ent_method == 'ENT':
                    ent_loss = entropy(F1, feats_unl, args.lamda)
                elif args.ent_method == 'MME':
                    ent_loss = adentropy(F1, feats_unl, args.lamda)
                loss += args.ent_wt * ent_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer_g)
        scaler.step(optimizer_f)
        scaler.update()

        log_info = OrderedDict({
            'Train Step': step
        })
        log_info.update({
            'Cls Loss': FormattedLogItem(cls_loss.item(), '{:.6f}')
        })
        if args.cons_wt > 0:
            log_info.update({
                'Consistency Loss' : loss_cons.item(),
            })
        if args.ent_method:
            log_info.update({
                # NOTE: this is negative entropy
                'Ent Loss': FormattedLogItem(ent_loss.item(), '{:.6f}')
            })
        if args.vat_tw > 0:
            log_info.update({
                'Target VAT loss': vat_loss_t.item()
            })

        if step % args.log_interval == 0:
            log_str = get_log_str(args, log_info, title='Training Log')
            print(log_str)
            wandb.log(rm_format(log_info))

        if step % args.save_interval == 0 and step > 0:
            # model.eval() done at the start of test()
            test_metrics = test(G, F1, target_loader_test, class_list)
            val_metrics = test(G, F1, target_loader_val, class_list)
            loss_test, acc_test = test_metrics[0], test_metrics[1]
            loss_val, acc_val = val_metrics[0], val_metrics[1]

            G.train()
            F1.train()

            if args.save_model:
                torch.save({
                    'G_state_dict' : G.state_dict(),
                    'F1_state_dict' : F1.state_dict(),
                    'optimizer_g_state_dict' : optimizer_g.state_dict(),
                    'optimizer_f_state_dict' : optimizer_f.state_dict(),
                    'step' : step,
                }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))

            if acc_val > best_acc:
                if args.save_model:
                    shutil.copyfile(
                        os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                        os.path.join(args.save_dir, 'model-best.pth.tar'))
                best_acc = acc_val
                best_acc_test = acc_test

            log_info = OrderedDict({
                'Train Step': step,
                'Best Acc Val': FormattedLogItem(best_acc, '{:.2f}'),
                'Best Acc Test': FormattedLogItem(best_acc_test, '{:.2f}'),
                'Current Acc Val': FormattedLogItem(acc_val, '{:.2f}'),
                # logging since it is computed anyway
                'Current Acc Test': FormattedLogItem(acc_test, '{:.2f}')
            })
            log_str = get_log_str(args, log_info, title='Validation Log')
            print(log_str)
            wandb.log(rm_format(log_info))

if __name__ == '__main__':
    args = parse_args()

    # W&B logging setup
    wandb = WandbWrapper(debug=~args.use_wandb)
    if not args.expt_name:
        args.expt_name = gen_unique_name()
    if args.project == '':
        args.project = 'ssda_mme'
    wandb.init(name=args.expt_name, dir=args.save_dir,
               config=args, reinit=True, project=args.project)

    os.makedirs(args.save_dir, exist_ok=True)
    main(args, wandb)
    wandb.join()

