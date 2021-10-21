import os
import sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn
from tqdm import tqdm

from model.basenet import AlexNetBase, VGGBase
from model.resnet import resnet34
from utils.ioutils import parse_args
from utils.return_dataset import return_dataset_no_transform
from utils.misc import AverageMeter

def get_support_feats(G, loader_lbl):
    support_feats = []
    print('Collecting support features...')
    with tqdm(total=len(loader_lbl)) as pbar, torch.no_grad():
        for data in loader_lbl:
            images = data[0].cuda(non_blocking=True)
            feats = G(images)
            if args.normalize:
                feats = nn.functional.normalize(feats)
            support_feats.append(feats)
            pbar.update(1)

    support_feats = torch.cat(support_feats, dim=0)
    return support_feats

def get_acc(G, support_feats, loader_lbl, loader_unl, class_list):
    acc = AverageMeter()
    print('Computing accuracy...')
    with tqdm(total=len(loader_unl)) as pbar, torch.no_grad():
        for data in loader_unl:
            images = data[0].cuda(non_blocking=True)
            labels = data[1].cuda(non_blocking=True)
            feats = G(images)
            if args.normalize:
                feats = nn.functional.normalize(feats)

            dists = torch.cdist(feats, support_feats)
            mean_dists = []
            for cl in range(len(class_list)):
                curr_cl_idxs = torch.arange(len(support_feats))[loader_lbl.dataset.labels == cl]
                mean_dists.append(dists[:, curr_cl_idxs].mean(dim=1, keepdim=True))
            mean_dists = torch.cat(mean_dists, dim=1)
            preds = mean_dists.argmin(dim=1)

            acc.update(((preds == labels).sum()) / float(len(preds)), len(preds))
            pbar.update(1)
    return acc.avg

def main(args):
    source_loader, target_loader, target_loader_unl, class_list = return_dataset_no_transform(args)
    torch.manual_seed(args.seed)

    if args.net == 'resnet34':
        G = resnet34(pretrained=args.pt)
    elif args.net == 'alexnet':
        G = AlexNetBase(pret=args.pt)
    elif args.net == 'vgg':
        G = VGGBase(pret=args.pt)
    else:
        raise ValueError('Model cannot be recognized.')
    G.cuda()
    G.eval()

    if args.backbone_path:
        if os.path.isfile(args.backbone_path):
            checkpoint = torch.load(args.backbone_path)
            G.load_state_dict(checkpoint['G_state_dict'])
        else:
            raise Exception(
                'Path for backbone {} not found'.format(args.backbone_path))

    support_feats = get_support_feats(G, source_loader)
    source_nn_acc = get_acc(
        G, support_feats, source_loader, target_loader_unl, class_list)
    support_feats = get_support_feats(G, target_loader)
    target_nn_acc = get_acc(
        G, support_feats, target_loader, target_loader_unl, class_list)
    print('Source NN Accuracy : {:.2f}'.format(100. * source_nn_acc))
    print('Target NN Accuracy : {:.2f}'.format(100. * target_nn_acc))

    return source_nn_acc, target_nn_acc

if __name__ == '__main__':
    args = parse_args()
    main(args)