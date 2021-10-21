import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))

import torch

from model.basenet import AlexNetBase, VGGBase
from model.resnet import resnet34
from utils.ioutils import parse_args
from utils.proxy_distance import compute_proxy_distance
from utils.return_dataset import return_dataset_no_transform


def main(args):
    source_loader, target_loader, target_loader_unl,\
    class_list = return_dataset_no_transform(args)

    # Training settings
    if args.net == 'resnet34':
        G = resnet34()
    elif args.net == 'alexnet':
        G = AlexNetBase()
    elif args.net == 'vgg':
        G = VGGBase()
    else:
        raise ValueError('Model cannot be recognized.')

    if args.backbone_path:
        if os.path.isfile(args.backbone_path):
            checkpoint = torch.load(args.backbone_path)
            G.load_state_dict(checkpoint['G_state_dict'])
        else:
            raise Exception(
                'Path for backbone {} not found'.format(args.backbone_path))
    G.cuda()
    source_feats = []
    target_feats = []

    with torch.no_grad():
        for i, loader in enumerate(
                [source_loader, target_loader, target_loader_unl]):
            if i==0:
                feat_accumulator = source_feats
            else:
                feat_accumulator = target_feats

            print('Processing loader {}'.format(i))
            with tqdm(total=len(loader)) as pbar:
                for idx, data in enumerate(loader):
                    imgs = data[0].cuda(non_blocking=True)
                    feats = G(imgs)
                    feat_accumulator.append(feats.cpu())
                    pbar.update(1)

    source_feats = torch.cat(source_feats)
    target_feats = torch.cat(target_feats)

    pdist = compute_proxy_distance(source_feats, target_feats)
    print('Proxy distance : {}'.format(pdist))


if __name__ == '__main__':
    args = parse_args()
    main(args)