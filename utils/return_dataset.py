import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

from dataset.dataset import ImageDataset, return_classlist
from utils.misc import get_data
from utils.transforms import TwoCropsTransform, WeakStrongAug
from utils.transforms import get_transforms, ToByteTensor

# constants
FILELIST = 'filelist'
TENSOR = 'tensor'
SMALL_CLASS_SET = np.arange(5)

def return_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.dset_ram:
        data_type = TENSOR
    else:
        data_type = FILELIST

    # NOTE: ToPILImage() should work correctly with byte tensors as inputs too
    prefetch_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), ToByteTensor()])

    if args.cons_wt > 0:
        # NOTE: assuming this is not being used together with moco currently
        if args.cons_aug_level == 0:
            print('Warning: Running with aug_level=0 for consistency')
        target_dataset_unl_transform = WeakStrongAug(
            get_transforms(args, 'train', data_type, aug_level=0),
            get_transforms(args, 'train', data_type, aug_level=args.cons_aug_level)
        )
    else:
        target_dataset_unl_transform = get_transforms(
            args, 'train', data_type, args.aug_level)

    class_list = return_classlist(image_set_file_s)

    print("%d classes in this dataset" % len(class_list))

    if args.batch_size:
        bs = args.batch_size
    elif args.net == 'alexnet': # defaults
        bs = 32
    else:
        bs = 24

    # Prefetch data for validation
    # NOTE: chopping dataset length so less time is spent in
    # prefetching in debug mode
    target_dataset_val = ImageDataset(
        image_set_file_t_val, root=root,
        transform=get_transforms(args, 'test', FILELIST),
        data_type=FILELIST)

    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=args.num_workers, shuffle=False,
                                    drop_last=False)

    print('Prefetching target val data')
    target_dataset_val = TensorDataset(*get_data(target_loader_val))
    target_loader_val = torch.utils.data.DataLoader(
        target_dataset_val,
        batch_size=min(bs, len(target_dataset_val)),
        num_workers=0, shuffle=False,
        drop_last=False, pin_memory=True)

    if args.dset_ram:
        target_dataset_unl = ImageDataset(
            image_set_file_unl, root=root,
            transform=prefetch_transform,
            data_type=FILELIST)
        target_loader_unl = torch.utils.data.DataLoader(
            target_dataset_unl, batch_size=bs * 2,
            num_workers=args.num_workers, shuffle=False,
            drop_last=False)

        source_dataset = ImageDataset(
            image_set_file_s, root=root,
            transform=prefetch_transform,
            data_type=FILELIST)

        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=bs * 2,
            num_workers=args.num_workers, shuffle=False,
            drop_last=False)

        target_dataset = ImageDataset(
            image_set_file_t, root=root,
            transform=prefetch_transform,
            data_type=FILELIST)
        target_loader = torch.utils.data.DataLoader(
            target_dataset, batch_size=bs * 2,
            num_workers=args.num_workers, shuffle=False,
            drop_last=False)

        print('Prefetching target unl data')
        target_unl_imgs, target_unl_labels = get_data(target_loader_unl)
        print('Prefetching target labelled data')
        target_imgs, target_labels = get_data(target_loader)
        print('Prefetching source data')
        source_imgs, source_labels = get_data(source_loader)

        # Final datasets
        target_dataset_test = ImageDataset(
            (target_unl_imgs, target_unl_labels),
            transform=get_transforms(args, 'test', TENSOR),
            data_type=TENSOR)
        target_loader_test = torch.utils.data.DataLoader(
            target_dataset_test,
            batch_size=bs * 2,
            num_workers=0, shuffle=False,
            drop_last=False, pin_memory=True)

        target_dataset_unl = ImageDataset(
            (target_unl_imgs, target_unl_labels),
            transform=target_dataset_unl_transform, data_type=TENSOR,
            rot=False)
        target_loader_unl = torch.utils.data.DataLoader(
            target_dataset_unl, batch_size=bs * 2, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)

        target_dataset = ImageDataset(
            (target_imgs, target_labels),
            transform=get_transforms(args, 'train', TENSOR, args.aug_level),
            rot=False, data_type=TENSOR)
        if len(target_dataset) <= bs:
            target_loader = torch.utils.data.DataLoader(
                target_dataset, batch_size=bs, num_workers=args.num_workers,
                shuffle=False, drop_last=False, pin_memory=True)
        else:
            target_loader = torch.utils.data.DataLoader(
                target_dataset, batch_size=bs, num_workers=args.num_workers,
                shuffle=True, drop_last=True, pin_memory=True)

        source_dataset = ImageDataset(
            (source_imgs, source_labels),
            transform=get_transforms(args, 'train', TENSOR, args.aug_level),
            rot=False, data_type=TENSOR)
        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=bs, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)
    else:
        source_dataset = ImageDataset(
            image_set_file_s, root=root,
            transform=get_transforms(args, 'train', FILELIST, args.aug_level),
            rot=False, data_type=FILELIST)
        target_dataset = ImageDataset(
            image_set_file_t, root=root,
            transform=get_transforms(args, 'train', FILELIST, args.aug_level),
            rot=False, data_type=FILELIST)
        target_dataset_unl = ImageDataset(
            image_set_file_unl, root=root,
            transform=target_dataset_unl_transform,
            rot=False, data_type=FILELIST)
        target_dataset_test = ImageDataset(
            image_set_file_unl, root=root,
            transform=get_transforms(args, 'test', FILELIST),
            data_type=FILELIST)

        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=bs, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)
        target_loader = torch.utils.data.DataLoader(
            target_dataset, batch_size=min(bs, len(target_dataset)),
            num_workers=args.num_workers, shuffle=True,
            drop_last=True, pin_memory=True)
        target_loader_unl = torch.utils.data.DataLoader(
            target_dataset_unl, batch_size=bs * 2, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)
        target_loader_test = torch.utils.data.DataLoader(
            target_dataset_test, batch_size=bs * 2, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list


def return_dataset_pretrain(args, pt_type='rot'):
    """
    pt_type is a string indicating type of pretraining method to use:
    'rot' : Rotation prediction
    'moco' : Momentum Contrast
    """
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    # For pretraining, treating both domains equally
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.target + '.txt')
    class_list = return_classlist(image_set_file_s)

    prefetch_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), ToByteTensor()])
    if args.dset_ram:
        data_type = TENSOR
    else:
        data_type = FILELIST
    transform = get_transforms(args, 'train', data_type, args.aug_level)
    if pt_type == 'moco':
        transform = TwoCropsTransform(transform)

    assert args.batch_size is not None, \
        'Provide batch sizes. No defaults configured for pretraining.'

    bs = args.batch_size

    if args.dset_ram:
        source_dataset = ImageDataset(
            image_set_file_s, root=root,
            transform=prefetch_transform,
            data_type=FILELIST)
        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=bs * 2,
            num_workers=args.num_workers, shuffle=False,
            drop_last=False)

        target_dataset = ImageDataset(
            image_set_file_t, root=root,
            transform=prefetch_transform,
            data_type=FILELIST)
        target_loader = torch.utils.data.DataLoader(
            target_dataset, batch_size=bs * 2,
            num_workers=args.num_workers, shuffle=False,
            drop_last=False)

        print('Prefetching source data')
        source_imgs, source_labels = get_data(source_loader)
        print('Prefetching target data')
        target_imgs, target_labels = get_data(target_loader)

        source_dataset = ImageDataset(
            (source_imgs, source_labels), transform=transform,
            rot=(pt_type=='rot'), data_type=TENSOR)
        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=bs, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)

        target_dataset = ImageDataset(
            (target_imgs, target_labels), transform=transform,
            rot=(pt_type=='rot'), data_type=TENSOR)
        target_loader = torch.utils.data.DataLoader(
            target_dataset, batch_size=bs, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)
    else:
        source_dataset = ImageDataset(
            image_set_file_s, root=root, transform=transform,
            rot=(pt_type=='rot'), data_type=FILELIST)
        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=bs, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)

        target_dataset = ImageDataset(
            image_set_file_s, root=root, transform=transform,
            rot=(pt_type=='rot'), data_type=FILELIST)
        target_loader = torch.utils.data.DataLoader(
            target_dataset, batch_size=bs, num_workers=args.num_workers,
            shuffle=True, drop_last=True, pin_memory=True)

    return source_loader, target_loader, class_list

def return_dataset_no_transform(args):
    """
    Fow evaluation purposes. Note that this does not include the option of
    prefetching dataset to RAM at the moment
    """
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    data_transform = get_transforms(args, 'test')


    source_dataset = ImageDataset(image_set_file_s, root=root,
                                  transform=data_transform,
                                  rot=False)
    target_dataset = ImageDataset(image_set_file_t, root=root,
                                  transform=data_transform,
                                  rot=False)
    target_dataset_unl = ImageDataset(image_set_file_unl, root=root,
                                      transform=data_transform,
                                      rot=False)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.batch_size:
        bs = args.batch_size
    elif args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=args.num_workers, shuffle=False,
                                                drop_last=False, pin_memory=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    drop_last=False, pin_memory=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=args.num_workers,
                                    shuffle=False,
                                    drop_last=False, pin_memory=True)

    return source_loader, target_loader, target_loader_unl, class_list