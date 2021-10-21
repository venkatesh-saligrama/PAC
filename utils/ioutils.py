import argparse
import logging
import math
import os
import random
import string
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(description='SSDA Classification')
    parser.add_argument('--expt_name', type=str, default='',
                        help='Name of the experiment for wandb')
    parser.add_argument('--steps', type=int, default=50000, metavar='N',
                        help='maximum number of iterations '
                             'to train (default: 50000)')
    parser.add_argument('--T', type=float, default=0.05, metavar='T',
                        help='temperature (default: 0.05)')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay used by optimizer')

    parser.add_argument('--backbone_path', type=str, default='',
                        help='Path to checkpoint to load backbone from')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size of source labelled data to use. If not '
                             'provided, certain default sizes used')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='Number of worker threads to use in each dataloader')
    parser.add_argument('--max_num_threads', type=int, default=12,
                        help='Maximum number of threads that the process should '
                             'use. Uses torch.set_num_threads()')
    parser.add_argument('--dset_ram', type=boolfromstr, default=False,
                        help='Whether to prefetch all image data onto RAM')

    parser.add_argument('--project', type=str, default='',
                        help='wandb project to use')
    parser.add_argument('--save_dir', type=str, default='expts/tmp_last',
                        help='dir to save experiment results to')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='If not set, model will not be saved')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before saving a model')
    parser.add_argument('--ckpt_freq', type=int, default=4, metavar='N',
                        help='Checkpointing frequency in number of save '
                             'intervals. These checkpoints would be saved '
                             'as checkpoint_n.pth.tar where n is the step.')

    parser.add_argument('--net', type=str, default='resnet34',
                        choices=['alexnet', 'vgg', 'resnet34'],
                        help='which network to use')
    parser.add_argument('--source', type=str, default='real',
                        help='source domain')
    parser.add_argument('--target', type=str, default='sketch',
                        help='target domain')
    parser.add_argument('--dataset', type=str, default='multi',
                        choices=['multi', 'office', 'office_home', 'visda17'],
                        help='the name of dataset')
    parser.add_argument('--num', type=int, default=3,
                        help='number of labeled examples in the target')

    parser.add_argument('--aug_level', type=int, default=3,
                        help='Level of augmentation to apply to data. Currently:'
                             '0 : Resize, randomcrop, random flip'
                             '1 : 0 + color jitter '
                             '2 : 0 + Randaugment'
                             '3 : Randaugment + Color jittering'
                             '4 : 3 with lower rotation and sheer')

    parser.add_argument('--fs_ss', action='store_true', default=False,
                        help='Whether to use file_system as '
                             'torch.mutiprocessing sharing strategy')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint path to resume from')
    # dev option
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Debug mode turns off wandb logging')

    # classifier options
    parser.add_argument('--cls_layers', type=str, default='',
                        help='Number of nodes in different layers of '
                             'the classifier (\',\' delimited string)')
    parser.add_argument('--cls_normalize', type=boolfromstr, default=True,
                        help='Normalize the features in the classifier')
    parser.add_argument('--cls_bias', type=boolfromstr, default=False,
                        help='Whether to add bias parameter in the final '
                             'layer classifier')

    # CONSISTENCY REGULARIZATION OPTIONS
    parser.add_argument('--cons_wt', type=float, default=1.,
                        help='Weight of term for consistency regularization.')
    parser.add_argument('--cons_threshold', type=float, default=0.9,
                        help='Threshold to use for consistency pseudo-labels')
    parser.add_argument('--cons_aug_level', type=int, default=3,
                        help='Augmentation level for consistency regularization. '
                             'Same options as aug_level')

    # VIRTUAL ADVERSARIAL TRAINING OPTIONS
    parser.add_argument('--vat_tw', type=float, default=0.01,
                        help='Weight of target domain VAT loss')
    parser.add_argument('--vat_radius', type=float, default=3.5,
                        help='Weight of source domain VAT loss')

    # OPTIONS FOR ENTROPY MINIMIZATION
    parser.add_argument('--ent_method', type=str, default='',
                        choices=['', 'ENT', 'MME'],
                        help='MME from Saito et. al., ENT is entropy minimization')
    parser.add_argument('--ent_wt', type=float, default=1.,
                        help='Weight of the entropy term in final objective')

    # PRETRAINING : ADDITIONAL MOCO OPTIONS
    parser.add_argument('--queue_len', type=int, default=4096,
                        help='Length of queue used by MoCo')
    parser.add_argument('--momentum', type=float, default=0.999,
                        help='Momentum for MoCo')

    return parser

def post_process_args(args):
    if args.cls_layers:
        args.cls_layers = [int(num) for num in args.cls_layers.split(',')]
    else:
        args.cls_layers = []
    return args

def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    args = post_process_args(args)
    return args

def get_default_args():
    parser = get_parser()
    args = parser.parse_args([])
    args = post_process_args(args)
    return args

def boolfromstr(s):
    if s.lower().startswith('true'):
        return True
    elif s.lower().startswith('false'):
        return False
    else:
        raise Exception('Incorrect option passed for a boolean')

class FormattedLogItem:
    def __init__(self, item, fmt):
        self.item = item
        self.fmt = fmt
    def __str__(self):
        return self.fmt.format(self.item)

def rm_format(dict_obj):
    ret = dict_obj
    for key in ret:
        if isinstance(ret[key], FormattedLogItem):
            ret[key] = ret[key].item
    return ret

def get_log_str(args, log_info, title='Expt Log', sep_ch='-'):
    now = str(datetime.now().strftime("%H:%M %d-%m-%Y"))
    log_str = (sep_ch * math.ceil((80 - len(title))/2.) + title
               + (sep_ch * ((80 - len(title))//2)) + '\n')
    log_str += '{:<25} : {}\n'.format('Time', now)
    for key in log_info:
        log_str += '{:<25} : {}\n'.format(key, log_info[key])
    log_str += sep_ch * 80
    return log_str

def write_to_log(args, log_str, mode='a+'):
    with open(os.path.join(args.save_dir, 'log.txt'), mode) as outfile:
        print(log_str, file=outfile)

def get_logger(args):
    log_config = {
        'level': logging.INFO,
        'format': '{asctime:s} {levelname:<8s} {filename:<12s} : {message:s}',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'filename': os.path.join(args.save_dir, 'events.log'),
        'filemode': 'w',
        'style': '{'}
    logging.basicConfig(**log_config)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    cfmt = logging.Formatter('{asctime:s} : {message:s}',
                             style='{', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(cfmt)
    logger = logging.getLogger(__name__)
    logger.addHandler(console)

    return logger

def gen_unique_name(length=4):
    """
    Returns a string of 'length' lowercase letters
    """
    return ''.join([random.choice(
        string.ascii_lowercase) for i in range(length)])

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class WandbWrapper():
    def __init__(self, debug=False):
        self.debug = debug
        if debug:
            print('Wandb Wrapper : debug mode. No logging with wandb')

    def init(self, *args, **kwargs):
        if not self.debug:
            import wandb
            wandb.init(*args, **kwargs)
            self.run = wandb.run
        else:
            self.run = AttrDict({'dir' : kwargs['dir']})

    def log(self, *args, **kwargs):
        if not self.debug:
            import wandb
            wandb.log(*args, **kwargs)

    def join(self, *args, **kwargs):
        if not self.debug:
            import wandb
            wandb.join(*args, **kwargs)