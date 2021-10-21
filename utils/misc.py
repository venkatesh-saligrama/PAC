import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='avg_meter_var', fmt=':f'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_data(loader):
    all_imgs = []
    all_labels = []
    for i, data in enumerate(loader):
        all_imgs.append(data[0])
        all_labels.append(data[1])
    all_imgs = torch.cat(all_imgs)
    all_labels = torch.cat(all_labels)
    return all_imgs, all_labels