import torch
import torch.nn as nn
from utils.misc import AverageMeter

def test(G, F1, loader, class_list, conf_mat=False):
    G.eval()
    F1.eval()
    test_loss = AverageMeter()
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = []
    all_feats = []
    all_labels = []
    all_preds = []
    all_confs = []
    criterion = nn.CrossEntropyLoss()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t = data_t[0].cuda()
            gt_labels_t = data_t[1].cuda()
            feat = G(im_data_t)
            output1 = F1(feat)
            feat = torch.nn.functional.normalize(feat)
            all_feats.append(feat.cpu())
            all_labels.append(data_t[1])
            output_all.append(output1)
            size += im_data_t.size(0)
            conf1, pred1 = output1.data.max(1)
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            all_preds.append(pred1.cpu())
            all_confs.append(conf1.cpu())
            test_loss.update(criterion(output1, gt_labels_t), len(loader))

    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)
    all_confs = torch.cat(all_confs)
    all_preds = torch.cat(all_preds)

    if conf_mat:
        return test_loss.avg, 100. * float(correct) / size, \
               all_feats, all_labels, all_preds, all_confs, confusion_matrix
    else:
        return test_loss.avg, 100. * float(correct) / size, \
               all_feats, all_labels, all_preds, all_confs