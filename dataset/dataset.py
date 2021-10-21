import os
import os.path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# constants
FILELIST = 'filelist'
TENSOR = 'tensor'
T2I = transforms.ToPILImage()
I2T = transforms.ToTensor()

def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class ImageDataset(Dataset):
    def __init__(self, image_list, root="./data/multi/", transform=None,
                 target_transform=None, rot=False, data_type=FILELIST):
        """
        image_list : filelist or a list of tensors containing
                     all images, all_labels
        rot : boolean flag on whether to return samples with different rotations
        """
        super(ImageDataset, self).__init__()
        assert data_type == TENSOR or data_type == FILELIST, \
            'ImageDataset.__init__: data_type can be one of tensor, filelist'

        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform
        self.rot = rot
        self.root = root # ignored when data_type is tensor

        if self.data_type == FILELIST:
            self.imgs, self.labels = self.make_dataset(image_list)
        else:
            self.imgs, self.labels = image_list

        # Used for pseudo-labelling
        self.pseudo = False # use pseudo labels
        self.pseudo_labels = -1 * torch.ones(self.labels.shape).int()

    def load_img(self, image_name):
        path = os.path.join(self.root, image_name)
        return Image.open(path).convert('RGB')

    def get_img(self, idx):
        if self.data_type == FILELIST:
            return self.load_img(self.imgs[idx])
        else:
            # returns the image from the tensor
            return self.imgs[idx]

    def make_dataset(self, image_list):
        with open(image_list) as f:
            image_names = [x.split(' ')[0] for x in f.readlines()]
        with open(image_list) as f:
            label_list = []
            selected_list = []
            for ind, x in enumerate(f.readlines()):
                label = x.split(' ')[1].strip()
                label_list.append(int(label))
                selected_list.append(ind)
            image_names = np.array(image_names)
            label_list = np.array(label_list)
        image_names = image_names[selected_list]

        return image_names, label_list


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        if self.pseudo:
            target = self.pseudo_labels[index]
        else:
            target = self.labels[index]
        img = self.get_img(index)

        assert self.transform is not None, 'No transform used, cannot ' \
                                           'convert image to tensor'

        if self.rot:
            if self.data_type == TENSOR:
                all_rotated_imgs = [
                    self.transform(I2T(TF.rotate(T2I(img), -90))),
                    self.transform(img),
                    self.transform(I2T(TF.rotate(T2I(img), 90))),
                    self.transform(I2T(TF.rotate(T2I(img), 180)))]
                all_rotated_imgs = torch.stack(all_rotated_imgs, dim=0)
            else:
                all_rotated_imgs = [
                    self.transform(TF.rotate(img, -90)),
                    self.transform(img),
                    self.transform(TF.rotate(img, 90)),
                    self.transform(TF.rotate(img, 180))]
                all_rotated_imgs = torch.stack(all_rotated_imgs, dim=0)
        else:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.rot:
            target = torch.tensor(len(all_rotated_imgs) * [target])
            rot_target = torch.LongTensor([0, 1, 2, 3])
            return all_rotated_imgs, target, rot_target, index
        else:
            return img, target, index

    def __len__(self):
        return len(self.imgs)

    def chop_class(self, classes):
        mask = np.isin(self.labels, classes)
        self.imgs = self.imgs[mask]
        self.pseudo_labels = self.pseudo_labels[mask]
        self.labels = self.labels[mask]


