# Multiple routined related to RandAugment borrowed from
# https://github.com/ildoonet/pytorch-randaugment

import random

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
from PIL import Image
from torchvision import transforms

TENSOR = 'tensor'
FILELIST = 'filelist'
PARAMETER_MAX = 10

class TwoCropsTransform:
    """Take two random crops of one image as the query and key for MoCo."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        img1 = self.base_transform(x)
        img2 = self.base_transform(x)
        return [img1, img2]

class WeakStrongAug:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        img1 = self.base_transform1(x)
        img2 = self.base_transform2(x)
        return [img1, img2]


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs

def rot_pt_augment_pool():
    # Reduced rotation and sheer augmentations for rotation pretraining
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 5, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.05, 0),
            (ShearY, 0.05, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.05, 0),
            (TranslateY, 0.05, 0)]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m, augment_pool):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = augment_pool

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img


def get_transforms(args, split, data_type='filelist', aug_level=0):
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    assert data_type == TENSOR or data_type == FILELIST, \
        'transforms.get_transforms: data_type can be one of tensor, filelist'
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if split == 'test':
        transform_list = [
            transforms.CenterCrop(crop_size),
        ]
    elif split == 'train':
        if aug_level == 0:
            # Mellow transform used for pseudo-labeling during consistency
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
            ]
        elif aug_level == 1:
            # Color Jittering
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
            ]
        elif aug_level == 2:
            # Randaugment
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                RandAugmentMC(n=2, m=10, augment_pool=fixmatch_augment_pool())
            ]
        elif aug_level == 3:
            # Color jittering + Rand augment
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                RandAugmentMC(n=2, m=10, augment_pool=fixmatch_augment_pool()),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8)
            ]
        elif aug_level == 4:
            # lower rotation and sheer augmentations for rotation prediction
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                RandAugmentMC(n=2, m=10, augment_pool=rot_pt_augment_pool()),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8)
            ]
        else:
            raise Exception('get_transforms : augmentation not recognized')
    else:
        raise Exception('get_transforms: split not recognized')

    if data_type == FILELIST:
        # add resize
        transform_list = [transforms.Resize((256, 256))] + transform_list
    else:
        # convert to PIL Image
        transform_list = [transforms.ToPILImage()] + transform_list

    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)
    transform = transforms.Compose(transform_list)

    return transform

class ToByteTensor(object):
    """
    Convert a 0-1 FloatTensor image to ByteTensor
    """
    def __call__(self, pic):
        return pic.mul(255).byte()

    def __repr__(self):
        return self.__class__.__name__ + '()'