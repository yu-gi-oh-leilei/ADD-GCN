import os, sys, pdb
from PIL import Image
import random
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .coco import COCO2014
from .cocodetection import CoCoDataset
from .voc import VOC2007, VOC2012
from .voc07 import Voc07Dataset
from .nuswide import NusWideDataset
from .mirflickr25k import MirFlickr25kPreProcessing
from .vg500dataset import VGDataset

data_dict = {'CoCoDataset': CoCoDataset,
            'COCO2014': COCO2014,
            'VOC2007': VOC2007,
            'VOC2012': VOC2012,
            'NusWideDataset': NusWideDataset,
            'mirflickr25k': MirFlickr25kPreProcessing,
            'VG_100K': VGDataset}

def collate_fn(batch):
    ret_batch = dict()
    for k in batch[0].keys():
        if k == 'image' or k == 'target':
            ret_batch[k] = torch.cat([b[k].unsqueeze(0) for b in batch])
        else:
            ret_batch[k] = [b[k] for b in batch]
    # print(ret_batch)
    return ret_batch

class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__


def get_transform(args, is_train=True):
    if is_train:
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(args.image_size, scale=(0.1, 1.5), ratio=(1.0, 1.0)),
            # transforms.RandomResizedCrop(args.image_size, scale=(0.1, 2.0), ratio=(1.0, 1.0)),
            transforms.Resize((args.image_size+64, args.image_size+64)),
            MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def make_data_loader(args, is_train=True):
    root_dir = os.path.join(args.data_root_dir, args.data)

    # Build val_loader
    transform = get_transform(args, is_train=False)
    if args.data == 'COCO2014':
        val_dataset = COCO2014(root_dir, phase='val', transform=transform)
    elif args.data == 'CoCoDataset':
        dataset_dir = os.path.join(args.data_root_dir, 'COCO2014')
        val_dataset = CoCoDataset(
                    image_dir=osp.join(dataset_dir, 'val2014'),
                    anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
                    input_transform=transform,
                    labels_path='data/coco/val_label_vectors_coco14.npy',
                    filename_path='data/coco/val_filename_coco14.npy',
                ) 
    elif args.data in ('VOC2007', 'VOC2012'):
        val_dataset = data_dict[args.data](root_dir, phase='test', transform=transform)
        # dataset_dir = os.path.join(args.data_root_dir, 'VOC2007')
        # val_dataset = Voc07Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
        #                             anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'), 
        #                             transform = transform, 
        #                             labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
        #                             dup=None)
    elif args.data in ('nus_wide', 'nuswide'):
        dataset_dir = os.path.join(args.data_root_dir, 'nuswide')
        img_dir = os.path.join(dataset_dir, 'Flickr')
        anno_path = os.path.join(dataset_dir, 'ImageList', 'TestImagelist.txt')
        labels_path = os.path.join(dataset_dir, 'Groundtruth', 'Labels_Test.txt')
        val_dataset = NusWideDataset(
                    img_dir = img_dir,
                    anno_path = anno_path,
                    labels_path = labels_path,
                    transform = transform)
    elif args.data in ('mirflickr25k'):
        data_dir = os.path.join(args.data_root_dir, 'mirflickr25k')
        val_dataset = MirFlickr25kPreProcessing(
                        data_dir, set='test', target_transform=transform)

    elif args.data in ('vg', 'vg500','VG_100K'):
        vg_root = osp.join(args.data_root_dir, 'VG')
        test_dir = osp.join(vg_root,'VG_100K')
        test_list = osp.join(vg_root,'test_list_500.txt')
        test_label = osp.join(vg_root,'vg_category_500_labels_index.json')
        val_dataset = VGDataset(test_dir, test_list, transform, test_label, phase='Test')

    else:
        raise NotImplementedError('Value error: No matched dataset!')
    num_classes = val_dataset[0]['target'].size(-1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, 
                            collate_fn=collate_fn, drop_last=False)
    
    if not is_train:
        return None, val_loader, num_classes
    
    # Build train_loader
    transform = get_transform(args, is_train=True)
    if args.data == 'COCO2014':
        train_dataset = COCO2014(root_dir, phase='train', transform=transform)
    elif args.data == 'CoCoDataset':
        dataset_dir = os.path.join(args.data_root_dir, 'COCO2014')
        train_dataset = CoCoDataset(
                    image_dir=osp.join(dataset_dir, 'train2014'),
                    anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
                    input_transform=transform,
                    labels_path='data/coco/train_label_vectors_coco14.npy',
                    filename_path='data/coco/train_filename_coco14.npy',
                ) 
    elif args.data in ('VOC2007', 'VOC2012'):
        train_dataset = data_dict[args.data](root_dir, phase='trainval', transform=transform)
        # dataset_dir = os.path.join(args.data_root_dir, 'VOC2007')
        # train_dataset = Voc07Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
        #                             anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
        #                             transform = transform, 
        #                             labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
        #                             dup=None)

    elif args.data in ('nus_wide', 'nuswide'):
        dataset_dir = os.path.join(args.data_root_dir, 'nuswide')
        img_dir = os.path.join(dataset_dir, 'Flickr')
        anno_path = os.path.join(dataset_dir, 'ImageList', 'TrainImagelist.txt')
        labels_path = os.path.join(dataset_dir, 'Groundtruth', 'Labels_Train.txt')
        train_dataset = NusWideDataset(
                    img_dir = img_dir,
                    anno_path = anno_path,
                    labels_path = labels_path,
                    transform = transform)
    elif args.data in ('mirflickr25k'):
        data_dir = os.path.join(args.data_root_dir, 'mirflickr25k')
        train_dataset = MirFlickr25kPreProcessing(
                        data_dir, set='train', target_transform=transform)
    elif args.data in ('vg', 'vg500', 'VG_100K'):
        vg_root = osp.join(args.data_root_dir, 'VG')
        train_dir = osp.join(vg_root,'VG_100K')
        train_list = osp.join(vg_root,'train_list_500.txt')
        train_label = osp.join(vg_root,'vg_category_500_labels_index.json')
        train_dataset = VGDataset(train_dir, train_list, transform, train_label, phase='Train')
    else:
        raise NotImplementedError('Value error: No matched dataset!')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True, 
                            collate_fn=collate_fn, drop_last=True)
    return train_loader, val_loader, num_classes
