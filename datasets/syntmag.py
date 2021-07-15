# -*- coding: utf-8 -*-
"""

"""

import os   
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import datasets.syntmag_labels as syntmag_labels
from config import cfg
import logging
import datasets.edge_utils as edge_utils

trainid_to_name = syntmag_labels.trainId2name
id_to_trainid = syntmag_labels.label2trainid
num_classes = 3
ignore_label = 255 #pour avoir une classe ignore label
root = '/content/gdrive/My Drive/' # cfg.DATASET.CITYSCAPES_DIR
DATASET_CV_SPLITS = 3
splits = 3

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    return new_mask

def add_items(items, aug_items, cities, img_path, mask_path, mask_postfix, mode, maxSkip):

    for c in cities:
        c_items = [name.split('_img.png')[0] for name in
                   os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_img.png'),
                        os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
            
def make_cv_splits(img_dir_name):
    '''
    Create splits of train/val data.
    A split is a lists of directories.
    split0 is aligned with the default Cityscapes train/val.
    - code pillé de cityscape
    '''
    trn_path = os.path.join(root, img_dir_name, 'image', 'train')
    val_path = os.path.join(root, img_dir_name, 'image', 'val')


    trn_cities = ['train/' + c for c in os.listdir(trn_path)]
    val_cities = ['val/' + c for c in os.listdir(val_path)]

    # want reproducible randomly shuffled
    trn_cities = sorted(trn_cities)

    all_cities = val_cities + trn_cities
    num_val_cities = len(val_cities)
    num_cities = len(all_cities)

    cv_splits = []
    for split_idx in range(splits):
        split = {}
        split['train'] = []
        split['val'] = []
        offset = split_idx * num_cities // splits
        for j in range(num_cities):
            if j >= offset and j < (offset + num_val_cities):
                split['val'].append(all_cities[j])
            else:
                split['train'].append(all_cities[j])
        cv_splits.append(split)

    return cv_splits

def make_split_coarse(img_path):
    '''
    Create a train/val split for coarse
    return: city split in train
    '''
    all_cities = os.listdir(img_path)
    all_cities = sorted(all_cities)  # needs to always be the same
    val_cities = [] # Can manually set cities to not be included into train split

    split = {}
    split['val'] = val_cities
    split['train'] = [c for c in all_cities if c not in val_cities]
    return split

def make_test_split(img_dir_name):
    test_path = os.path.join(root, img_dir_name, 'image', 'test')
    test_cities = ['test/' + c for c in os.listdir(test_path)]

    return test_cities

def make_dataset(mode, maxSkip=0, fine_coarse_mult=6, cv_split=0):
    '''
    Assemble list of images + mask files

    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na

    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    '''
    items = []
    aug_items = []

    assert mode in ['train', 'val', 'test', 'trainval']
    img_dir_name = 'synt_mag' #'leftImg8bit_trainvaltest'
    img_path = os.path.join(root, img_dir_name, 'image')
    mask_path = os.path.join(root, img_dir_name,'mask')
    mask_postfix = '_seg.png'
    cv_splits = make_cv_splits(img_dir_name)
    if mode == 'trainval':
        modes = ['train', 'val']
    else:
        modes = [mode]
    for mode in modes:
        if mode == 'test':
            cv_splits = make_test_split(img_dir_name)
            add_items(items, cv_splits, img_path, mask_path,
                  mask_postfix)
        else:
            logging.info('{} subdic: '.format(mode) + str(cv_splits[cv_split][mode]))

            add_items(items, aug_items, cv_splits[cv_split][mode], img_path, mask_path,
                  mask_postfix, mode, maxSkip)
        
    logging.info('Syntmag-{}: {} images'.format(mode, len(items)+len(aug_items)))
    
    return items, aug_items


class Syntmag(data.Dataset):

    def __init__(self, mode, maxSkip=0, joint_transform=None, sliding_crop=None,
                 transform=None, target_transform=None, dump_images=False,
                 cv_split=None, eval_mode=False, 
                 eval_scales=None, eval_flip=False):
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        if eval_scales != None:
            self.eval_scales = [float(scale) for scale in eval_scales.split(",")]

        if cv_split:
            self.cv_split = cv_split
            assert cv_split < DATASET_CV_SPLITS, \
                'expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, DATASET_CV_SPLITS)
        else:
            self.cv_split = 0
        self.imgs, _ = make_dataset(mode, self.maxSkip, cv_split=self.cv_split)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _eval_get_item(self, img, mask, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool)+1):
            imgs = []
            if flip :
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w,h = img.size
                target_w, target_h = int(w * scale), int(h * scale) 
                resize_img =img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(tensor_img)
            return_imgs.append(imgs)
        return return_imgs, mask
        
    def __getitem__(self, index):

        #préparer les chemins
        img_path, mask_path = self.imgs[index]
        
        #obtenir le nom de l'image
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        #charger les images
        img, mask = Image.open(img_path), Image.open(mask_path)
        
        #changer les id_training
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in id_to_trainid.items():
            mask_copy[mask == k] = v
        
        if self.eval_mode:
            return self._eval_get_item(img, mask, self.eval_scales, self.eval_flip), img_name
        
        mask = Image.fromarray(mask_copy.astype(np.uint8))
        
        # Image Transformations
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            
        #get edge maps
        _edgemap = mask.numpy()
        _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
        edgemap = torch.from_numpy(_edgemap).float()
            
        #Debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)
    
        return img, mask, edgemap, img_name

    def __len__(self):
        return len(self.imgs)

