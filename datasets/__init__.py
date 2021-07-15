# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from datasets import cityscapes, syntmag
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader
from random import random
import torch

def default_collate(batch):
    """
    Override `default_collate` https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader

    :param batch: list of tuples (img, mask, label)
    :return: 3 elements: tensor data, list of tensors of masks, tensor of labels.
    """        
    x_min = min([batch[0][0].shape[1], batch[1][0].shape[1]])
    y_min = min([batch[0][0].shape[2], batch[1][0].shape[2]])
    
    data, mask, edge, names = [], [], [], []
    
    for i in batch:
        ix_min = int((i[0].shape[1]-x_min)*random())
        iy_min = int((i[0].shape[2]-y_min)*random())
        
        data.append(i[0][:,ix_min:x_min+ix_min,iy_min:y_min+iy_min])
        mask.append(i[1][ix_min:x_min+ix_min,iy_min:y_min+iy_min])
        edge.append(i[2][:,ix_min:x_min+ix_min,iy_min:y_min+iy_min])
        names.append(i[3])
    
    data = torch.stack(data)
    mask = torch.stack(mask)
    edge = torch.stack(edge)
    
    return data, mask, edge, names

def setup_loaders(args):
    '''
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    Function to prepare the dataset to train the model
    input : args - the arguments of the train python file
    output : train_loader - the training dataset,
             val_loader - the validation dataset,    
             train_set - ??? tbd
    '''
    if args.dataset == 'cityscapes':
        args.dataset_cls = cityscapes
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
    elif args.dataset == 'syntmag':
        args.dataset_cls = syntmag
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
            #normalization de l'image par la médiane et l'écart-type
        mean_std = ([0.157, 0.157, 0.157], [0.106, 0.106, 0.106])    
    else:
        raise
     
    args.num_workers = 4 * args.ngpu
    if args.test_mode:
        args.num_workers = 0 #1

    ### Transformations géométrique de l'image et du masque
    # Pour le jeu de données cityscape
    if args.dataset =='cityscape':
        train_joint_transform_list = [
            joint_transforms.RandomSizeAndCrop(args.crop_size,
                                               False,
                                               pre_size=args.pre_size,
                                               scale_min=args.scale_min,
                                               scale_max=args.scale_max,
                                               ignore_index=args.dataset_cls.ignore_label),
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip()]
    
    # Dans le cas de notre jeu de données
    elif args.dataset =='syntmag':
        train_joint_transform_list = [
            joint_transforms.RandomResize(args.resize),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomVerticallyFlip()]
    
    #Rotation
    if args.rotate != None:
        train_joint_transform_list += [joint_transforms.RandomRotate(args.rotate)]
        
    #Cisaillement
    if args.shear != None:
        train_joint_transform_list += [joint_transforms.RandomShear(args.shear)]
    
    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    ### Transformation de l'image d'entrée
    # Ajout de bruit gaussien
    train_input_transform = [extended_transforms.RandomNoise()]
    
    #Augmentation des couleurs (cityscape)
    if args.color_aug:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=args.color_aug,
            contrast=args.color_aug,
            saturation=args.color_aug,
            hue=args.color_aug)]

    # Lissage des images
    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
    else:
        pass
    
    # Lissage automatique des images synthétiques
    if args.dataset == 'syntmag':
        train_input_transform += [extended_transforms.PureGaussianBlur(),
                                  extended_transforms.ToRGB()]
        #extended_transforms.RandomNoise(),

    train_input_transform += [standard_transforms.ToTensor(), #changer image PIL H*X*C à torch C*H*W [0. -1.]
                              standard_transforms.Normalize(*mean_std)] #normalisation par la médiane et l'ecart-type
    
    train_input_transform = standard_transforms.Compose(train_input_transform)
    
    val_input_transform = []
    
    if args.dataset == 'syntmag':
        val_input_transform += [extended_transforms.ToRGB()]
        
    val_input_transform += [standard_transforms.ToTensor(),
                            standard_transforms.Normalize(*mean_std)]
    
    val_input_transform = standard_transforms.Compose(val_input_transform)

    target_transform = extended_transforms.MaskToTensor()
    
    target_train_transform = extended_transforms.MaskToTensor()

    if args.dataset == 'cityscapes':
        city_mode = 'train' 
        city_quality = 'fine'
        train_set = args.dataset_cls.CityScapes(
            city_quality, city_mode, 0, 
            joint_transform=train_joint_transform,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            cv_split=args.cv)
        val_set = args.dataset_cls.CityScapes('fine', 'val', 0, 
                                              transform=val_input_transform,
                                              target_transform=target_transform,
                                              cv_split=args.cv)

    elif args.dataset == 'syntmag':
        city_mode = 'train' 
        train_set = args.dataset_cls.Syntmag(
            city_mode, 0, 
            joint_transform=train_joint_transform,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            cv_split=args.cv)
        val_set = args.dataset_cls.Syntmag('val', 0, 
                                           transform=val_input_transform,
                                           target_transform=target_transform,
                                           cv_split=args.cv)
        
    else:
        raise
    
    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              shuffle=(train_sampler is None),
                              drop_last=True,
                              sampler = train_sampler,
                              collate_fn = default_collate)
    
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , 
                            shuffle=False, 
                            drop_last=False, 
                            sampler = val_sampler,
                            collate_fn = default_collate)

    return train_loader, val_loader, train_set

