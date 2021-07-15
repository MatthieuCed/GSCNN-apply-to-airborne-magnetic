# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Code borrowded from:
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/transforms.py
#
#
# MIT License
#
# Copyright (c) 2017 ZijunDeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""

import random
import numpy as np
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import torch
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as torch_tr
from scipy import ndimage
from config import cfg
from scipy.ndimage.interpolation import shift
from scipy.misc import imsave #import imageio.imwrite as imsave #scipy.misc.imsave change to imageio.imwrite
from skimage.segmentation import find_boundaries

class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class RelaxedBoundaryLossToTensor(object):
    def __init__(self,ignore_id, num_classes):
        self.ignore_id=ignore_id
        self.num_classes= num_classes


    def new_one_hot_converter(self,a):
        ncols = self.num_classes+1
        out = np.zeros( (a.size,ncols), dtype=np.uint8)
        out[np.arange(a.size),a.ravel()] = 1
        out.shape = a.shape + (ncols,)
        return out

    def __call__(self,img):
        img_arr = np.array(img)
        import scipy; scipy.misc.imsave('orig.png',img_arr) #imsave('orig.png',img_arr)

        img_arr[img_arr==self.ignore_id]=self.num_classes       
        
        if cfg.STRICTBORDERCLASS != None:
            one_hot_orig = self.new_one_hot_converter(img_arr)
            mask = np.zeros((img_arr.shape[0],img_arr.shape[1]))
            for cls in cfg.STRICTBORDERCLASS:
                mask = np.logical_or(mask,(img_arr == cls))
        one_hot = 0

        #print(cfg.EPOCH, "Non Reduced", cfg.TRAIN.REDUCE_RELAXEDITERATIONCOUNT)
        border = cfg.BORDER_WINDOW
        if (cfg.REDUCE_BORDER_EPOCH !=-1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            border = border // 2
            border_prediction = find_boundaries(img_arr, mode='thick').astype(np.uint8)
            print(cfg.EPOCH, "Reduced")
        
        for i in range(-border,border+1):
            for j in range(-border, border+1):
                shifted= shift(img_arr,(i,j), cval=self.num_classes)
                one_hot += self.new_one_hot_converter(shifted)       
        
        one_hot[one_hot>1] = 1
        
        if cfg.STRICTBORDERCLASS != None:
            one_hot = np.where(np.expand_dims(mask,2), one_hot_orig, one_hot)
    
        one_hot = np.moveaxis(one_hot,-1,0)
    

        if (cfg.REDUCE_BORDER_EPOCH !=-1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
                one_hot = np.where(border_prediction,2*one_hot,1*one_hot)
                print(one_hot.shape)
        return torch.from_numpy(one_hot).byte()
        #return torch.from_numpy(one_hot).float()
        exit(0)

class ResizeHeight(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.target_h = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        target_w = int(w / h * self.target_h)
        return img.resize((target_w, self.target_h), self.interpolation)


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))

class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = random.random() * 1.5
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=False)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))

class PureGaussianBlur(object):
    def __call__(self, img):
        blurred_img = gaussian(np.array(img), sigma=1.2, multichannel=False)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))
        
class RandomBilateralBlur(object):
    def __call__(self, img):
        sigma = random.uniform(0.,0.75)
        blurred_img = denoise_bilateral(np.array(img), sigma_spatial=sigma, multichannel=False)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))
    
class RandomNoise(object):
    def __call__(self, img):
        img = noising(np.asarray(img)) 
        return Image.fromarray(img.astype(np.uint8))
    
class ToRGB(object):
    def __call__(self, img):
        if len(img.size)==2:
            return img.convert('RGB')
        else:
            return img

try:
    import accimage
except ImportError:
    accimage = None


def noising(image):
    """
    Fonction pour ajouter du bruit speckle, gaussian et salt and pepper
    de manière aléatoire
    """
    # # adding speckle noise
    # delta = np.random.uniform(0,0.05)
    # gauss = np.random.randn(*image.shape)
    # gauss = gauss.reshape(*image.shape)        
    # image = image + image * gauss * delta
    
    #adding Gaussian noise  
    mean = 0
    var = np.var(image)*np.random.uniform(0.2,0.5)
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(*image.shape)
    image = image + gauss
    
    return image
    # #adding salt and pepper noise
    # s_vs_p = 0.5
    # amount = np.random.uniform(0,0.005)
    # out = np.copy(image)
    
    # # Salt mode
    # num_salt = np.ceil(amount * image.size * s_vs_p)
    # coords = [np.random.randint(0, i - 1, int(num_salt))
    # for i in image.shape]
    # out[coords] = np.max(image)
    
    # # Pepper mode
    # num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    # coords = [np.random.randint(0, i - 1, int(num_pepper))
    # for i in image.shape]
    # out[coords] = np.min(image)
    # return out

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = torch_tr.Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)
