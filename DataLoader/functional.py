'''
some function to handle image
'''

import os
import random
import time
import numpy as np
from PIL import Image, ImageOps
import numbers
from torchvision.transforms import Pad,RandomHorizontalFlip
from torchvision.transforms import ToTensor, ToPILImage

class MyCrop(object):
    def __init__(self):
        pass

    def crop(self, img, i, j, h, w):
        """Crop the given PIL Image.
        Args:
            img (PIL Image): Image to be cropped.
            i: Upper pixel coordinate.
            j: Left pixel coordinate.
            h: Height of the cropped image.
            w: Width of the cropped image.
        Returns:
            PIL Image: Cropped image.
        """
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        return img.crop((j, i, j + w, i + h))
    
    def padChannel(self, channel, padding):
        return np.pad(channel, ((padding, padding),(padding, padding)), 'constant', constant_values=(0,0))

    def padImage(self, img, padding):
        img = np.array(img)

        channel1 = img[:,:,0]
        channel2 = img[:,:,1]
        channel3 = img[:,:,2]

        channel1 = self.padChannel(channel1, padding)
        channel2 = self.padChannel(channel2, padding)
        channel3 = self.padChannel(channel3, padding)

        img = np.dstack((channel1,channel2,channel3))
        return Image.fromarray(img)

class RandomCrop(MyCrop):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
            
        #对pytorch包内RandomCrop做了修改，可以同时处理image和target，保证为同一区域。
    """
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def getParams(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
            
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, target=None):  #crop the same area of ori-image and target 
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = self.padImage(img, self.padding)

        i, j, h, w = self.getParams(img, self.size)
        
        if target is not None:
            return self.crop(img, i, j, h, w), self.crop(target, i, j, h, w)
        else:
            return self.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomFlip(MyCrop):
    """Randomflip the given PIL Image randomly with a given probability. horizontal or vertical
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """ 
        # make sure that crop area of  image and target are the same
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if target is not None:
                target = target.transpose(Image.FLIP_LEFT_RIGHT) #left or right
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if target is not None:
                target = target.transpose(Image.FLIP_TOP_BOTTOM) # bottom or top
        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class CenterCrop(MyCrop):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
       
    def __call__(self, img, target=None):  
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """

        w, h = img.size
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        if target is not None:
            return self.crop(img, j, i, tw, th), self.crop(target, j, i, tw, th)
        else:
            return self.crop(img, j, i, tw, th)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomRotate(MyCrop):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, target=None):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        if target is not None:
            return img.rotate(rotate_degree, Image.BILINEAR), target.rotate(rotate_degree, Image.NEAREST)
        else:
            return img.rotate(rotate_degree, Image.BILINEAR)
