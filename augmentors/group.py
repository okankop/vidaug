"""
Augmenters that apply to a group of augmentations, like selecting
an augmentation from a list, or applying all the augmentations in
a list sequentially

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.HorizontalFlip(),
                          va.VerticalFlip() ])

List of augmenters:
    * Sequential
    * OneOf
    * SomeOf
    * Sometimes

"""

import numpy as np
import PIL
import random


class Sequential(object):
    """
    Composes several augmentations together.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.

        random_order (bool): Whether to apply the augmentations in random order.
    """

    def __init__(self, transforms, random_order=False):
        self.transforms = transforms
        self.rand = random_order

    def __call__(self, clip):
        if self.rand:
            rand_transforms = self.transforms[:]
            random.shuffle(rand_transforms)
            for t in rand_transforms:
                clip = t(clip)
        else:
            for t in self.transforms:
                clip = t(clip)

        return clip


class OneOf(object):
    """
    Selects one augmentation from a list.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        select = random.choice(self.transforms)
        clip = select(clip)
        return clip


class SomeOf(object):
    """
    Selects a given number of augmentation from a list.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations.

        N (int): The number of augmentations to select from the list.

        random_order (bool): Whether to apply the augmentations in random order.

    """

    def __init__(self, transforms, N, random_order=True):
        self.transforms = transforms
        self.rand = random_order
        if N > len(transforms):
            raise TypeError('The number of applied augmentors should be smaller than the given augmentation number')
        else:
            self.N = N

    def __call__(self, clip):
        if self.rand:
            tmp = self.transforms[:]
            selected_trans = [tmp.pop(random.randrange(len(tmp))) for _ in range(self.N)]
            for t in selected_trans:
                clip = t(clip)
            return clip
        else:
            indices = [i for i in range(len(self.transforms))]
            selected_indices = [indices.pop(random.randrange(len(indices)))
                                for _ in range(self.N)]
            selected_indices.sort()
            selected_trans = [self.transforms[i] for i in selected_indices]
            for t in selected_trans:
                clip = t(clip)
            return clip


class Sometimes(object):
    """
    Applies an augmentation with a given probability.

    Args:
        p (float): The probability to apply the augmentation.

        transform (an "Augmentor" object): The augmentation to apply.

    Example: Use this this transform as follows:
        sometimes = lambda aug: va.Sometimes(0.5, aug)
        sometimes(va.HorizontalFlip)
    """

    def __init__(self, p, transform):
        self.transform = transform
        if (p > 1.0) | (p < 0.0):
            raise TypeError('Expected p to be in [0.0 <= 1.0], ' +
                            'but got p = {0}'.format(p))
        else:
            self.p = p

    def __call__(self, clip):
        if random.random() < self.p:
            clip = self.transform(clip)
        return clip
