"""
Augmenters that apply temporal transformations.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.RandomRotate(30),
                          va.RandomResize(0.2)  ])

List of augmenters:
    * TemporalBeginCrop
    * TemporalCenterCrop
    * TemporalRandomCrop
    * InverseOrder
    * Downsample
    * Upsample
    * TemporalFit
    * TemporalElasticTransformation
    * DropRandomFrames
    * DuplicateRandomFrames
    
"""

import numpy as np
import PIL
import random
import math



class TemporalBeginCrop(object):
    """
    Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        out = clip[:self.size]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalCenterCrop(object):
    """
    Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        center_index = len(clip) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalRandomCrop(object):
    """
    Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        rand_end = max(0, len(clip) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class InverseOrder(object):
    """
    Inverts the order of clip frames.
    """

    def __call__(self, clip):
        for i in range(len(clip)):
            nb_images = len(clip)
            return [clip[img] for img in reversed(range(1, nb_images))]


class Downsample(object):
    """
    Temporally downsample a video by deleting some of its frames.

    Args:
        ratio (float): Downsampling ratio in [0.0 <= ratio <= 1.0].
    """
    def __init__(self , ratio=1.0):
        if ratio < 0.0 or ratio > 1.0:
            raise TypeError('ratio should be in [0.0 <= ratio <= 1.0]. ' +
                            'Please use upsampling for ratio > 1.0')
        self.ratio = ratio

    def __call__(self, clip):
        nb_return_frame = np.floor(self.ratio * len(clip)).astype(int)
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i-1] for i in return_ind]


class Upsample(object):
    """
    Temporally upsampling a video by deleting some of its frames.

    Args:
        ratio (float): Upsampling ratio in [1.0 < ratio < infinity].
    """
    def __init__(self , ratio=1.0):
        if ratio < 1.0:
            raise TypeError('ratio should be 1.0 < ratio. ' +
                            'Please use downsampling for ratio <= 1.0')
        self.ratio = ratio

    def __call__(self, clip):
        nb_return_frame = np.floor(self.ratio * len(clip)).astype(int)
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i-1] for i in return_ind]


class TemporalFit(object):
    """
    Temporally fits a video to a given frame size by
    downsampling or upsampling.

    Args:
        size (int): Frame size to fit the video.
    """
    def __init__(self, size):
        if size < 0:
            raise TypeError('size should be positive')
        self.size = size

    def __call__(self, clip):
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=self.size)]

        return [clip[i-1] for i in return_ind]


class TemporalElasticTransformation(object):
    """
    Stretches or schrinks a video at the beginning, end or middle parts.
    In normal operation, augmenter stretches the beggining and end, schrinks
    the center.
    In inverse operation, augmenter shrinks the beggining and end, stretches
    the center.
    """

    def __call__(self, clip):
        nb_images = len(clip)
        new_indices = self._get_distorted_indices(nb_images)
        return [clip[i] for i in new_indices]

    def _get_distorted_indices(self, nb_images):
        inverse = random.randint(0, 1)

        if inverse:
            scale = random.random()
            scale *= 0.21
            scale += 0.6
        else:
            scale = random.random()
            scale *= 0.6
            scale += 0.8

        frames_per_clip = nb_images

        indices = np.linspace(-scale, scale, frames_per_clip).tolist()
        if inverse:
            values = [math.atanh(x) for x in indices]
        else:
            values = [math.tanh(x) for x in indices]

        values = [x / values[-1] for x in values]
        values = [int(round(((x + 1) / 2) * (frames_per_clip - 1), 0)) for x in values]
        return values
    
    
class DropRandomFrames(object):
    """
    Randomly drops each frame in the video with a set probability

    Args:
        drop_chance (float): probability of deleting any frame
    """
    def __init__(self, drop_chance=0.0):
        if drop_chance < 0.0 or drop_chance > 1.0:
            raise ValueError("drop_chance must be a float in the range [0.0, 1.0]")
        self.drop_chance = drop_chance

    def __call__(self, clip):
        vid_size = len(clip)
        indices = np.array(range(vid_size))
        keep_vector = np.random.random(size=vid_size) > self.drop_chance

        return [clip[i] for i in indices[keep_vector]]


class DuplicateRandomFrames(object):
    """
    Randomly duplicates each frame in the video with a set probability

    Args:
        duplicate_chance (float): probability of a frame being duplicated
    """
    def __init__(self, duplicate_chance=0.0):
        if duplicate_chance < 0.0 or duplicate_chance > 1.0:
            raise ValueError("duplicate_chance must be a float in the range [0.0, 1.0]")
        self.duplicate_chance = duplicate_chance

    def __call__(self, clip):
        vid_size = len(clip)
        indices = list(range(vid_size))

        for i in range(len(indices) - 1, -1, -1):
            if np.random.random() < self.duplicate_chance:
                indices.insert(i, indices[i])

        return [clip[i] for i in indices]

