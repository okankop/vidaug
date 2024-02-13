"""
Augmenters that apply transformations on the pixel intensities.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.RandomRotate(30),
                          va.RandomResize(0.2)  ])

List of augmenters:
    * InvertColor
    * Add
    * Multiply
    * Pepper
    * Salt
"""


import numpy as np
import random
import PIL
from PIL import ImageOps



class InvertColor(object):
    """
    Inverts the color of the video.
    """

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            return [np.invert(img) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            inverted = [ImageOps.invert(img) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return inverted


class Add(object):
    """
    Add a value to all pixel intesities in an video.

    Args:
        value (int): The value to be added to pixel intesities.
    """

    def __init__(self, value=0):
        if value > 255 or value < -255:
            raise TypeError('The video is blacked or whitened out since ' +
                            'value > 255 or value < -255.')
        self.value = value

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.int32)
            image += self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Multiply(object):
    """
    Multiply all pixel intensities with given value.
    This augmenter can be used to make images lighter or darker.

    Args:
        value (float): The value with which to multiply the pixel intensities
        of video.
    """

    def __init__(self, value=1.0):
        if value < 0.0:
            raise TypeError('The video is blacked out since for value < 0.0')
        self.value = value

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.float64)
            image *= self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Pepper(object):
    """
    Augmenter that sets a certain fraction of pixel intensities to 0, hence
    they become black.

    Args:
        ratio (int): Determines number of black pixels on each frame of video.
        Smaller the ratio, higher the number of black pixels.
    """
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float64)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final

class Salt(object):
    """
    Augmenter that sets a certain fraction of pixel intesities to 255, hence
    they become white.

    Args:
        ratio (int): Determines number of white pixels on each frame of video.
        Smaller the ratio, higher the number of white pixels.
   """
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float64)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final
