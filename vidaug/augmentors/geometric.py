"""
Augmenters that apply geometric transformations.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.RandomRotate(30),
                          va.RandomResize(0.2)  ])

List of augmenters:
    * GaussianBlur
    * ElasticTransformation
    * PiecewiseAffineTransform
    * Superpixel
"""

from skimage import segmentation, measure
import numpy as np
import random
import numbers
import scipy
import PIL
import cv2


class GaussianBlur(object):
    """
    Augmenter to blur images using gaussian kernels.

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, clip):

        if isinstance(clip[0], np.ndarray):
            return [scipy.ndimage.gaussian_filter(img, sigma=self.sigma, order=0) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.filter(PIL.ImageFilter.GaussianBlur(radius=self.sigma)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))


class ElasticTransformation(object):
    """
    Augmenter to transform images by moving pixels locally around using
    displacement fields.
    See
        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003
    for a detailed explanation.

    Args:
        alpha (float): Strength of the distortion field. Higher values mean
        more "movement" of pixels.

        sigma (float): Standard deviation of the gaussian kernel used to
        smooth the distortion fields.

        order (int): Interpolation order to use. Same meaning as in
        `scipy.ndimage.map_coordinates` and may take any integer value in
        the range 0 to 5, where orders close to 0 are faster.

        cval (int): The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to "constant".
        For standard uint8 images (value range 0-255), this value may also
        come from the range 0-255. It may be a float value, even for
        integer image dtypes.

        mode : Parameter that defines the handling of newly created pixels.
        May take the same values as in `scipy.ndimage.map_coordinates`,
        i.e. "constant", "nearest", "reflect" or "wrap".
    """
    def __init__(self, alpha=0, sigma=0, order=3, cval=0, mode="constant",
                 name=None, deterministic=False):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        result = []
        nb_images = len(clip)
        for i in range(nb_images):
            image = clip[i]
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            result.append(self._map_coordinates(
                clip[i],
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in result]
        else:
            return result

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2),"shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3),"image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result



class PiecewiseAffineTransform(object):
    """
    Augmenter that places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.

     Args:
         displacement (init): gives distorted image depending on the valuse of displacement_magnification and displacement_kernel

         displacement_kernel (init): gives the blury effect

         displacement_magnification (float): it magnify the image
    """
    def __init__(self, displacement=0, displacement_kernel=0, displacement_magnification=0):
        self.displacement = displacement
        self.displacement_kernel = displacement_kernel
        self.displacement_magnification = displacement_magnification

    def __call__(self, clip):

        ret_img_group = clip
        if isinstance(clip[0], np.ndarray):
            im_size = clip[0].shape
            image_w, image_h = im_size[1], im_size[0]
        elif isinstance(clip[0], PIL.Image.Image):
            im_size = clip[0].size
            image_w, image_h = im_size[0], im_size[1]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        displacement_map = np.random.rand(image_h, image_w, 2) * 2 * self.displacement - self.displacement
        displacement_map = cv2.GaussianBlur(displacement_map, None,
                                            self.displacement_kernel)
        displacement_map *= self.displacement_magnification * self.displacement_kernel
        displacement_map = np.floor(displacement_map).astype('int32')

        displacement_map_rows = displacement_map[..., 0] + np.tile(np.arange(image_h), (image_w, 1)).T.astype('int32')
        displacement_map_rows = np.clip(displacement_map_rows, 0, image_h - 1)

        displacement_map_cols = displacement_map[..., 1] + np.tile(np.arange(image_w), (image_h, 1)).astype('int32')
        displacement_map_cols = np.clip(displacement_map_cols, 0, image_w - 1)

        if isinstance(clip[0], np.ndarray):
            return [img[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(img.shape) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [PIL.Image.fromarray(np.asarray(img)[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(np.asarray(img).shape)) for img in clip]



class Superpixel(object):
    """
    Completely or partially transform images to their superpixel representation.

    Args:
        p_replace (int) : Defines the probability of any superpixel area being
        replaced by the superpixel.

        n_segments (int): Target number of superpixels to generate.
        Lower numbers are faster.

        interpolation (str): Interpolation to use. Can be one of 'nearest',
        'bilinear' defaults to nearest
    """

    def __init__(self, p_replace=0, n_segments=0, max_size=360,
                 interpolation="bilinear"):
        self.p_replace = p_replace
        self.n_segments = n_segments
        self.interpolation = interpolation


    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        # TODO this results in an error when n_segments is 0
        replace_samples = np.tile(np.array([self.p_replace]), self.n_segments)
        avg_image = np.mean(clip, axis=0)
        segments = segmentation.slic(avg_image, n_segments=self.n_segments,
                                     compactness=10)

        if not np.max(replace_samples) == 0:
            print("Converting")
            clip = [self._apply_segmentation(img, replace_samples, segments) for img in clip]

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in clip]
        else:
            return clip

    def _apply_segmentation(self, image, replace_samples, segments):
        nb_channels = image.shape[2]
        image_sp = np.copy(image)
        for c in range(nb_channels):
            # segments+1 here because otherwise regionprops always misses
            # the last label
            regions = measure.regionprops(segments + 1,
                                          intensity_image=image[..., c])
            for ridx, region in enumerate(regions):
                # with mod here, because slic can sometimes create more 
                # superpixel than requested. replace_samples then does 
                # not have enough values, so we just start over with the
                # first one again.
                if replace_samples[ridx % len(replace_samples)] == 1:
                    mean_intensity = region.mean_intensity
                    image_sp_c = image_sp[..., c]
                    image_sp_c[segments == ridx] = mean_intensity

        return image_sp

