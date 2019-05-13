# Video Augmentation Techniques for Deep Learning
This python library helps you with augmenting videos for your deep learning architectures.
It converts input videos into a new, much larger set of slightly altered videos.


 <p align="center"><img src="videos/combined.gif" align="center" width="640" height="480" title="Augmentations" /></p>


  Original Video     
  
 <p align="center"><img src="videos/original.gif" align="center" width="320" height="240" title="Original Video" /></p>


## Requirements and installation

Required packages:
* numpy
* PIL
* scipy
* skimage
* OpenCV (i.e. `cv2`)

For installation, simply use `sudo pip install git+https://github.com/okankop/vidaug`.
Alternatively, the repository can be download via `git clone https://github.com/okankop/vidaug` and installed by using `python setup.py sdist && pip install dist/vidaug-0.1.tar.gz`.


## Examples

A classical video classification with CNN using augmentations on videos.
Train on batches of images and augment each batch via random crop, random crop and horizontal flip:
```python
from vidaug import augmentors as va

sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
seq = va.Sequential([
    va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
    va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
    sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
])

for batch_idx in range(1000):
    # 'video' should be either a list of images from type of numpy array or PIL images
    video = load_batch(batch_idx)
    video_aug = seq(video)
    train_on_video(video)
```



The videos below show examples for most augmentation techniques:

Augmentation Type                   |  Augmented Video
:----------------------------------:|:-------------------------:
Piecewise Affine Transform          |  ![Piecewise Affine Transform](videos/elastic.gif?raw=true "Piecewise Affine Transform")
Superpixel                          |  ![Superpixel](videos/segmented.gif?raw=true "Superpixel")
Gausian Blur                        |  ![Gausian Blur](videos/blurred.gif?raw=true "Gausian Blur")
Invert Color                        |  ![Invert Color](videos/inverted.gif?raw=true "Invert Color")
Rondom Rotate                       |  ![Rondom Rotate](videos/rotated.gif?raw=true "Rondom Rotate")
Random Resize                       |  ![Random Resize](videos/resized.gif?raw=true "Random Resize")
Translate                           |  ![Translate](videos/translated.gif?raw=true "Translate")
Center Crop                         |  ![Center Crop](videos/centercrop.gif?raw=true "Center Crop")
Horizontal Flip                     |  ![Horizontal Flip](videos/flipped.gif?raw=true "Horizontal Flip")
Vertical Flip                       |  ![Vertical Flip](videos/vertflip.gif?raw=true "Vertical Flip")
Add                                 |  ![Add](videos/add.gif?raw=true "Add")
Multiply                            |  ![Multiply](videos/multiply.gif?raw=true "Multiply")
Downsample                          |  ![Downsample](videos/downsample.gif?raw=true "Downsample")
Upsample                            |  ![Upsample](videos/upsample.gif?raw=true "Upsample")
Elastic Transformation              |  ![Elastic Transformation](videos/elasticTransformation.gif?raw=true "Elastic Transformation")
Salt                                |  ![Salt](videos/salt.gif?raw=true "Salt")
Pepper                              |  ![Cropping](videos/pepper.gif?raw=true "Pepper")
Shear                               |  ![Shear](videos/shear.gif?raw=true "Shear")

