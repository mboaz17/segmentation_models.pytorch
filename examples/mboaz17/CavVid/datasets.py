import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset as BaseDataset
# import airsim
import os
# import mpldatacursor

class CamvidDataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


# class AirSimDataset(BaseDataset):
#     """AirSim Dataset. Read images, apply augmentation and preprocessing transformations.
#
#     Args:
#         images_dir (str): path to images folder
#         masks_dir (str): path to segmentation masks folder
#         class_values (list): values of classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing
#             (e.g. noralization, shape manipulation, etc.)
#
#     """
#
#     CLASSES = airsim.AirSimDataset.class_names  # ['sky', 'building', 'pole', 'road', 'pavement',
#
#     # 'tree', 'signsymbol', 'fence', 'car',
#     # 'pedestrian', 'bicyclist', 'unlabelled']
#
#     def __init__(
#             self,
#             images_dir,
#             masks_dir,
#             classes=None,
#             augmentation=None,
#             preprocessing=None,
#     ):
#         self.ids = os.listdir(images_dir)
#         self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
#         self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
#
#         # convert str names to class values on masks
#         self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
#
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
#
#     def __getitem__(self, i):
#
#         # read data
#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.masks_fps[i], 0)
#
#         # extract certain classes from mask (e.g. cars)
#         masks = [(mask == v) for v in self.class_values]
#         mask = np.stack(masks, axis=-1).astype('float')
#
#         # apply augmentations
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#
#         # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#
#         return image, mask
#
#     def __len__(self):
#         return len(self.ids)


# ### Augmentations
# Since our dataset is very small we will apply a large number of different augmentations:
#  - horizontal flip
#  - affine transforms
#  - perspective transforms
#  - brightness/contrast/colors manipulations
#  - image blurring and sharpening
#  - gaussian noise
#  - random crops
#
# All this transforms can be easily applied with [**Albumentations**](https://github.com/albu/albumentations/) - fast augmentation library.
# For detailed explanation of image transformations you can look at [kaggle salt segmentation exmaple](https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb) provided by [**Albumentations**](https://github.com/albu/albumentations/) authors.

import albumentations as albu

def get_training_augmentation():
    N = 320
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=N, min_width=N, always_apply=True, border_mode=0),
        albu.RandomCrop(height=N, width=N, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_training_augmentation_none():
    """Add paddings to make image shape divisible by 32"""
    train_transform = [
        albu.PadIfNeeded(384, 480)
        # albu.PadIfNeeded(1088, 1440)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
        # albu.PadIfNeeded(1088, 1440)
    ]
    return albu.Compose(test_transform)


def get_augmentation_for_kitti():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(512, 1408)
    ]
    return albu.Compose(test_transform)


def get_augmentation_for_Airsim():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(1088, 1440)
    ]
    return albu.Compose(test_transform)


def get_conf_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.RandomCrop(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        # mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
    plt.show()
