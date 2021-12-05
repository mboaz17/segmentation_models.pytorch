import albumentations as albu
import cv2

M = 1280  # 2560
N = 1280  # 2560

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # albu.ShiftScaleRotate(scale_limit=0.20, rotate_limit=20, shift_limit=0.1, p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
        # albu.GridDistortion(p=0.5),
        # albu.Resize(*resize_to),
        # albu.ChannelShuffle(),
        # albu.InvertImg(),
        # albu.ToGray(),
        # albu.Normalize(),
        albu.PadIfNeeded(min_height=M, min_width=N, always_apply=True, border_mode=0),
        albu.RandomCrop(height=M, width=N, always_apply=True),

        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),
        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(M, N)
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


# Pixel-level transforms
# Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. The list of pixel-level transforms:
#
# Blur
# CLAHE
# ChannelDropout
# ChannelShuffle
# ColorJitter
# Downscale
# Emboss
# Equalize
# FDA
# FancyPCA
# FromFloat
# GaussNoise
# GaussianBlur
# GlassBlur
# HistogramMatching
# HueSaturationValue
# ISONoise
# ImageCompression
# InvertImg
# MedianBlur
# MotionBlur
# MultiplicativeNoise
# Normalize
# PixelDistributionAdaptation
# Posterize
# RGBShift
# RandomBrightnessContrast
# RandomFog
# RandomGamma
# RandomRain
# RandomShadow
# RandomSnow
# RandomSunFlare
# RandomToneCurve
# Sharpen
# Solarize
# Superpixels
# ToFloat
# ToGray
# ToSepia
# Spatial-level transforms
# Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. The following table shows which additional targets are supported by each transform.
#
# Transform	Image	Masks	BBoxes	Keypoints
# Affine	✓	✓	✓	✓
# CenterCrop	✓	✓	✓	✓
# CoarseDropout	✓	✓
# Crop	✓	✓	✓	✓
# CropAndPad	✓	✓	✓	✓
# CropNonEmptyMaskIfExists	✓	✓	✓	✓
# ElasticTransform	✓	✓
# Flip	✓	✓	✓	✓
# GridDistortion	✓	✓
# GridDropout	✓	✓
# HorizontalFlip	✓	✓	✓	✓
# Lambda	✓	✓	✓	✓
# LongestMaxSize	✓	✓	✓	✓
# MaskDropout	✓	✓
# NoOp	✓	✓	✓	✓
# OpticalDistortion	✓	✓
# PadIfNeeded	✓	✓	✓	✓
# Perspective	✓	✓	✓	✓
# PiecewiseAffine	✓	✓	✓	✓
# RandomCrop	✓	✓	✓	✓
# RandomCropNearBBox	✓	✓	✓	✓
# RandomGridShuffle	✓	✓
# RandomResizedCrop	✓	✓	✓	✓
# RandomRotate90	✓	✓	✓	✓
# RandomScale	✓	✓	✓	✓
# RandomSizedBBoxSafeCrop	✓	✓	✓
# RandomSizedCrop	✓	✓	✓	✓
# Resize	✓	✓	✓	✓
# Rotate	✓	✓	✓	✓
# SafeRotate	✓	✓	✓	✓
# ShiftScaleRotate	✓	✓	✓	✓
# SmallestMaxSize	✓	✓	✓	✓
# Transpose	✓	✓	✓	✓
# VerticalFlip	✓	✓	✓	✓
