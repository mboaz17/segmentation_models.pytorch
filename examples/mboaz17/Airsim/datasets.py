import os
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """Airsim Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = {
        'road': (203, 174, 42),
        'sidewalk': (177, 172, 224),
        'grass': (160, 183, 145),
        'dense vegetation': (224, 241, 137),
        'buildings': (232, 224, 132),
        'parking lots': None,
        'water': None,
        'construction': None,
        'sports field': None,
        'sky': None,
        'ground': None,
        'cars': (249, 173, 199),
        'people': (205, 228,  85),
        'poles': (121, 160, 208),
        'other': (141, 238, 180)
        # ### UNKONWN COLORS ###
        # (128, 219, 130),
        # (170,  71, 248),
    }

    # class_intervals = None
    class_intervals = [50, 50, 50, 50, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            stride=1, bias=0,
    ):
        def sort_func_key(x):  # in order to sort by the numerical value of the image name
            return int(x.split('.')[0])

        self.ids = os.listdir(images_dir)
        try:
            self.ids.sort(key=sort_func_key)
        except:
            self.ids.sort()
        self.ids = self.ids[bias::stride]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id).replace('JPG','png') for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES[cls] for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # extract certain classes from mask (e.g. cars)
        masks = [np.all(mask == v, axis=2) for v in self.class_values]
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
