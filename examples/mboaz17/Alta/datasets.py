import os
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """Alta Dataset. Read images, apply augmentation and preprocessing transformations.

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
        # '__background__': (0, 0, 0),
        'bicycle': (255, 50, 50),
        'building': (255, 127, 50),
        'fence': (255, 204, 50),
        'other objects': (229, 255, 50),
        'person': (153, 255, 50),
        'pole': (76, 255, 50),
        'rough terrain': (50, 255, 101),
        'shed': (50, 255, 178),
        'soft terrain': (50, 255, 255),
        'stairs': (50, 178, 255),
        'transportation terrain': (50, 101, 255),
        'vegetation': (76, 50, 255),
        'vehicle': (153, 50, 255),
        'walking terrain': (229, 50, 255),
        'water': (255, 50, 204),
    }
    # samples_num_all = [1822091, 4697038, 5583920]
    # class_intervals = samples_num_all / np.min(samples_num_all)
    class_intervals = [1, 50, 1, 1, 1, 1, 50, 1, 50, 1, 50, 50, 1, 50, 1]
    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.ids.sort()
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
