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

    CLASSES_ORIG = {
        # '__background__': (0, 0, 0),
        'building': (255, 127, 50),
        'transportation terrain': (50, 101, 255),
        'rough terrain': (50, 255, 101),
        'soft terrain': (50, 255, 255),
        'vegetation': (76, 50, 255),
        'walking terrain': (229, 50, 255),
        'vehicle': (153, 50, 255),
        'fence': (255, 204, 50),
        'other objects': (229, 255, 50),
        'person': (153, 255, 50),
        'pole': (76, 255, 50),
        'shed': (50, 255, 178),
        'stairs': (50, 178, 255),
        'water': (255, 50, 204),
        'bicycle': (255, 50, 50),
    }
    if 1:
        CLASSES = {
            # '__background__': (0, 0, 0),
            'building': (255, 127, 50),
            'transportation terrain': (50, 101, 255),
            'rough terrain': (50, 255, 101),
            'soft terrain': (50, 255, 255),
            'vegetation': (76, 50, 255),
            'walking terrain': (229, 50, 255),
            'other objects': (229, 255, 50),
        }
        class_mapping_curr2orig = { 0:[0], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6,7,8,9,10,11,12,13,14] }
        # class_intervals = [5, 10, 10, 1, 10, 10, 10]
        class_intervals = [10, 10, 5, 1, 5, 5, 5]
    else:
        CLASSES = CLASSES_ORIG
        class_mapping_curr2orig = {cls:[cls] for cls in range(len(CLASSES))}
        # class_intervals = [1 for cls in CLASSES.keys()]
        class_intervals = [10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    CLASS_NAMES = [k for k,v in CLASSES.items()]
    # samples_num_all = [1822091, 4697038, 5583920]
    # class_intervals = samples_num_all / np.min(samples_num_all)

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            sampling_interval=None,
    ):
        self.ids = os.listdir(images_dir)
        self.ids.sort()
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id).replace('JPG','png') for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values_orig = [self.CLASSES_ORIG[cls] for cls in self.CLASSES_ORIG]
        self.class_values = [self.CLASSES[cls] for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.sampling_interval = sampling_interval

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.sampling_interval:
            height = int(image.shape[0]/self.sampling_interval[0])
            width = int(image.shape[1]/self.sampling_interval[1])
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            # image = image[::self.sampling_interval[0], ::self.sampling_interval[1], :]
            row_shift = int(np.floor( self.sampling_interval[0]/2 ))
            col_shift = int(np.floor( self.sampling_interval[1]/2 ))
            mask = mask[row_shift::self.sampling_interval[0], col_shift::self.sampling_interval[1], :]

        # extract certain classes from mask (e.g. cars)
        # masks = [np.all(mask == v, axis=2) for v in self.class_values_orig]
        masks = []
        for cls in self.class_mapping_curr2orig:
            for i, cls_ind in enumerate(self.class_mapping_curr2orig[cls]):
                v = self.class_values_orig[cls_ind]
                if i==0:
                    mask_curr = np.all(mask == v, axis=2)
                else:
                    mask_curr = mask_curr | np.all(mask == v, axis=2)
            masks.append(mask_curr)

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
