import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from datasets import *
from torch.utils.data import DataLoader
from examples.mboaz17.conf_utils.conf_est import ConfEst
import examples.mboaz17.conf_utils.vae as vae

# %%
np.random.seed(0)
DATA_DIR = '../../data/CamVid/'

# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')

# %%

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

# %%
# Lets look at data we have
# dataset = CamvidDataset(x_train_dir, y_train_dir, classes=['car'])
# image, mask = dataset[4]  # get some sample
# visualize(
#     image=image,
#     cars_mask=mask.squeeze(),
# )

#### Visualize resulted augmented images and masks
augmented_dataset = CamvidDataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    classes=['car'],
)

# same image with different random transforms
# for i in range(3):
#     image, mask = augmented_dataset[1]
#     visualize(image=image, mask=mask.squeeze(-1))

## Create model and train
import torch
import numpy as np
import segmentation_models_pytorch as smp

# %%

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = [CamvidDataset.CLASSES[i] for i in [0, 1, 3]]
# CLASSES_Camvid = ['sky', 'building', 'pole', 'road', 'pavement',
#                'tree', 'signsymbol', 'fence', 'car',
#                'pedestrian', 'bicyclist', 'unlabelled']
class_intervals = np.ones((len(CLASSES)))
# class_intervals[1] = 1e6
class_values = [np.uint8(255 * np.random.rand(1, 3)) for c in CLASSES]
ACTIVATION = None  # 'softmax2d'  # 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# %%

train_dataset = CamvidDataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = CamvidDataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# %%

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

# loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.CrossEntropyLoss(class_intervals=class_intervals, activation='softmax2d')
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

vae_obj = vae.VAE(channels_num=128, n_latent=8)
optimizer = torch.optim.Adam([
    dict(params=list(model.parameters()) + list(vae_obj.parameters()), lr=0.0001),
    # dict(params=model.parameters(), lr=0.0001),
])

# %%

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 10 epochs
max_score = 0
iter_num = 5
for i in range(0, iter_num):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader, vae_obj=vae_obj)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if 1: #max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

        with open('./vae_model.pkl', 'wb') as output:
            pickle.dump(vae_obj, output, pickle.HIGHEST_PROTOCOL)
            print('vae model saved')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
