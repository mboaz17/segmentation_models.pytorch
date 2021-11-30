import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%
# images_dir = '/media/isl12/Alta/V7_Exp_25_1_21'
# annotations_dir = '/media/isl12/Alta/V7_Exp_25_1_21_annot'
# dataset_name = 'Agamim/Path/A/30'

images_dir = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Alta/train'
annotations_dir = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Alta/trainannot'
dataset_name = ''

x_train_dir = os.path.join(images_dir, dataset_name)
y_train_dir = os.path.join(annotations_dir, dataset_name)
x_valid_dir = os.path.join(images_dir, dataset_name)
y_valid_dir = os.path.join(annotations_dir, dataset_name)
x_test_dir = os.path.join(images_dir, dataset_name)
y_test_dir = os.path.join(annotations_dir, dataset_name)

# %%
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
    plt.show()

### Dataloader
# %%

from torch.utils.data import DataLoader
from datasets import Dataset

# %%
# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['vegetation'])
# image, mask = dataset[4]  # get some sample
# visualize(
#     image=image,
#     cars_mask=mask.squeeze(),
# )

### Augmentations
from augmentations import get_training_augmentation, get_validation_augmentation, get_preprocessing

#### Visualize resulted augmented images and masks

augmented_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    classes=['transportation terrain'],
)

# same image with different random transforms
# for i in range(3):
#     image, mask = augmented_dataset[0]
#     visualize(image=image, mask=mask.squeeze(-1))

## Create model and train
import torch
import numpy as np
import segmentation_models_pytorch as smp

# %%

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = [cls for ind, cls in enumerate(augmented_dataset.CLASS_NAMES) if ind>=-1]  # ind in [1, 6, 8, 10, 11, 12, 13]]
CLASS_INTERVALS = [interval for ind, interval in enumerate(augmented_dataset.class_intervals) if ind>=-1]  # ind in [1, 6, 8, 10, 11, 12, 13]]
ACTIVATION = 'softmax2d'  # 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
# model = torch.load('./AgamimPathA_70/Dice_4images_imbalanced.pth')

# pretrained_model = torch.load('./AgamimPathA_70/Dice_1_6_8_10_11_12_13.pth')
# model.encoder = pretrained_model.encoder
# model.decoder = pretrained_model.decoder

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# %%

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
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

# loss = smp.utils.losses.CrossEntropyLoss()  # (class_intervals=train_dataset.class_intervals)
loss = smp.utils.losses.DiceLoss(class_intervals=CLASS_INTERVALS)
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
    # dict(params=model.segmentation_head.parameters(), lr=0.0001),
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

# %%

# train model for 40 epochs

max_score = 0
iter_num = 40
for i in range(0, iter_num):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    # valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if 1:  # max_score < valid_logs['iou_score']:
        # max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

## Test best saved model

# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# %%

# create test dataset
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

# %%

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

# logs = test_epoch.run(test_dataloader)

## Visualize predictions
# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir,
    classes=CLASSES,
)

# %%

for i in range(5):
    n = i  # np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask_pos = gt_mask.sum(axis=0)
    gt_mask = gt_mask.argmax(axis=0)

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_score = (pr_mask.squeeze().cpu().numpy()).max(axis=0)
    pr_mask = (pr_mask.squeeze(dim=0).cpu().numpy().round())
    pr_mask = pr_mask.argmax(axis=0)
    pr_mask_vis = 0*image_vis
    gt_mask_vis = 0*image_vis
    for i in range(len(train_dataset.class_values)):
        inds = (pr_mask == i).nonzero()
        pr_mask_vis[inds[0], inds[1], :] = train_dataset.class_values[i]
        inds = ((gt_mask == i) & (gt_mask_pos > 0)).nonzero()
        gt_mask_vis[inds[0], inds[1], :] = train_dataset.class_values[i]

    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask_vis,
        predicted_mask=pr_mask_vis,
        softmax_score=pr_score,
    )

# TODO:
# 1) Support class_intervals for batch size > 1
# 2) Support class_intervals for DiceLoss?