import os
import pickle
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from examples.mboaz17.conf_utils.conf_est import ConfEst
from datasets import *
from torch.utils.data import DataLoader

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
CLASSES = [CamvidDataset.CLASSES[i] for i in [0, 3, 4]]
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
model = torch.load('./best_model.pth')

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# %%
conf_dataset = CamvidDataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_conf_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# same image with different random transforms
# for i in range(5):
#     image, mask = conf_dataset[1]
#     visualize(image=image) #, mask=mask.squeeze(-1))

conf_loader = DataLoader(conf_dataset, batch_size=1, shuffle=False, num_workers=0)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# %%

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

# loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.CrossEntropyLoss(class_intervals=class_intervals, activation='softmax2d')
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

## Define a confidence object
conf_obj = ConfEst()

# run model for 1 epoch

max_score = 0
iter_num = 0
for i in range(0, iter_num):
    # train_logs = train_epoch.run(train_loader, conf_obj=conf_obj)
    valid_logs = valid_epoch.run(conf_loader, conf_obj=conf_obj)
    with open('./conf_model.pkl', 'wb') as output:
        pickle.dump(conf_obj, output, pickle.HIGHEST_PROTOCOL)
        print('confidence model saved')

if iter_num == 0:  # then load conf_obj
    with open('./conf_model.pkl', 'rb') as input:
        conf_obj = pickle.load(input)

## Test best saved model

# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# %%

# create test dataset
test_dataset = CamvidDataset(
    x_test_dir, y_test_dir,
    # x_train_dir, y_train_dir,
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
test_dataset_vis = CamvidDataset(
    x_test_dir, y_test_dir,
    # x_train_dir, y_train_dir,
    classes=CLASSES,
)

# %%

# dataset_mode = 'from_dataset'
dataset_mode = 'from_folder'
# images_folder = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Kitti/2011_09_26_drive_0001_extract/image_02/data'
images_folder = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Kitti/2011_09_26_drive_0009_extract/image_02/data'

for i in range(20):

    if dataset_mode == 'from_dataset':
        n = np.random.choice(len(test_dataset))
        image_vis = test_dataset_vis[n][0].astype('uint8')
        shape_orig = image_vis.shape[:2]
        gt_mask_vis = test_dataset_vis[n][1].astype('uint8')
        image, gt_mask = test_dataset[n]
        gt_mask_vis = gt_mask_vis[:, :, :1].squeeze()
    elif dataset_mode == 'from_folder':
        images_list = os.listdir(images_folder)
        n = np.random.choice(len(images_list))
        image_vis = cv2.imread(os.path.join(images_folder, images_list[n]))
        shape_orig = image_vis.shape[:2]
        # Apply the same steps on the image as for an image from the dataset
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
        image = get_augmentation_for_kitti()(image=image_vis)['image']
        image = get_preprocessing(preprocessing_fn)(image=image)['image']

    shape_new = image.shape[1:]
    h_pad = int((shape_new[0] - shape_orig[0])/2)
    v_pad = int((shape_new[1] - shape_orig[1])/2)

    if dataset_mode == 'from_dataset':
        gt_mask = gt_mask.squeeze()
        max_vals = gt_mask.max(axis=0)
    else:
        gt_mask = np.zeros((image.shape[1], image.shape[2]))
        max_vals = np.zeros((image.shape[1], image.shape[2]))

    # gt_mask = gt_mask.argmax(axis=0)
    gt_mask = gt_mask[h_pad:shape_new[0]-h_pad, v_pad:shape_new[1]-v_pad]
    max_vals = max_vals[h_pad:shape_new[0]-h_pad, v_pad:shape_new[1]-v_pad]
    untagged_indices = (max_vals == 0).nonzero()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask, score_map = best_model.predict(x_tensor, conf_obj=conf_obj, mode='compare')
    pr_score = (pr_mask.squeeze().cpu().numpy()).max(axis=0)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round()).argmax(axis=0)
    score_map = (score_map.squeeze().cpu().numpy())
    pr_mask = pr_mask[h_pad:shape_new[0]-h_pad, v_pad:shape_new[1]-v_pad]
    pr_mask_vis = 0*image_vis
    gt_mask_vis = 0*image_vis

    for i in range(len(class_values)):
        inds = (pr_mask == i).nonzero()
        pr_mask_vis[inds[0], inds[1], :] = class_values[i]
        inds = (gt_mask == i).nonzero()
        gt_mask_vis[inds[0], inds[1], :] = class_values[i]
    gt_mask_vis[untagged_indices[0], untagged_indices[1], :] = 0

    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask_vis,
        predicted_mask=pr_mask_vis,
        softmax_score=pr_score > 0.99,  # np.percentile(pr_score, 50),
        score_map=score_map > 0.01,  # np.percentile(score_map, 50),
    )
    print(np.percentile(pr_score, 60))
    print(np.percentile(score_map, 60))

# TODO: for imporiving confidnece scores
# * Estimate one histogram model per class?
# * Estimate many histogram models, using clustering?
# * Multi-dim histogram models (2 or 3)
# * Determine importance of each feature level
# * Determine number of bins per feature level
# * Weigh each image using the number of contributing pixels (not equally)
# * Determine best histogram matching criterion
# * Take features before the nonlinearity (but it's more technically complicated)
# * Calculate true prob density after dim reduction - perhaps using PCA