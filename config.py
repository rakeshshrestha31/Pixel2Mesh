import os
import numpy as np

# dataset root

DATASET_ROOT = "./datasets/data"
SHAPENET_ROOT = os.path.join(DATASET_ROOT, "shapenet")
IMAGENET_ROOT = os.path.join(DATASET_ROOT, "imagenet")

# ellipsoid path
ELLIPSOID_PATH = os.path.join(DATASET_ROOT, "ellipsoid/info_ellipsoid.dat")

# pretrained weights path
PRETRAINED_WEIGHTS_PATH = {
    "vgg16": os.path.join(DATASET_ROOT, "pretrained/vgg16-397923af.pth"),
    "resnet50": os.path.join(DATASET_ROOT, "pretrained/resnet50-19c8e357.pth"),
    "vgg16p2m": os.path.join(DATASET_ROOT, "pretrained/vgg16-p2m.pth"),
}

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# factor for rescaling DTU points cloud to be compatible for training
DTU_RESCALE_FACTOR = 5.5e-4

# transformation to align with Pixel2Mesh (shapenet) coordinate frame
T_shapenet_dtu = np.asarray([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0]
])

# voxel size (for downsampling point cloud)
DTU_VOXEL_SIZE = 5
