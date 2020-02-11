#!/usr/bin/env python

# standard library imports
import os
import cv2
import numpy as np
from skimage import io, transform
import itertools
import json
import tqdm

# custom imports
import config
from datasets.base_dataset import BaseDataset

# torch imports
import torch
import torchvision

base_dataset = BaseDataset()

def check_baselines(file_root):
    # dictionary from scene to best subset
    best_subsets = {}
    best_images = []
    for is_train in [True, False]:
        file_list_name = 'train_list_p2mpp' if is_train \
                                else 'test_list_p2mpp'
        # Read file list
        with open(os.path.join(file_root, "meta", file_list_name + ".txt"), "r") as fp:
            file_names = fp.read().split("\n")[:-1]

        for filename in tqdm.tqdm(file_names):
            label,label_appendix, _ = filename.split("_", maxsplit=3)

            img_path = os.path.join(
                file_root, "data_tf/ShapeNetImages/ShapeNetRendering",
                label, label_appendix, "rendering"
            )
            camera_meta_data = np.loadtxt(
                os.path.join(img_path, 'rendering_metadata.txt')
            )

            imgs = []
            T_world_cams = []

            for view in range(24):
                img = read_rgb_image(
                    os.path.join(img_path, str(view).zfill(2) + '.png')
                )
                T_cam_world = read_camera_pose(camera_meta_data[view])
                T_world_cam = np.linalg.inv(T_cam_world)

                imgs.append(img)
                T_world_cams.append(T_world_cam)

            best_subset = find_best_subset(T_world_cams)
            best_subsets['%s/%s'%(label, label_appendix)] = best_subset
            best_images.extend([imgs[i] for i in best_subset])

            # if len(best_subsets) > 10:
            #     break

    # best_images_tensor = torch.stack(best_images, 0)
    # grid_images_tensor = torchvision.utils.make_grid(best_images_tensor, nrow=3)
    # grid_images_np = (grid_images_tensor.permute(1, 2, 0) * 255).numpy() \
    #                         .astype(np.uint8)
    # cv2.imwrite('/tmp/best_subsets.png', grid_images_np)

    with open('/tmp/best_subsets.json', 'w') as f:
        json.dump(best_subsets, f, indent=4)


def find_best_subset(T_world_cams):
    all_subsets = list(itertools.combinations(
        range(len(T_world_cams)), 3
    ))
    subset_baselines = []
    for subset_indices in all_subsets:
        total_baseline = 0
        for subset_pair_indices in \
                itertools.combinations(subset_indices, 2):
            pose0 = T_world_cams[subset_pair_indices[0]]
            pose1 = T_world_cams[subset_pair_indices[1]]
            baseline_vector = pose1[:3, 3] - pose0[:3, 3]
            baseline = np.linalg.norm(baseline_vector)
            total_baseline += baseline
        subset_baselines.append(total_baseline)

    min_subset_idx = np.argmin(subset_baselines)
    return all_subsets[min_subset_idx]


def read_camera_pose(meta_data):
    global base_dataset
    camR, camT = base_dataset.cameraMat(meta_data)
    T_cam_world = np.eye(4, 4, dtype=np.float32)
    T_cam_world[:3, :3] = camR
    T_cam_world[:3, 3] =  -np.matmul(camR, camT)
    return T_cam_world


def read_rgb_image(img_path):
    img = io.imread(img_path)
    img[np.where(img[:, :, 3] == 0)] = 255
    img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = img[:, :, :3].astype(np.float32)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

    return img


if __name__ == '__main__':
    check_baselines(config.SHAPENET_ROOT)

