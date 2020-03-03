import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate
import cv2
import json

import config
from datasets.base_dataset import BaseDataset
from datasets.shapenet import ShapeNet

class ShapeNetDepth(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name, normalization,
                 shapenet_options, intrinsics, nDepths=48):
        super().__init__()
        self.file_root = file_root
        with open(os.path.join(self.file_root, "meta", "shapenet.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}
        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]

        self.normalization = normalization
        self.intrinsics = intrinsics
        self.nDepths = nDepths

    def __getitem__(self, index):
        filename = self.file_names[index]
        label,label_appendix, _ = filename.split("_", maxsplit=3)
        img_path = os.path.join(
            self.file_root, "data_tf/ShapeNetImages/ShapeNetRendering",
            label, label_appendix, "rendering"
        )
        depth_path = img_path.replace("rendering", "rendering_depth")
        camera_meta_data = np.loadtxt(os.path.join(img_path, 'rendering_metadata.txt'))

        imgs = []
        depths = []
        imgs_normalized = []
        proj_matrices = []
        proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
        proj_mat[0, 3, 3] = 1
        proj_mat[1, 2, 2] = 1

        for idx, view in enumerate([0, 6, 7]):
            img_file = os.path.join(img_path, str(view).zfill(2) + '.png')
            img = ShapeNet.read_image(img_file)
            img_normalized = self.normalize_img(img.clone()) \
                                if self.normalization else img

            depth_file = os.path.join(depth_path, str(view).zfill(2) + '.png')
            depth = ShapeNet.read_depth(depth_file)

            proj_mat[0, :4, :4] = self.read_camera_mat(camera_meta_data[view])
            proj_mat[1, :3, :3] = self.intrinsics

            proj_matrices.append(proj_mat.copy())
            imgs_normalized.append(img_normalized)
            imgs.append(img)
            depths.append(depth)

        imgs = np.stack(imgs)
        imgs_normalized = np.stack(imgs_normalized)

        proj_matrices = np.stack(proj_matrices)
        depths = torch.stack(depths, dim=0)
        masks = (depths > 1e-7).float()

        depth_min = 0.1
        depth_interval = 0.025
        depth_values = (np.asarray(range(self.nDepths)).astype(np.float32) * depth_interval) + depth_min

        return {
            "images": imgs_normalized ,
            "images_orig": imgs,
            "labels": self.labels_map[label],
            "filename": filename,
            "proj_matrices": proj_matrices,
            "depth_values": depth_values,
            "depths": depths,
            "masks": masks,
        }

    def read_camera_mat(self, meta_data_file):
        camR, camT = self.cameraMat(meta_data_file)
        camera_mat = np.eye(4, 4, dtype=np.float32)
        camera_mat[:3, :3] = camR
        camera_mat[:3, 3] = -np.matmul(camR, camT)
        return camera_mat

    def __len__(self):
        return len(self.file_names)
