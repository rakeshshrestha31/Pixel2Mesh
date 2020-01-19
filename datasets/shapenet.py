import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate
import cv2

import config
from datasets.base_dataset import BaseDataset


class ShapeNet(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options, intrinsics, is_train , nDepths=48):
        super().__init__()
        self.file_root = file_root
        with open(os.path.join(self.file_root, "meta", "shapenet.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}
        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.tensorflow = "_tf" in file_list_name # tensorflow version of data
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.is_train = is_train
        self.intrinsics = intrinsics
        self.nDepths = nDepths

    def __getitem__(self, index):

        filename = self.file_names[index]
        label,label_appendix, _ = filename.split("_", maxsplit=3)
        if self.is_train:
            pkl_path = os.path.join(self.file_root, "data_tf/p2mppdata/train", filename)
        else:
            pkl_path = os.path.join(self.file_root, "data_tf/p2mppdata/test", filename)
        with open(pkl_path) as f:
            data = pickle.load(open(pkl_path, 'rb'), encoding="latin1")
            if self.is_train:
                data = data.astype(np.float32)
            else:
                data = data[1].astype(np.float32)
        pts, normals = data[:, :3], data[:, 3:]  # pts,normals -> <class 'tuple'>: (7982, 3),
        img_path = os.path.join(self.file_root, "data_tf/ShapeNetImages/ShapeNetRendering", label, label_appendix,
                                "rendering")
        depth_path = img_path.replace("rendering", "rendering_depth")

        camera_meta_data = np.loadtxt(os.path.join(img_path, 'rendering_metadata.txt'))
        imgs = []
        imgs_normalized = []
        proj_matrices = []
        proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
        for idx, view in enumerate([0, 6, 7]):
            img = io.imread(os.path.join(img_path, str(view).zfill(2) + '.png'))
            img[np.where(img[:, :, 3] == 0)] = 255
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            img = img[:, :, :3].astype(np.float32)

            img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
            img_normalized = self.normalize_img(img.clone()) if self.normalization else img

            if idx == 0:
                depth_file = os.path.join(depth_path, str(view).zfill(2) + '.png')
                if os.path.isfile(depth_file):
                    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
                    depth = cv2.resize(depth, (config.IMG_SIZE//4, config.IMG_SIZE//4))
                    depth = depth.astype(np.float32) / 1000
                    depth = torch.from_numpy(depth)
                else:
                    depth = torch.from_numpy(
                        # np.ones((config.IMG_SIZE // 4, config.IMG_SIZE // 4), dtype=np.float32)
                        np.random.random((config.IMG_SIZE//4, config.IMG_SIZE//4)).astype(np.float32)
                    )

            camR, camT = self.cameraMat(camera_meta_data[view])
            proj_mat[0, :3, :3] = camR
            proj_mat[0, :3, 3] = -np.matmul(camR, camT)
            proj_mat[0, -1, -1] = 1
            proj_mat[1, :3, :3] = self.intrinsics
            proj_mat[1, 2, 2] = 1
            proj_matrices.append(proj_mat.astype(np.float32))
            imgs_normalized.append(img_normalized)
            imgs.append(img)

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        imgs = np.stack(imgs)
        imgs_normalized = np.stack(imgs_normalized)
        proj_matrices = np.stack(proj_matrices)

        mask = (depth > 1e-7).float()

        depth_min = 0.1
        depth_interval = 0.025
        depth_values = (np.asarray(range(self.nDepths)).astype(np.float32) * depth_interval) + depth_min
        # depth_values = np.arange(depth_min, depth_interval * self.nDepths + depth_min, depth_interval,
        #                          dtype=np.float32)
        return {
            "images": imgs_normalized ,
            "images_orig": imgs,
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": filename,
            "length": length,
            "proj_matrices": proj_matrices,
            "depth_values": depth_values,
            "depth": depth,
            "mask": mask
        }

    def __len__(self):
        return len(self.file_names)


class ShapeNetImageFolder(BaseDataset):

    def __init__(self, folder, normalization, shapenet_options):
        super().__init__()
        self.normalization = normalization
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.file_list = []
        for fl in os.listdir(folder):
            file_path = os.path.join(folder, fl)
            # check image before hand
            try:
                if file_path.endswith(".gif"):
                    raise ValueError("gif's are results. Not acceptable")
                Image.open(file_path)
                self.file_list.append(file_path)
            except (IOError, ValueError):
                print("=> Ignoring %s because it's not a valid image" % file_path)

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "filepath": self.file_list[item]
        }

    def __len__(self):
        return len(self.file_list)


def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    def shapenet_collate(batch):
        if len(batch) > 1:
            all_equal = True
            for t in batch:
                if t["length"] != batch[0]["length"]:
                    all_equal = False
                    break
            points_orig, normals_orig = [], []
            if not all_equal:
                for t in batch:
                    pts, normal = t["points"], t["normals"]
                    length = pts.shape[0]
                    choices = np.resize(np.random.permutation(length), num_points)
                    t["points"], t["normals"] = pts[choices], normal[choices]
                    points_orig.append(torch.from_numpy(pts))
                    normals_orig.append(torch.from_numpy(normal))
                ret = default_collate(batch)
                ret["points_orig"] = points_orig
                ret["normals_orig"] = normals_orig
                return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret

    return shapenet_collate
