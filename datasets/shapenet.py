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
import h5py

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
        # read best views
        self.best_views = self.read_best_subsets_file(
            os.path.join(self.file_root, "meta", "best_subsets_max_fov.hdf5")
        )

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
        depths = []
        imgs_normalized = []
        proj_matrices = []
        proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
        proj_mat[0, 3, 3] = 1
        proj_mat[1, 2, 2] = 1

        view_key = '%s/%s' % (label, label_appendix)
        views = self.best_views['subsets'][view_key]
        for idx, view in enumerate(views):
            img_file = os.path.join(img_path, str(view).zfill(2) + '.png')
            img = self.read_image(img_file)
            img_normalized = self.normalize_img(img.clone()) \
                                if self.normalization else img

            depth_file = os.path.join(depth_path, str(view).zfill(2) + '.png')
            depth = self.read_depth(depth_file)

            proj_mat[0, :4, :4] = self.read_camera_mat(camera_meta_data[view])
            proj_mat[1, :3, :3] = self.intrinsics

            proj_matrices.append(proj_mat.copy())
            imgs_normalized.append(img_normalized)
            imgs.append(img)
            depths.append(depth)

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        # assignment of each point to a view (at different point densities)
        points_to_view_assignments = {
            'assignment%d' % i: \
                np.asarray(self.best_views['assignment%d' % i][view_key],
                           dtype=np.int32)
            for i in range(3)
        }

        imgs = np.stack(imgs)
        imgs_normalized = np.stack(imgs_normalized)
        proj_matrices = np.stack(proj_matrices)

        depths = torch.stack(depths, dim=0)
        masks = (depths > 1e-7).float()

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
            "depths": depths,
            "masks": masks,
            **points_to_view_assignments
        }

    @staticmethod
    def read_best_subsets_file(h5_file):
        with h5py.File(h5_file, 'r') as f:
            files = f['files'][()]
            files = [filename.decode('utf-8') for filename in files]
            subsets = f['subsets'][()]
            subsets = {
                file_key: subset
                for file_key, subset in zip(files, subsets.tolist())
            }

            assignments = [f['assignment%d'%i][()] for i in range(3)]
            assignments = {
                'assignment%d' % i: {
                    file_key: assign
                    for file_key, assign in zip(files, assignment.tolist())
                } for i, assignment in enumerate(assignments)
            }
            return {'subsets': subsets, **assignments}


    ##
    #  @return list with points assignments at all point densities
    @staticmethod
    def get_points_assignments(input_batch):
        points_assignments_keys = ['assignment%d' % i for i in range(3)]
        points_assignments = [
            input_batch[key] for key in points_assignments_keys
        ]
        return points_assignments


    @staticmethod
    def read_image(img_file):
        img = io.imread(img_file)
        img[np.where(img[:, :, 3] == 0)] = 255
        img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        return img

    @staticmethod
    def read_depth(depth_file):
        if os.path.isfile(depth_file):
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (config.IMG_SIZE//4, config.IMG_SIZE//4))
            depth = depth.astype(np.float32) / 1000
            depth = torch.from_numpy(depth)
            return depth
        else:
            print('depth file not found:', depth_file)
            exit(1)

    def read_camera_mat(self, meta_data_file):
        camR, camT = self.cameraMat(meta_data_file)
        camera_mat = np.eye(4, 4, dtype=np.float32)
        camera_mat[:3, :3] = camR
        camera_mat[:3, 3] = -np.matmul(camR, camT)
        return camera_mat

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
