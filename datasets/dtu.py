from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch
import pickle
from PIL import Image
from skimage import io, transform
import config
from datasets.data_io import *
from datasets.base_dataset import BaseDataset
from utils.mesh import Ellipsoid

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(BaseDataset):
    def __init__(self, datapath, listfile, mode, nviews, normalization, ndepths=48, depth_interals_ratio=4, interval_scale=1.06, mesh_pos=[0., 0., 0]):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.depth_interals_ratio = depth_interals_ratio

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        # if self.debug_scan2:
        #     self.cache = {}
        #     pkl_filename = os.path.join(self.datapath, "Points/scan2.dat")
        #     print("pkl_filename", self.datapath, pkl_filename)
        #     with open(pkl_filename) as f:
        #         self.cache["cloud"] = pickle.load(open(pkl_filename, 'rb'), encoding="latin1")


    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale * self.depth_interals_ratio
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = np.array(Image.open(filename)).astype(np.float32)
        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255
        # img = cv2.resize(img, (img.shape[0]//4, img.shape[1]//4), cv2.INTER_LINEAR)
        img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        # img_normalized = self.normalize_img(img) if self.normalization else img
        img_normalized = img / 255.
        img = img_normalized

        return img, img_normalized

    def read_mask(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = cv2.resize(np_img, (56, 56), interpolation=cv2.INTER_NEAREST)
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        h, w = depth.shape
        depth = cv2.resize(depth, (56, 56), interpolation=cv2.INTER_NEAREST)
        return depth

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        imgs_normalized = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified_resized/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train_resized/{:0>8}_cam.txt').format(vid)
            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            img, img_normalized = self.read_img(img_filename)
            imgs.append(img)
            imgs_normalized.append(img_normalized)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                pkl_filename = os.path.join(self.datapath, "Points_resized/{}_train/view_{}.dat".format(scan, vid))
                depth = self.read_depth(depth_filename)
                depth *= config.DTU_RESCALE_FACTOR
                mask = self.read_mask(mask_filename)
                with open(pkl_filename) as f:
                    data = pickle.load(open(pkl_filename, 'rb'), encoding="latin1")
                    pts, normals = data[:, :3], data[:, 3:]

        pts -= np.array(self.mesh_pos)
        imgs = np.stack(imgs)
        imgs_normalized = np.stack(imgs_normalized)
        proj_matrices = np.stack(proj_matrices)
        if imgs.shape[0] == 1:
            imgs = np.squeeze(imgs, 0)
            imgs_normalized = np.squeeze(imgs_normalized, 0)
        length = pts.shape[0]
        return {"images": imgs_normalized,
                "images_orig": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth,
                "mask": mask,
                "depth_values": depth_values,
                "points": pts,
                "normals": normals,
                "length": length,
                "filename": scan,
                "labels": 0,
                "idx": idx
                }

    def get_initial_mesh(self, idx):
        meta = self.metas[idx]
        scan, _, ref_view, _ = meta

        # TODO: find a better way than to read the cam file twice
        # cuz already read in __getitem__
        proj_mat_filename = os.path.join(
            self.datapath, 'Cameras/train_resized/{:0>8}_cam.txt'
        ).format(ref_view)
        _, extrinsics, _, _ = self.read_cam_file(proj_mat_filename)
        return get_initial_mesh(
            get_initial_mesh_filename(self.datapath, scan),
            self.mesh_pos, extrinsics
        )

def get_initial_mesh_filename(datapath, scan):
    scan_idx = int(scan[4:])
    initial_mesh_filename = os.path.join(
        datapath,
        'initial_mesh/stl{:0>3}_total/info_ellipsoid.dat'.format(scan_idx)
    )
    return initial_mesh_filename

def get_initial_mesh(initial_mesh_filename, mesh_pos, extrinsics):
    initial_mesh = Ellipsoid(mesh_pos, initial_mesh_filename)

    extrinsics = extrinsics.copy()
    extrinsics = torch.from_numpy(extrinsics) \
        .type(initial_mesh.coord.type())
    if initial_mesh.coord.is_cuda:
        extrinsics = extrinsics.cuda(initial_mesh.coord.get_device())

    # transform initial mesh to ref_view frame
    initial_mesh.coord = transform_coords(initial_mesh.coord, extrinsics)
    return initial_mesh

def transform_coords(coords, transformation):
    coords_homogeneous = torch.cat((
        coords,
        torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
    ), dim=1)
    transformed_coords_homogeneous = torch.mm(
        transformation, coords_homogeneous.permute(1, 0)
    )
    return transformed_coords_homogeneous[:3, :].permute(1, 0)

