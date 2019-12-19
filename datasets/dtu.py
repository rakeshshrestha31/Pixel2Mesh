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


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(BaseDataset):
    def __init__(self, datapath, listfile, mode, nviews, normalization, debug_scan2=False, ndepths=192, interval_scale=1.06,  mesh_pos=[0., 0., -0.8]):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.normalization = normalization
        self.debug_scan2 = debug_scan2
        self.mesh_pos = mesh_pos

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
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
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

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

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
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

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
                # "depth": depth,
                "depth_values": depth_values,
                "points": pts,
                "normals": normals,
                "length": length,
                "filename": scan,
                "labels": 0
                }