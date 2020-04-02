from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize

import config
import numpy as np

class BaseDataset(Dataset):

    def __init__(self):
        self.normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)

    @staticmethod
    def cameraMat(param):
        theta = param[0] * np.pi / 180.0
        camy = param[3] * np.sin(param[1] * np.pi / 180.0)
        lens = param[3] * np.cos(param[1] * np.pi / 180.0)
        camx = lens * np.cos(theta)
        camz = lens * np.sin(theta)
        Z = np.stack([camx, camy, camz])
        x = camy * np.cos(theta + np.pi)
        z = camy * np.sin(theta + np.pi)
        Y = np.stack([x, lens, z])
        X = np.cross(Y, Z)
        cm_mat = np.stack([
            BaseDataset.normal(X), BaseDataset.normal(Y), BaseDataset.normal(Z)
        ])
        return cm_mat, Z

    @staticmethod
    def normal(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return np.divide(v, norm)
