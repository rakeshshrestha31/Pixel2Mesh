#!/usr/bin/env python

# standard library imports
import open3d as o3d
import os
import cv2
import numpy as np
from skimage import io, transform
import h5py
import tqdm

# custom imports
import config
from datasets.base_dataset import BaseDataset
from utils.mesh import Ellipsoid
from models.layers.gpooling import GUnpooling

# torch imports
import torch

base_dataset = BaseDataset()

colors = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
)

def select_maximum_fov(file_root, debug=False):
    best_subsets = {
        'files': [],
        # for each file, there is a 3-tuple of best views
        'subsets': [],
        # for each file, there are three sets of points at different densities
        # each have different assignments
        'assignments': [[], [], []]
    }

    # create ellipsoid
    ellipsoid = Ellipsoid([0.0, 0.0, 0.0])
    ellipsoid_pts = get_ellipsoid_points(ellipsoid)

    if debug:
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

            if debug:
                imgs = []
            T_cams_world = []

            for view in range(24):
                T_cam_world = read_camera_pose(camera_meta_data[view])
                T_cams_world.append(T_cam_world)

                if debug:
                    img = read_rgb_image(
                        os.path.join(img_path, str(view).zfill(2) + '.png')
                    )
                    imgs.append(img)

            best_subset = find_best_subset(T_cams_world)
            key = '%s/%s'%(label, label_appendix)
            best_T_cams_world = [T_cams_world[i] for i in best_subset]
            assignments = [
                assign_points(pts, best_T_cams_world)
                for pts in ellipsoid_pts
            ]

            best_subsets['files'].append(key.encode("ascii", "ignore"))
            best_subsets['subsets'].append(best_subset)
            for i, assignment in enumerate(assignments):
                best_subsets['assignments'][i].append(assignment.cpu().tolist())

            if debug:
                best_images.extend([imgs[i] for i in best_subset])
                for pts_idx, pts in enumerate(ellipsoid_pts):
                    draw_assignments(
                        pts, assignments[pts_idx], len(best_T_cams_world),
                        (label, label_appendix, str(pts_idx))
                    )
                    draw_cameras(best_T_cams_world, (label, label_appendix))
                if len(best_subsets['files']) > 10:
                    break

    return write_best_subsets(best_subsets)


def write_best_subsets(best_subsets):
    num_files = len(best_subsets['files'])
    max_file_len = np.max([len(i) for i in best_subsets['files']])

    with h5py.File('/tmp/best_subsets_max_fov.hdf5', 'w') as h5_file:
        h5_files = h5_file.create_dataset(
            'files', (num_files,), 'S%d' % max_file_len,
            best_subsets['files']
        )
        h5_subsets = h5_file.create_dataset(
            "subsets", (num_files, 3), 'i', best_subsets['subsets']
        )
        for i, assignment in enumerate(best_subsets['assignments']):
            h5_assignments = h5_file.create_dataset(
                "assignment%d" % i, (num_files, len(assignment[0])),
                'i', assignment
            )


##
def draw_assignments(points, assignments, num_cameras, labels):
    global colors
    assert(assignments.size(0) == points.size(0))

    colors = colors.type(points.dtype).to(get_tensor_device(points))

    one_hot_assignment = to_one_hot(assignments, num_cameras)
    one_hot_assignment = one_hot_assignment.type(points.dtype)
    assignment_colors = one_hot_assignment.matmul(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(assignment_colors.cpu().numpy())

    # scale for better visualization
    pcd.scale(5)
    o3d.io.write_point_cloud(
        '/tmp/{}_points.ply'.format('_'.join(labels)), pcd
    )


## adapted from https://github.com/raulmur/ORB_SLAM2/blob/
#               f2e6f51cdc8d067655d90a78c06261378e07e8f3/src/MapDrawer.cc
def draw_cameras(T_cams_world, labels):
    global colors

    global_cam_frustum = get_global_camera_frustum()

    # compute vertices in global frame
    T_world_cams = [np.linalg.inv(i) for i in T_cams_world]
    cam_frustum_vertices = [
        unmake_homogeneous(
            np.matmul(make_homogeneous_array(global_cam_frustum['vertices']),
                      T.transpose())
        )
        for T in T_world_cams
    ]

    # compute faces offsetted by right amount
    vertices_len = np.cumsum([i.shape[0] for i in cam_frustum_vertices])
    vertices_offset = np.zeros(vertices_len.shape)
    vertices_offset[1:] = vertices_len[0:-1]
    cam_frustum_faces = [
        global_cam_frustum['faces'] + offset
        for offset in vertices_offset
    ]

    cam_frustum_colors = [
        colors[i, :].unsqueeze(0) \
            .expand(global_cam_frustum['vertices'].shape[0], -1).cpu().numpy()
        for i in range(len(T_cams_world))
    ]

    cam_frustum_vertices = np.concatenate(cam_frustum_vertices, axis=0)
    cam_frustum_faces = np.concatenate(cam_frustum_faces, axis=0)
    cam_frustum_colors = np.concatenate(cam_frustum_colors, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(cam_frustum_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(cam_frustum_faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(cam_frustum_colors)

    o3d.io.write_triangle_mesh(
        '/tmp/{}_frustums.ply'.format('_'.join(labels)), mesh, write_ascii=True
    )


def get_global_camera_frustum():
    camera_size = 0.1
    w = camera_size
    h = w * 0.75
    z = -w * 0.6
    cam_frustum_mesh = {
        'vertices': np.array([
            [0, 0, 0],
            [w, h, z],
            [-w, h, z],
            [-w, -h, z],
            [w, -h, z]
        ]),
        'faces': np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 1, 4],
            [1, 2, 3],
            [3, 4, 1]
        ], dtype=np.uint64)
    }
    return cam_frustum_mesh


##
#  @param tensor 1D long tensor with indices
#  @param n number of classes
def to_one_hot(tensor, n):
    one_hot = torch.zeros(
        (tensor.size(0), n),
        dtype=tensor.dtype, device=get_tensor_device(tensor)
    )
    one_hot[range(tensor.size(0)), tensor] = 1
    return one_hot


def get_tensor_device(tensor):
    return tensor.get_device() if tensor.is_cuda else torch.device('cpu')


##
#  @param points tensor nx3
#  @return tensor nx4
def make_homogeneous_tensor(points: torch.Tensor):
    return torch.cat(
        (
            points, torch.ones((points.size(0), 1),
                               dtype=points.dtype,
                               device=get_tensor_device(points))
        ), 1
    )


##
#  @param points array nx3
#  @return array nx4
def make_homogeneous_array(points: np.ndarray):
    return np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1
    )


##
#  @param points tensor nx4
#  @return tensor nx3
def unmake_homogeneous(points):
    return points[:, :3]


##
#  @param points tensor nx3
#  @return tensor nx1 with index of T_cams_world for assignment
def assign_points(points, T_cams_world):
    T_cams_world_transpose = [
        torch.from_numpy(i).transpose(1, 0) if type(i) == np.ndarray \
            else i.transpose(1, 0)
        for i in T_cams_world
    ]
    # convert to local camera frame
    points_locals_list = [
        unmake_homogeneous(make_homogeneous_tensor(points).matmul(T_transpose))
        for T_transpose in T_cams_world_transpose
    ]
    # dims: num_cams x num_points x num_point_dim (num_point_dim=3)
    points_locals = torch.stack(points_locals_list, 0)

    # find distance from camera
    points_distances = points_locals.norm(dim=-1)

    # dims: num_cams x num_point_dim
    points_centroids = points_locals.mean(dim=1)
    num_points = points_locals.size(1)
    points_visibilities = points_locals[:, :, 2] \
        > points_centroids[:, 2].unsqueeze(1).expand(-1, num_points)

    # find angle from camera
    points_angles = \
        torch.atan2(points_locals[:, :, 0], points_locals[:, :, 2]) \
            + torch.atan2(points_locals[:, :, 1], points_locals[:, :, 2])

    # infinite distance of points that are invisible
    visibility_offset = ((~points_visibilities).type(points_distances.dtype) * 1e16)
    points_distances += visibility_offset
    points_angles += visibility_offset

    # assignments = points_distances.argmin(dim=0)
    assignments = points_angles.argmin(dim=0)
    return assignments


def find_best_subset(T_world_cams):
    # TODO: choose a subset that maximizes FoV
    return [0, 6, 7]


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


def get_ellipsoid_points(ellipsoid):
    unpooling = [
        GUnpooling(ellipsoid.unpool_idx[0]),
        GUnpooling(ellipsoid.unpool_idx[1])
    ]
    ellipsoid_pts = [None for _ in range(3)]
    ellipsoid_pts[0] = ellipsoid.coord
    ellipsoid_pts[1] = unpooling[0](ellipsoid_pts[0].unsqueeze(0)).squeeze(0)
    ellipsoid_pts[2] = unpooling[1](ellipsoid_pts[1].unsqueeze(0)).squeeze(0)
    return ellipsoid_pts


if __name__ == '__main__':
    select_maximum_fov(config.SHAPENET_ROOT)

