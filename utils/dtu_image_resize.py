#!/usr/bin/env python

import sys
import numpy as np
import argparse
import parse
import pickle
import cv2
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser(description='resize DTU image')
    parser.add_argument('input_image', type=str)
    parser.add_argument('input_ply', type=str)
    parser.add_argument('input_cam_info', type=str)
    parser.add_argument('output_image', type=str)
    # parser.add_argument('output_cam_info', type=str)
    parser.add_argument('output_dat', type=str)
    parser.add_argument('--resize_x', default=224, type=int)
    parser.add_argument('--resize_y', default=224, type=int)
    parser.add_argument('--pcd-size', default=0.55, type=float)
    parser.add_argument('--scale-intrinsics',
                        dest='scale_intrinsics', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')

    return parser.parse_args()

def parse_cam_info(cam_info_file, scale_intrinsics):
    with open(cam_info_file, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(
        ' '.join(lines[1:5]), dtype=np.float32, sep=' '
    ).reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(
        ' '.join(lines[7:10]), dtype=np.float32, sep=' '
    ).reshape((3, 3))
    if scale_intrinsics:
        intrinsics[:2, :] *= 4

    # depth_min & depth_interval: line 11
    depth_max = float(lines[11].split()[0])
    depth_min = float(lines[11].split()[1])
    return {
        'intrinsics': intrinsics, 'extrinsics': extrinsics,
        'depth_min': depth_min, 'depth_max': depth_max
    }

# transformation to align with Pixel2Mesh (shapenet) coordinate frame
T_shapenet_dtu = np.asarray([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0]
])

def process_ply(ply_file, T_cam_world, dat_file, pcd_size, debug):
    pcd = o3d.io.read_point_cloud(ply_file)
    # downsample the cloud
    pcd = pcd.voxel_down_sample(voxel_size=5)

    # find size of the cloud
    points = np.asarray(pcd.points, dtype=np.float32)
    max_points = points.max(axis=0)
    min_points = points.min(axis=0)
    raw_pcd_size = np.linalg.norm(max_points - min_points)

    # transform to cam frame
    pcd = pcd.transform(T_cam_world)

    # resize
    pcd = pcd.scale(pcd_size/raw_pcd_size, center=False)

    # rotate to align with Pixel2Mesh coordinate frame
    pcd = pcd.transform(T_shapenet_dtu)

    # make dat file
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    points_normals = np.concatenate((points, normals), axis=1)
    with open(dat_file, 'wb') as f:
        pickle.dump(points_normals, f, protocol=2)

    if debug:
        np.set_printoptions(precision=3, suppress=True)
        o3d.io.write_point_cloud('/tmp/transformed.ply', pcd)
        print(pcd)
        print(points_normals)
        max_points = points.max(axis=0)
        min_points = points.min(axis=0)
        points_size = max_points - min_points
        print('max:', max_points)
        print('min:', min_points)
        print('size', points_size)

    return pcd

def process_image(args):
    input_image = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)
    output_image = cv2.resize(
        input_image, (args.resize_x, args.resize_y), cv2.INTER_AREA
    )
    # add alpha channel
    b, g, r = cv2.split(output_image)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    output_image = cv2.merge((b, g, r, alpha))
    cv2.imwrite(args.output_image, output_image)
    return {'input_image': input_image, 'output_image': output_image}

def compute_new_intrinsics(input_intrinsics, input_size, output_size):
    # note: the size is [H (y), W(x)]
    output_intrinsics = input_intrinsics.copy()
    size_ratio = \
        tuple([float(output_size[i]) / input_size[i] for i in range(2)])
    output_intrinsics[0, 0] *= size_ratio[1]
    output_intrinsics[0, 2] *= size_ratio[1]
    output_intrinsics[1, 1] *= size_ratio[0]
    output_intrinsics[1, 2] *= size_ratio[0]
    return output_intrinsics

def project_pcd(pcd_local, intrinsics, image_size, image_file):
    points = np.asarray(pcd_local.points)
    colors = np.asarray(pcd_local.colors)
    normalized_points = points.copy()
    normalized_points[:, 0] /= points[:, 2]
    normalized_points[:, 1] /= points[:, 2]
    normalized_points[:, 2] /= points[:, 2]
    projected_coords = np.matmul(intrinsics, normalized_points.transpose())

    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    for i in range(projected_coords.shape[1]):
        u = int(np.round(projected_coords[0, i]))
        v = int(np.round(projected_coords[1, i]))
        if 0 < u < image_size[1] and 0 < v < image_size[0]:
            image[v, u, :] = (colors[i, :] * 255).astype(np.uint8)
    cv2.imwrite(image_file, image)

if __name__ == '__main__':
    args = parse_args()
    images = process_image(args)
    input_cam_info = parse_cam_info(args.input_cam_info, args.scale_intrinsics)
    T_cam_world = input_cam_info['extrinsics']
    pcd = process_ply(
        args.input_ply, T_cam_world,
        args.output_dat, args.pcd_size, args.debug
    )
    output_intrinsics = compute_new_intrinsics(
        input_cam_info['intrinsics'],
        images['input_image'].shape, images['output_image'].shape
    )
    print('new intrinsics\n', output_intrinsics)
    # project_pcd(
    #     pcd, input_cam_info['intrinsics'], images['input_image'].shape,
    #     '/tmp/input_projection.png'
    # )
    project_pcd(
        pcd, output_intrinsics, images['output_image'].shape,
        '/tmp/output_projection.png'
    )
