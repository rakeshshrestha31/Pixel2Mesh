#!/usr/bin/env python

import sys
import os
import copy
import itertools
import numpy as np
import argparse
import parse
import pickle
import cv2
import open3d as o3d
import glob
# from .ply2img import gen_trans_pcl
import config

CAM_INFO_FORMAT = \
    'extrinsic\n' \
    '{T00:f} {T01:f} {T02:f} {T03:f} \n' \
    '{T10:f} {T11:f} {T12:f} {T13:f} \n' \
    '{T20:f} {T21:f} {T22:f} {T23:f} \n' \
    '{T30:f} {T31:f} {T32:f} {T33:f} \n\n' \
    'intrinsic\n' \
    '{K00:f} {K01:f} {K02:f} \n' \
    '{K10:f} {K11:f} {K12:f} \n' \
    '{K20:f} {K21:f} {K22:f} \n\n' \
    '{depth_max:f} {depth_min:f}'


def parse_cam_info(cam_info_file):
    with open(cam_info_file, 'r') as f:
        cam_info_string = f.read()
        parsed = parse.parse(
            CAM_INFO_FORMAT, cam_info_string
        )
        return parsed


def get_cam_transform(cam_info):
    return np.asarray([
        [cam_info['T00'], cam_info['T01'], cam_info['T02'], cam_info['T03']],
        [cam_info['T10'], cam_info['T11'], cam_info['T12'], cam_info['T13']],
        [cam_info['T20'], cam_info['T21'], cam_info['T22'], cam_info['T23']],
        [cam_info['T30'], cam_info['T31'], cam_info['T32'], cam_info['T33']]
    ])

def update_extrinsics(cam_info, extrinsics):
    for i, j in itertools.product(range(4), range(4)):
        cam_info['T%d%d' % (i, j)] = extrinsics[i, j]

def process_ply(ply_file, input_cam_info,
                dat_file, output_cam_file, scale, debug):
    T_cam_world = get_cam_transform(input_cam_info)
    pcd = o3d.io.read_point_cloud(ply_file)
    # downsample the cloud
    # o3d.visualization.draw_geometries([pcd])
    pcd = pcd.voxel_down_sample(voxel_size=config.DTU_VOXEL_SIZE)
    # o3d.visualization.draw_geometries([pcd])


    # transform point cloud to shapenet frame and correct scale
    transformation = np.linalg.multi_dot((config.T_shapenet_dtu, T_cam_world))
    pcd = pcd.transform(transformation)
    pcd.scale(scale, center=False)

    # find size of the cloud
    points = np.asarray(pcd.points, dtype=np.float32)
    max_points = points.max(axis=0)
    min_points = points.min(axis=0)
    raw_pcd_size = np.linalg.norm(max_points - min_points)
    print('pcd size:', raw_pcd_size)

    # make dat file
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    points_normals = np.concatenate((points, normals), axis=1)
    with open(dat_file, 'wb') as f:
        pickle.dump(points_normals, f, protocol=2)

    # make ply file (just for debugging, not used by network)
    o3d.io.write_point_cloud(dat_file.replace('.dat', '.ply'), pcd)

    # make new cam_file
    output_T_cam_world = T_cam_world.copy()
    output_T_cam_world[:3, 3] *= scale
    # transform the extrinsics to shapenet frame from dtu
    output_T_cam_world = np.linalg.multi_dot((
        config.T_shapenet_dtu, output_T_cam_world,
        np.linalg.inv(config.T_shapenet_dtu)
    ))
    output_cam_info = copy.deepcopy(input_cam_info).named
    update_extrinsics(output_cam_info, output_T_cam_world)
    with open(output_cam_file, 'w') as f:
        f.write(CAM_INFO_FORMAT.format(**output_cam_info))

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
        print('size', np.linalg.norm(points_size))

def process_image(input_image_path, output_image_path):
    print("input_image_path:", input_image_path)
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    output_image = cv2.resize(
        input_image, (224, 224), cv2.INTER_AREA
    )
    # add alpha channel
    b, g, r = cv2.split(output_image)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    output_image = cv2.merge((b, g, r, alpha))
    cv2.imwrite(output_image_path, output_image)
    return output_image

def parse_args():
    parser = argparse.ArgumentParser(description='resize DTU image')
    parser.add_argument('rectified_images_dir', type=str)
    parser.add_argument('points_dir', type=str)
    parser.add_argument('cams_dir', type=str)
    parser.add_argument('--rescale-factor', default=config.DTU_RESCALE_FACTOR,
                        type=float)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # scans_dir = glob.glob(args.rectified_images_dir + '/*')
    scans_dir = [
        os.path.join(args.rectified_images_dir, 'scan2'),
        os.path.join(args.rectified_images_dir, 'scan4')
    ]

    for scan_iter in range(len(scans_dir)):
        scan = scans_dir[scan_iter].split('Rectified/scan')[-1].split('_train')[0]
        ply_file = (args.points_dir + "/stl{:0>3}_total.ply").format(scan)
        for file_index in range(0, 49):
            for light_cond in range(0, 7):
                rgb_file = (args.rectified_images_dir + "/scan{}_train/rect_{:0>3}_{}_r5000.png").format(
                    scan, file_index+1, light_cond)
                output_dat_file = rgb_file.replace("/Rectified/", "/Points_resized/").split("/rect")[0]+"/view_{}".format(file_index)+".dat"
                output_dat_dir = output_dat_file.split("/view")[0]

                os.makedirs(output_dat_dir, exist_ok=True)

                output_rgb_file = rgb_file.replace("Rectified", "Rectified_resized")
                output_rgb_dir = output_rgb_file.split("/rect")[0]
                os.makedirs(output_rgb_dir, exist_ok=True)

                process_image(rgb_file,output_rgb_file)
            input_cam_file = args.cams_dir + '/train/{:0>8}_cam.txt'.format(file_index)
            output_cam_file = args.cams_dir \
                         + '/train_resized/{:0>8}_cam.txt'.format(file_index)
            os.makedirs(os.path.dirname(output_cam_file), exist_ok=True)

            input_cam_info = parse_cam_info(input_cam_file)
            process_ply(
                ply_file, input_cam_info,
                output_dat_file, output_cam_file, args.rescale_factor, True # False
            )
        # one in world frame (just for debugging, not used by network)
        update_extrinsics(input_cam_info.named, np.eye(4))
        output_dat_file = rgb_file \
            .replace("/Rectified/", "/Points_resized/") \
            .split("/rect")[0]+"/view_world" \
            .format(file_index)+".dat"
        process_ply(
            ply_file, input_cam_info,
            output_dat_file,
            os.path.join(args.cams_dir, 'train_resized', 'world.txt'),
            args.rescale_factor, True # False
        )


# if __name__ == '__main__':
#     args = parse_args()
#     process_image(args)
#     input_cam_info = parse_cam_info(args.input_cam_info)
#     T_cam_world = get_cam_transform(input_cam_info)
#     process_ply(
#         args.input_ply, T_cam_world,
#         args.output_dat, args.pcd_size, args.debug
#     )
