#!/usr/bin/env python

import sys
import numpy as np
import pickle
import argparse
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser(description='convert dat to ply')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    with open(args.input_file, 'rb') as f:
        points_normals = pickle.load(f, encoding='latin1')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_normals[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(points_normals[:, 3:6])
    o3d.io.write_point_cloud(args.output_file, pcd)

    if args.debug:
        np.set_printoptions(precision=3, suppress=True)
        print(pcd)
        print(points_normals)
        max_points = points_normals.max(axis=0)
        min_points = points_normals.min(axis=0)
        print('max:', max_points)
        print('min:', min_points)
        print('size', max_points - min_points)
