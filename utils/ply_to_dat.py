#!/usr/bin/env python

import sys
import numpy as np
import pickle
import argparse
import open3d

def parse_args():
    parser = argparse.ArgumentParser(description='convert ply to dat')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    point_cloud = open3d.io.read_point_cloud(args.input_file)
    points = np.asarray(point_cloud.points, dtype=np.float32)
    normals = np.asarray(point_cloud.normals, dtype=np.float32)
    points_normals = np.concatenate((points, normals), axis=1)
    with open(args.output_file, 'wb') as f:
        pickle.dump(points_normals, f, protocol=2)

    if args.debug:
        np.set_printoptions(precision=3, suppress=True)
        print(point_cloud)
        print(points)
        print(normals)
        print(points_normals)
        open3d.io.write_point_cloud('/tmp/sample.ply', point_cloud)
