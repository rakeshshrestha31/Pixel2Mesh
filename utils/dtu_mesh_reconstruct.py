#!/usr/bin/env python

import argparse
import sys
import subprocess
import os
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruction-exec', type=str, required=True)
    parser.add_argument('--trimmer-exec', type=str, required=True)
    parser.add_argument('--decimation-exec', type=str, required=True)
    parser.add_argument('--points-dir', type=str, required=True)
    parser.add_argument('--new-points-dir', type=str, required=True)
    parser.add_argument('--dense-reconstruction-depth', type=int, default=10)
    parser.add_argument('--trim', type=int, default=8)
    parser.add_argument('--tmp-reconstruction-dir', type=str, default='/tmp')
    parser.add_argument('--num-faces', type=int, default=350)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    subprocess.run(['mkdir', '-p', args.new_points_dir])
    dense_reconstruction_filename = os.path.join(args.tmp_reconstruction_dir,
                                                 'dense_construction.ply')
    abs_output_ply_filename = os.path.join(args.tmp_reconstruction_dir,
                                           'sparse_reconstruction.ply')
    tmp_mesh = '/tmp/tmp.ply'
    for relative_filename in os.listdir(args.points_dir):
        abs_input_ply_filename = os.path.join(args.points_dir,
                                              relative_filename)
        abs_output_obj_filename = os.path.join(args.new_points_dir,
                                               relative_filename[:-4] + '.obj')

        if relative_filename[-4:] != '.ply' \
           or not os.path.isfile(abs_input_ply_filename):
            continue

        # TODO: only for debug, remove this
        if relative_filename not in ['stl002_total.ply', 'stl004_total.ply']:
            continue

        print('reconstructing', abs_input_ply_filename)
        subprocess.run([
            args.reconstruction_exec,
            '--in', abs_input_ply_filename,
            '--out', tmp_mesh,
            '--depth', str(args.dense_reconstruction_depth),
            # '--pointWeight', str(0), # original (unsceen) version
            # '--degree', str(1), # 1-degree B-spline for efficiency
            '--normals', '--colors', # maintain normals and colors
            '--density'
        ])

        print('trimming', tmp_mesh)
        subprocess.run([
            args.trimmer_exec,
            '--in', tmp_mesh,
            '--out', dense_reconstruction_filename,
            '--trim', str(args.trim),
        ])

        print('decimating', dense_reconstruction_filename)
        subprocess.run([
            args.decimation_exec, dense_reconstruction_filename,
            abs_output_ply_filename, str(args.num_faces),
            # '-Ty', # preserve topology
            '-C' # remove duplicate or unreferenced vertices
        ])

        pcd = o3d.io.read_triangle_mesh(abs_output_ply_filename)
        o3d.io.write_triangle_mesh(abs_output_obj_filename, pcd)
