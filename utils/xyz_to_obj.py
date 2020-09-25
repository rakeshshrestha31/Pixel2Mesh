#/usr/bin/env python
##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='xyz to obj')
    parser.add_argument('xyz_file', type=str)
    parser.add_argument('faces_file', type=str)
    parser.add_argument('obj_dir', type=str)
    parser.add_argument('out_postfix', type=str, default='')
    return parser.parse_args()


def convert(xyz_file, faces_file, obj_dir, out_postfix):
    vertices = np.loadtxt(xyz_file)[:, :3]
    vertices = np.hstack((np.full([vertices.shape[0],1], 'v'), vertices))
    faces = np.loadtxt(faces_file, dtype='|S32')
    mesh = np.vstack((vertices, faces))
    os.makedirs(obj_dir, exist_ok=True)
    np.savetxt(
        os.path.join(
            obj_dir,
            xyz_file.split('/')[-1] + out_postfix + '.obj'
        ),
        mesh, fmt='%s', delimiter=' '
    )


if __name__ == '__main__':
    args = parse_args()
    convert(args.xyz_file, args.faces_file, args.obj_dir, args.out_postfix)

