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

if __name__ == '__main__':
    args = parse_args()
    vertices = np.loadtxt(args.xyz_file)[:, :3]
    vertices = np.hstack((np.full([vertices.shape[0],1], 'v'), vertices))
    faces = np.loadtxt(args.faces_file, dtype='|S32')
    mesh = np.vstack((vertices, faces))
    np.savetxt(
        os.path.join(
            args.obj_dir,
            args.xyz_file.split('/')[-1] + args.out_postfix + '.obj'
        ),
        mesh, fmt='%s', delimiter=' '
    )

