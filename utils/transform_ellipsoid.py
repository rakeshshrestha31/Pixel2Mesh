#!/usr/bin/env python

import sys
import subprocess
import os
import pickle
import copy
import numpy as np

def load_obj(fn, no_normal=False):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()
    print(fn, lines)
    vertices = []; normals = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('vn '):
            normals.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    mesh = dict()
    mesh['faces'] = np.vstack(faces)
    mesh['vertices'] = np.vstack(vertices)

    if (not no_normal) and (len(normals) > 0):
        assert len(normals) == len(vertices), 'ERROR: #vertices != #normals'
        mesh['normals'] = np.vstack(normals)

    return mesh

def write_obj(path, vertices, faces):
    with open(path, 'w') as o:
        for v in vertices:
            o.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for f in faces:
            o.write('f {} {} {}\n'.format(f[0]+1, f[1]+1, f[2]+1))

SCALE = 0.3
TRANSLATION = [0, 0, -0.25]
TRANSFORMATION = np.asarray([
    [SCALE, 0, 0, TRANSLATION[0]],
    [0, SCALE, 0, TRANSLATION[1]],
    [0, 0, SCALE, TRANSLATION[2]],
    [0, 0, 0, 1]
], dtype=np.float32)

def transform_coords(coords):
    coords_homogeneous = np.concatenate(
        (coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)), axis=1
    )
    transformed_coords_homogeneous = np.matmul(
        TRANSFORMATION, coords_homogeneous.transpose()
    )
    return transformed_coords_homogeneous[:3, :].transpose()

def transform_dat(original_ellipsoid_path, final_ellipsoid_path):
    with open(original_ellipsoid_path, 'rb') as f:
        original_dat = pickle.load(f, encoding='latin1')
    coords = original_dat[0]
    transformed_coords = transform_coords(coords)
    final_dat = list(copy.deepcopy(original_dat))
    final_dat[0] = transformed_coords
    final_dat = tuple(final_dat)
    with open(final_ellipsoid_path, 'wb') as f:
        pickle.dump(final_dat, f, protocol=2)

def transform_obj(original_obj_path, final_obj_path):
    original_obj = load_obj(original_obj_path)
    final_obj = copy.deepcopy(original_obj)
    final_obj['vertices'] = transform_coords(original_obj['vertices'])
    write_obj(final_obj_path, final_obj['vertices'], final_obj['faces'])

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ', sys.argv[0],
              '<original_ellipsoid_dir>', '<final_ellipsoid_dir>')
        exit(0)
    original_ellipsoid_dir = sys.argv[1]
    final_ellipsoid_dir = sys.argv[2]
    subprocess.run(['mkdir', '-p',  final_ellipsoid_dir])

    transform_dat(
        os.path.join(original_ellipsoid_dir, 'info_ellipsoid.dat'),
        os.path.join(final_ellipsoid_dir, 'info_ellipsoid.dat')
    )

    # faces: f * 3, original ellipsoid, and two after deformations
    for i in range(1, 4):
        subprocess.run([
            'cp', os.path.join(original_ellipsoid_dir, 'face%d.obj' % i),
            os.path.join(final_ellipsoid_dir, 'face%d.obj' % i)
        ])
        # no need to transform. The face obj don't have vertices, just faces!
        # transform_obj(
        #     os.path.join(original_ellipsoid_dir, 'face%d.obj' % i),
        #     os.path.join(final_ellipsoid_dir, 'face%d.obj' % i)
        # )
