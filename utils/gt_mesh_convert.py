#/usr/bin/env python
##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

import numpy as np
import argparse
import trimesh

from datasets.base_dataset import BaseDataset
P2M_SCALE_FACTOR = 0.57

def parse_args():
    parser = argparse.ArgumentParser(description='gt convert')
    parser.add_argument('original_gt_mesh', type=str)
    parser.add_argument('rendering_metadata', type=str)
    parser.add_argument('converted_mesh', type=str)
    return parser.parse_args()

## returns T_ref_world
def get_reference_pose(rendering_metadata_file):
    metadata = np.loadtxt(rendering_metadata_file)
    R, t = BaseDataset.cameraMat(metadata[0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -np.matmul(R, t)
    return T

def convert(args):
    scene = trimesh.load_mesh(args.original_gt_mesh, 'obj')
    if isinstance(scene, trimesh.Trimesh):
        scene = scene.scene()
    scene = scene.scaled(P2M_SCALE_FACTOR)
    mesh = trimesh.util.concatenate([
        trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
        for g in scene.geometry.values()
    ])
    ref_pose = get_reference_pose(args.rendering_metadata)
    mesh.apply_transform(ref_pose)
    trimesh.repair.fix_normals(mesh, True)
    trimesh.repair.fix_inversion(mesh, True)
    mesh.export(args.converted_mesh)

if __name__ == '__main__':
    args = parse_args()
    convert(args)
