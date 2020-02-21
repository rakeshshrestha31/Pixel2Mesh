#!/usr/bin/env python

import open3d
import argparse
import sys, os
import skimage.io
import skimage.transform
import numpy as np
import cv2

import torch
import torchvision.utils

import config
import utils.render_blender as render_blender

def parse_args():
    parser = argparse.ArgumentParser(
        description='Renders given obj file by rotation a camera around it.'
    )
    parser.add_argument('shapenet_dir', type=str,
                        help='directory of the original shapenetv1 dataset')
    parser.add_argument('rendering_dir', type=str,
                        help='Directory of the rendering meta data.')
    parser.add_argument('xms_exec', type=str,
                        help='xms executable for rendering depth')
    parser.add_argument('rendering_file', type=str, help='rendering_metadata.txt file')
    parser.add_argument('--render-depths', dest='render_depths', action='store_true')
    return parser.parse_args()

def split_path(path):
    if path == '/' or path == '':
        return [path]
    else:
        path1, path2 = os.path.split(path)
        return split_path(path1) + [path2]

def read_images(args):
    labels = split_path(args.rendering_file)[-4:-2]
    images_path = os.path.join(args.rendering_dir, *labels, 'rendering')
    images = []
    for i in range(24):
        image_file = os.path.join(images_path, '{0:02}.png'.format(i))
        image = skimage.io.imread(image_file)
        images.append(image)
    return images

def read_depths(args):
    labels = split_path(args.rendering_file)[-4:-2]
    depths_path = os.path.join(args.rendering_dir, *labels, 'rendering_depth')
    depths = []
    for i in range(24):
        depth_file = os.path.join(depths_path, '{0:02}.png'.format(i))
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        depths.append(depth)
    return depths

def preprocess_depth(depth):
    # make three channel
    depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 2:
        depth = np.repeat(np.expand_dims(depth, 0), 3, 0)
    return torch.from_numpy(depth)

def preprocess_image(image):
    image = skimage.transform.resize(image, (config.IMG_SIZE,)*2)
    image[np.where(image[:, :, 3] == 0)] = 0.0
    image = image[:, :, :3].astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)

def verify_depths(args):
    views = [1, 2, 6, 12, 15, 17, 18]
    labels = split_path(args.rendering_file)[-4:-2]
    if args.render_depths:
        depths = render_blender.render_object(labels, args, True)
    else:
        depths = read_depths(args)
    depths = [depths[i] for i in views]
    depths = [preprocess_depth(i) for i in depths]

    images = read_images(args)
    images = [images[i] for i in views]
    images = [preprocess_image(i) for i in images]

    # pairs = [(i, j) for i, j in zip(depths, images)]
    # flat_pairs = [j for i in pairs for j in i]
    grid_tensor = torch.stack(depths + images, dim=0)
    torchvision.utils.save_image(
        grid_tensor, '/tmp/renderings.png', nrow=len(views),
        padding=10, pad_value=1.0
    )

if __name__ == '__main__':
    args = parse_args()
    verify_depths(args)

