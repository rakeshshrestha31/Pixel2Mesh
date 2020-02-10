#!/usr/bin/env python

import argparse, sys, os
import multiprocessing as mp
import functools
import time
import numpy as np

# multi-processing stuffs
NCORE = 8

def verify_object(rendering_dir, obj_category):
    obj, category = obj_category
    depth_dir = os.path.join(
        rendering_dir, obj, category, 'rendering_depth'
    )
    depth_files = [
        os.path.join(depth_dir, '{0:02}.png'.format(i))
        for i in range(23)
    ]
    depth_files_available = [
        os.path.isfile(depth_file) for depth_file in depth_files
    ]

    if not np.all(depth_files_available):
        # print('Depth file', depth_file, 'does not exist')
        print(obj, category)
        return

    # depth_time = os.path.getmtime(depth_file)
    # if time.time() - depth_time > 27500:
    #     print('Depth file', depth_file, 'too old')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='verify if the shapenet dataset is complete')

    parser.add_argument('rendering_dir', type=str,
                        help='Directory of the rendering meta data.')
    parser.add_argument('--objects-categories-file', type=str, default='',
                        help='file containing objects and categories'
                             'that are to be rendered')

    args = parser.parse_args()

    if args.objects_categories_file:
        shapenet_objects_categories = np.loadtxt(
            args.objects_categories_file, dtype=str
        ).tolist()
        shapenet_objects_categories = (
            tuple(i.split('_')[:2]) for i in shapenet_objects_categories
        )
    else:
        shapenet_objects = (
            i for i in os.listdir(args.rendering_dir)
            if os.path.isdir(os.path.join(args.rendering_dir, i))
        )
        shapenet_objects_categories = (
            (obj, category)
            for obj in shapenet_objects
            for category in os.listdir(os.path.join(args.rendering_dir, obj))
            if os.path.isdir(os.path.join(args.rendering_dir, obj, category))
        )

    bound_verify_object = functools.partial(verify_object, args.rendering_dir)

    with mp.Pool(NCORE) as p:
        p.map(bound_verify_object, shapenet_objects_categories)
