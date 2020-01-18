#!/usr/bin/env python

import argparse, sys, os
import multiprocessing as mp
import functools
import time

# multi-processing stuffs
NCORE = 1


def verify_object(rendering_dir, obj_category):
    obj, category = obj_category
    depth_file = os.path.join(
        rendering_dir, obj, category, 'rendering_depth', '00.png'
    )
    if not os.path.isfile(depth_file):
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

    args = parser.parse_args()

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
