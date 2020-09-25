#/usr/bin/env python
##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

import os
import argparse
import numpy as np
from utils import xyz_to_obj
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='xyz to obj')
    parser.add_argument('in_dir', type=str)
    parser.add_argument('faces_file', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('out_postfix', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for filename in tqdm.tqdm(os.listdir(args.in_dir)):
        full_filename = os.path.join(args.in_dir, filename)
        if not os.path.isfile(full_filename):
            continue
        xyz_to_obj.convert(
            full_filename, args.faces_file, args.out_dir, args.out_postfix
        )


