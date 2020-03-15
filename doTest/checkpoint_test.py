#!/usr/bin/env python
##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

import sys
import argparse
import random
import torch

from models.p2m import P2MModel
from utils.mesh import Ellipsoid
from options import update_options, options, reset_options
from functions.saver import CheckpointSaver

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', default='debug', type=str)

    parser.add_argument('--use-rgb-features', dest='use_rgb_features',
                        action='store_true')
    parser.add_argument('--dont-use-rgb-features', dest='use_rgb_features',
                        action='store_false')

    parser.add_argument('--use-costvolume-features', dest='use_costvolume_features',
                        action='store_true')
    parser.add_argument('--dont-use-costvolume-features', dest='use_costvolume_features',
                        action='store_false')

    parser.add_argument('--use-contrastive-depth', dest='use_contrastive_depth',
                        action='store_true')
    parser.add_argument('--dont-use-contrastive-depth', dest='use_contrastive_depth',
                        action='store_false')

    parser.add_argument('--use-predicted-depth-as-feature',
                        dest='use_predicted_depth_as_feature',
                        action='store_true')
    parser.add_argument('--dont-use-predicted-depth-as-feature',
                        dest='use_predicted_depth_as_feature',
                        action='store_false')

    parser.add_argument('--feature-fusion-method',
                        help='[concat|stats|attention]', type=str)
    parser.add_argument('--num-attention-heads',
                        help='number of attention heads', type=int)
    parser.add_argument('--num-attention-features',
                        help='number of attention features '
                             '< 0 indicates no change in features number '
                             'from input',
                        type=int)

    # set the default booleans to None otherwise it will be False and
    # overwrite the default options
    parser.set_defaults(
        use_rgb_features=None,
        use_costvolume_features=None,
        use_contrastive_depth=None,
        use_predicted_depth_as_feature=None
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger, writer = reset_options(options, args)

    ellipsoid = Ellipsoid(options.dataset.mesh_pos)
    model = P2MModel(
        options.model, ellipsoid,
        options.dataset.camera_f, options.dataset.camera_c,
        options.dataset.mesh_pos, options.train.freeze_cv,
        options.mvsnet_checkpoint
    )
    checkpoint_saver = CheckpointSaver(
        logger, checkpoint_dir=str(options.checkpoint_dir),
        checkpoint_file=options.checkpoint
    )
    checkpoint = checkpoint_saver.load_checkpoint()

    def update_key(key):
        return key.replace('vgg', 'depth_vgg')

    checkpoint['model'] = {
        update_key(key): value for key, value in checkpoint['model'].items()
    }
    ret = model.load_state_dict(
        checkpoint['model'], strict=False
    )
    print('ret:', ret)
    torch.save(checkpoint, '/tmp/tmp_checkpoint.pt')
    # print('missing:', missing_keys)
    # print('unexpected_keys:', unexpected_keys)

