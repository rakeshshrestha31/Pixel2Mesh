import argparse
import sys
import random
import numpy as np
import torch
# import open3d as o3d
from functions.trainer import Trainer
from options import update_options, options, reset_options



def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Training Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # training
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', default='debug', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--backbone', default='vgg16', type=str)
    parser.add_argument('--num_views', default=3, help='num_views', type=int)
    parser.add_argument('--seed', default=3, help='seed', type=int)
    parser.add_argument('--freeze-cv', dest='freeze_cv', action='store_true')
    parser.add_argument('--only-depth-training', dest='only_depth_training', action='store_true')
    parser.add_argument('--all-loss-training', dest='only_depth_training', action='store_false')
    parser.add_argument('--depth-loss-weight', default=1000, help='depth loss weight', type=float)
    parser.add_argument('--lr', default=1e-4, help='initial learning rate', type=float)
    parser.add_argument('--lr-factor', default=0.1, help='learning rate factor', type=float)
    parser.add_argument('--lr-step', nargs='+', type=int)
    parser.set_defaults(only_depth_training=False)
    parser.set_defaults(freeze_cv=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(options, logger, writer)
    trainer.train()


if __name__ == "__main__":
    main()
