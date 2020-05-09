import argparse
import sys

from functions.evaluator import Evaluator
from options import update_options, options, reset_options
from utils.args_utils import add_model_args


def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Evaluation Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--shuffle', help='shuffle samples', default=False, action='store_true')
    parser.add_argument('--checkpoint', help='trained checkpoint file', type=str, required=True)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', help='subfolder name of this experiment', required=True, type=str)
    parser.add_argument('--gpus', help='number of GPUs to use', type=int)
    parser.add_argument('--input-views', help='input_views', nargs='+', type=int)

    parser.add_argument('--test-upsampled-chamfer-loss',
                        dest='test_upsampled_chamfer_loss', action='store_true')
    parser.add_argument('--prediction-dir', help='location to save predicted mesh', type=str)
    add_model_args(parser)

    parser.set_defaults(
        only_depth_training=None,
        freeze_cv=None,
        upsampled_normal_loss=None,
        use_backprojected_depth_loss=None,
        train_upsampled_chamfer_loss=None,
        test_upsampled_chamfer_loss=None
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args, phase='eval')

    evaluator = Evaluator(options, logger, writer)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
