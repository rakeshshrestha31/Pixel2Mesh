import os
import pprint
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import yaml
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

from logger import create_logger

options = edict()

options.name = 'p2m'
options.version = None
options.num_workers = 1
options.num_gpus = 1
options.pin_memory = True

options.log_dir = "logs"
options.log_level = "info"
options.summary_dir = "summary"
options.checkpoint_dir = "checkpoints"
options.checkpoint = None

options.dataset = edict()
options.dataset.name = "shapenet"
options.dataset.subset_train = "train_tf"
options.dataset.subset_eval = "test_tf"
options.dataset.camera_f_base = [361.54125, 360.3975]
options.dataset.camera_c_base = [82.900625 , 66.383875 ]

options.dataset.camera_f = [506.360294118, 631.169001751]
options.dataset.camera_c = [116.107317927,116.258975482]
options.dataset.mesh_pos = [0., 0., -0.8]
options.dataset.normalization = True
options.dataset.num_classes = 13
options.dataset.num_views = 1
options.dataset.proj_matrix = None
options.dataset.debug_scan2 = False

options.dataset.shapenet = edict()
options.dataset.shapenet.num_points = 3000
options.dataset.shapenet.resize_with_constant_border = False

options.dataset.predict = edict()
options.dataset.predict.folder = "./tmp"

options.model = edict()
options.model.name = "pixel2mesh"
options.model.hidden_dim = 192
options.model.last_hidden_dim = 192
options.model.coord_dim = 3
# options.model.backbone = "vgg16"
options.model.backbone = "encoder8"
options.model.gconv_activation = True
# provide a boundary for z, so that z will never be equal to 0, on denominator
# if z is greater than 0, it will never be less than z;
# if z is less than 0, it will never be greater than z.
options.model.z_threshold = 0
# align with original tensorflow model
# please follow experiments/tensorflow.yml
options.model.align_with_tensorflow = False

options.loss = edict()
options.loss.weights = edict()
options.loss.weights.normal = 1.6e-4
options.loss.weights.edge = 0.3
options.loss.weights.laplace = 0.5
options.loss.weights.move = 0.1
options.loss.weights.constant = 1.
options.loss.weights.chamfer = [1., 1., 1.]
options.loss.weights.chamfer_opposite = 1.
options.loss.weights.reconst = 0.

options.train = edict()
options.train.num_epochs = 200
options.train.batch_size = 16
options.train.summary_steps = 10
options.train.checkpoint_steps = 100000
options.train.test_epochs = 5
options.train.use_augmentation = True
options.train.shuffle = True

options.test = edict()
options.test.dataset = []
options.test.summary_steps = 20
options.test.batch_size = 32
options.test.shuffle = False
options.test.weighted_mean = False

options.optim = edict()
options.optim.name = "adam"
options.optim.adam_beta1 = 0.9
options.optim.sgd_momentum = 0.9
options.optim.lr = 5.0E-5
options.optim.wd = 1.0E-6
options.optim.lr_step = [30, 45]
options.optim.lr_factor = 0.1

def _update_dict(full_key, val, d):
    for vk, vv in val.items():
        if vk not in d:
            raise ValueError("{}.{} does not exist in options".format(full_key, vk))
        if isinstance(vv, list):
            d[vk] = np.array(vv)
        elif isinstance(vv, dict):
            _update_dict(full_key + "." + vk, vv, d[vk])
        else:
            d[vk] = vv


def _update_options(options_file):
    # do scan twice
    # in the first round, MODEL.NAME is located so that we can initialize MODEL.EXTRA
    # in the second round, we update everything

    with open(options_file) as f:
        options_dict = yaml.safe_load(f)
        # do a dfs on `BASED_ON` options files
        if "based_on" in options_dict:
            for base_options in options_dict["based_on"]:
                _update_options(os.path.join(os.path.dirname(options_file), base_options))
            options_dict.pop("based_on")
        _update_dict("", options_dict, options)


def update_options(options_file):
    _update_options(options_file)


def gen_options(options_file):
    def to_dict(ed):
        ret = dict(ed)
        for k, v in ret.items():
            if isinstance(v, edict):
                ret[k] = to_dict(v)
            elif isinstance(v, np.ndarray):
                ret[k] = v.tolist()
        return ret

    cfg = to_dict(options)

    with open(options_file, 'w') as f:
        yaml.safe_dump(dict(cfg), f, default_flow_style=False)


def slugify(filename):
    filename = os.path.relpath(filename, ".")
    if filename.startswith("experiments/"):
        filename = filename[len("experiments/"):]
    return os.path.splitext(filename)[0].lower().replace("/", "_").replace(".", "_")


def reset_options(options, args, phase='train'):
    if hasattr(args, "batch_size") and args.batch_size:
        options.train.batch_size = options.test.batch_size = args.batch_size
    if hasattr(args, "version") and args.version:
        options.version = args.version
    if hasattr(args, "num_epochs") and args.num_epochs:
        options.train.num_epochs = args.num_epochs
    if hasattr(args, "checkpoint") and args.checkpoint:
        options.checkpoint = args.checkpoint
    if hasattr(args, "folder") and args.folder:
        options.dataset.predict.folder = args.folder
    if hasattr(args, "gpus") and args.gpus:
        options.num_gpus = args.gpus
    if hasattr(args, "shuffle") and args.shuffle:
        options.train.shuffle = options.test.shuffle = True
    if hasattr(args, "backbone") and args.backbone:
        options.model.backbone = args.backbone
    if hasattr(args, "num_views") and args.num_views:
        options.dataset.num_views = args.num_views
    if hasattr(args, "dataset") and args.dataset:
        options.dataset.name = args.dataset
        options.dataset.train_list = "./datasets/data/shapenet/meta/train_dtu.txt"
        options.dataset.test_list = "./datasets/data/shapenet/meta/test_dtu.txt"
        options.dataset.debug_scan2 = args.debug_scan2
        # options.dataset.proj_matrix =
        if options.dataset.debug_scan2:
            options.dataset.train_list = "./datasets/data/shapenet/meta/train_dtu_scan2.txt"
            options.dataset.test_list = "./datasets/data/shapenet/meta/test_dtu_scan4.txt"

    options.name = args.name

    if options.version is None:
        prefix = ""
        if args.options:
            prefix = slugify(args.options) + "_"
        options.version = prefix + datetime.now().strftime('%m%d%H%M%S')  # ignore %Y
    options.log_dir = os.path.join(options.log_dir, options.name)
    print('=> creating {}'.format(options.log_dir))
    os.makedirs(options.log_dir, exist_ok=True)

    options.checkpoint_dir = os.path.join(options.checkpoint_dir, options.name, options.version)
    print('=> creating {}'.format(options.checkpoint_dir))
    os.makedirs(options.checkpoint_dir, exist_ok=True)

    options.summary_dir = os.path.join(options.summary_dir, options.name, options.version)
    print('=> creating {}'.format(options.summary_dir))
    os.makedirs(options.summary_dir, exist_ok=True)

    logger = create_logger(options, phase=phase)
    options_text = pprint.pformat(vars(options))
    logger.info(options_text)

    print('=> creating summary writer')
    writer = SummaryWriter(options.summary_dir)

    return logger, writer


if __name__ == "__main__":
    parser = ArgumentParser("Read options and freeze")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    update_options(args.input)
    gen_options(args.output)
