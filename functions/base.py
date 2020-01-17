import os
import time
from datetime import timedelta
from logging import Logger

import torch
import torch.nn
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import default_collate

import config
from datasets.imagenet import ImageNet
from datasets.dtu import MVSDataset
from datasets.shapenet import ShapeNet, get_shapenet_collate, ShapeNetImageFolder
from functions.saver import CheckpointSaver


class CheckpointRunner(object):
    def __init__(self, options, logger: Logger, summary_writer: SummaryWriter,
                 dataset=None, training=True, shared_model=None):
        self.options = options
        self.logger = logger

        # GPUs
        if not torch.cuda.is_available() and self.options.num_gpus > 0:
            raise ValueError("CUDA not found yet number of GPUs is set to be greater than 0")
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            logger.info("CUDA visible devices is activated here, number of GPU setting is not working")
            self.gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            self.options.num_gpus = len(self.gpus)
            enumerate_gpus = list(range(self.options.num_gpus))
            logger.info("CUDA is asking for " + str(self.gpus) + ", PyTorch to doing a mapping, changing it to " +
                        str(enumerate_gpus))
            self.gpus = enumerate_gpus
        else:
            self.gpus = list(range(self.options.num_gpus))
            logger.info("Using GPUs: " + str(self.gpus))

        # initialize summary writer
        self.summary_writer = summary_writer

        # initialize dataset
        if dataset is None:
            dataset = options.dataset  # useful during training
        self.dataset = self.load_dataset(dataset, training)
        self.dataset_collate_fn = self.load_collate_fn(dataset, training)

        # by default, epoch_count = step_count = 0
        self.epoch_count = self.step_count = 0
        self.time_start = time.time()

        # override this function to define your model, optimizers etc.
        # in case you want to use a model that is defined in a trainer or other place in the code,
        # shared_model should help. in this case, checkpoint is not used
        self.logger.info("Running model initialization...")
        self.init_fn(shared_model=shared_model)

        self.saver = CheckpointSaver(self.logger, checkpoint_dir=str(self.options.checkpoint_dir),
                                     checkpoint_file=self.options.checkpoint)
        if shared_model is None:
            # checkpoint is loaded if any
            self.init_with_checkpoint()

    @staticmethod
    def intrinsics_from_dataset(dataset):
        import numpy as np
        intrinsics = np.eye(3, 3)
        intrinsics[:2, :2] = np.diag(dataset.camera_f)
        intrinsics[:2, 2] = np.asarray(dataset.camera_c)
        return intrinsics

    def load_dataset(self, dataset, training):
        self.logger.info("Loading datasets: %s" % dataset.name)
        if dataset.name == "shapenet":
            intrinsics = self.intrinsics_from_dataset(dataset)
            return ShapeNet(config.SHAPENET_ROOT, dataset.subset_train if training else dataset.subset_eval,
                            dataset.mesh_pos, dataset.normalization, dataset.shapenet, intrinsics, training)
        elif dataset.name == "shapenet_demo":
            return ShapeNetImageFolder(dataset.predict.folder, dataset.normalization, dataset.shapenet)
        elif dataset.name == "imagenet":
            return ImageNet(config.IMAGENET_ROOT, "train" if training else "val")
        if dataset.name == "dtu":
            return MVSDataset(config.SHAPENET_ROOT+'/data_tf', dataset.train_list if training else dataset.test_list, "train" if training else "val", dataset.num_views, dataset.normalization, options=self.options)

        raise NotImplementedError("Unsupported dataset")

    def load_collate_fn(self, dataset, training):
        if dataset.name == "shapenet" or dataset.name == "dtu":
            return get_shapenet_collate(dataset.shapenet.num_points)
        else:
            return default_collate

    def init_fn(self, shared_model=None, **kwargs):
        raise NotImplementedError('You need to provide an _init_fn method')

    # Pack models and optimizers in a dict - necessary for checkpointing
    def models_dict(self):
        return None

    def optimizers_dict(self):
        # NOTE: optimizers and models cannot have conflicting names
        return None

    def init_with_checkpoint(self):
        checkpoint = self.saver.load_checkpoint()
        if checkpoint is None:
            self.logger.info("Checkpoint not loaded")
            return
        for model_name, model in self.models_dict().items():
            if model_name in checkpoint:
                if isinstance(model, torch.nn.DataParallel):
                    model.module.load_state_dict(checkpoint[model_name], strict=False)
                else:
                    model.load_state_dict(checkpoint[model_name], strict=False)
        if self.optimizers_dict() is not None:
            for optimizer_name, optimizer in self.optimizers_dict().items():
                if optimizer_name in checkpoint:
                    optimizer.load_state_dict(checkpoint[optimizer_name])
        else:
            self.logger.warning("Optimizers not found in the runner, skipping...")
        if "epoch" in checkpoint:
            self.epoch_count = checkpoint["epoch"]
        if "total_step_count" in checkpoint:
            self.step_count = checkpoint["total_step_count"]

    def dump_checkpoint(self, prefix, is_indexed=False):
        checkpoint = {
            "epoch": self.epoch_count,
            "total_step_count": self.step_count
        }
        for model_name, model in self.models_dict().items():
            if isinstance(model, torch.nn.DataParallel):
                checkpoint[model_name] = model.module.state_dict()
            else:
                checkpoint[model_name] = model.state_dict()
            for k, v in list(checkpoint[model_name].items()):
                if isinstance(v, torch.Tensor) and v.is_sparse:
                    checkpoint[model_name].pop(k)
        if self.optimizers_dict() is not None:
            for optimizer_name, optimizer in self.optimizers_dict().items():
                checkpoint[optimizer_name] = optimizer.state_dict()
        if is_indexed:
            filename = "%s_%06d_%06d" % (prefix, self.step_count, self.epoch_count)
        else:
            filename = prefix
        self.saver.save_checkpoint(checkpoint, filename)

    @property
    def time_elapsed(self):
        return timedelta(seconds=time.time() - self.time_start)
