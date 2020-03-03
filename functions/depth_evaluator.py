from logging import Logger

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from functions.check_best import CheckBest
from models.classifier import Classifier
from models.layers.chamfer_wrapper import ChamferDist
from models.p2m import P2MModel
from models.losses.p2m import P2MLoss
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
from utils.vis.depth_renderer import DepthRenderer


class DepthEvaluator(CheckpointRunner):

    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

        self.check_best_metrics = [
            CheckBest('depth_loss', 'best_test_loss_depth', is_loss=True),
            CheckBest('l1_loss', 'best_test_l1_depth', is_loss=True),
            CheckBest('l2_loss', 'best_test_l2_depth', is_loss=True),
        ]
        self.l1_loss_model = nn.L1Loss(reduction='mean')
        self.l2_loss_model = nn.MSELoss(reduction='mean')

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        # Renderer for visualization
        self.renderer = DepthRenderer()
        self.weighted_mean = self.options.test.weighted_mean
        self.num_classes = self.options.dataset.num_classes

        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = MVSNet(freeze_cv=False)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0
        self.total_step_count = 0

    def models_dict(self):
        return {'model': self.model}

    def evaluate_depth_loss(self, pred_depth, gt_depth, mask, labels):
        batch_size = pred_depth.size(0)
        num_views = pred_depth.size(1)
        for i in range(batch_size):
            label = labels[i].cpu().item()
            depth_loss = P2MLoss.depth_loss(
                gt_depth[i], pred_depth[i], mask[i]
            )
            self.depth_loss[label].update(depth_loss)

            masked_gt = gt_depth[i] * mask[i]
            masked_pred = pred_depth[i] * mask[i]
            l1_loss = self.l1_loss_model(masked_gt, masked_pred)
            self.l1_loss[label].update(l1_loss)

            l2_loss = self.l2_loss_model(masked_gt, masked_pred)
            # root to make unit mm for easy comparison
            l2_loss = torch.sqrt(l2_loss)
            self.l2_loss[label].update(l2_loss)

    def evaluate_step(self, input_batch):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            # Get ground truth

            model_input = {
                key: input_batch[key]
                for key in ['images', 'proj_matrices', 'depth_values']
            }
            if self.dataset.augment_ref_views:
                # only use the view 0 as reference
                model_input['view_lists'] = [(0, 1, 2)]
                input_depth_key = 'depth'
                input_mask_key = 'mask'
            else:
                input_depth_key = 'depths'
                input_mask_key = 'masks'

            # predict with model
            out = self.model(model_input)

            self.evaluate_depth_loss(
                out["depths"], input_batch[input_depth_key],
                input_batch[input_mask_key], input_batch["labels"]
            )

        return out

    # noinspection PyAttributeOutsideInit
    def evaluate(self):
        self.logger.info("Running evaluations...")

        # clear evaluate_step_count, but keep total count uncleared
        self.evaluate_step_count = 0

        test_data_loader = DataLoader(self.dataset,
                                      batch_size=self.options.test.batch_size * self.options.num_gpus,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.test.shuffle,
                                      collate_fn=self.dataset_collate_fn)

        self.depth_loss = [AverageMeter() for _ in range(self.num_classes)]
        self.l1_loss = [AverageMeter() for _ in range(self.num_classes)]
        self.l2_loss = [AverageMeter() for _ in range(self.num_classes)]

        # Iterate over all batches in an epoch
        for step, batch in enumerate(test_data_loader):
            # Send input to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Run evaluation step
            out = self.evaluate_step(batch)

            # Tensorboard logging every summary_steps steps
            if self.evaluate_step_count % self.options.test.summary_steps == 0:
                self.evaluate_summaries(batch, out)

            # add later to log at step 0
            self.evaluate_step_count += 1
            self.total_step_count += 1

        result_summary = self.get_result_summary()
        for key, val in result_summary.items():
            scalar = val
            if isinstance(val, AverageMeter):
                scalar = val.avg
            self.logger.info("Test [%06d] %s: %.6f" % (self.total_step_count, key, scalar))
            self.summary_writer.add_scalar("eval_" + key, scalar, self.total_step_count + 1)

        # save checkpoint if best
        for check_best in self.check_best_metrics:
            scalar = result_summary[check_best.key]
            if isinstance(scalar, AverageMeter):
                scalar = scalar.avg
            if check_best.check_best(scalar):
                self.logger.info("Test Step %06d/%06d (%06d), found best %s: %f" % (
                    self.evaluate_step_count,
                    len(self.dataset) // (
                          self.options.num_gpus * self.options.test.batch_size),
                    self.total_step_count,
                    check_best.metric_name, check_best.best,
                ))
                self.dump_checkpoint(check_best.metric_name, is_indexed=False)

    def average_of_average_meters(self, average_meters):
        s = sum([meter.sum for meter in average_meters])
        c = sum([meter.count for meter in average_meters])
        weighted_avg = s / c if c > 0 else 0.
        avg = sum([meter.avg for meter in average_meters]) / len(average_meters)
        ret = AverageMeter()
        if self.weighted_mean:
            ret.val, ret.avg = avg, weighted_avg
        else:
            ret.val, ret.avg = weighted_avg, avg
        return ret

    def get_result_summary(self):
        return {
            "depth_loss": self.average_of_average_meters(self.depth_loss),
            "l1_loss": self.average_of_average_meters(self.l1_loss),
            "l2_loss": self.average_of_average_meters(self.l2_loss)
        }

    def evaluate_summaries(self, input_batch, out_summary):
        self.logger.info(
                "Test Step %06d/%06d (%06d) " % (
                    self.evaluate_step_count,
                    len(self.dataset) // (
                            self.options.num_gpus * self.options.test.batch_size),
                    self.total_step_count,
                ) \
                + ", ".join([key + " " + (str(val) if isinstance(val, AverageMeter) else "%.6f" % val)
                             for key, val in self.get_result_summary().items()]))

        self.summary_writer.add_histogram("eval_labels", input_batch["labels"].cpu().numpy(),
                                          self.total_step_count)
        if self.renderer is not None:
            # Do visualization for the first 2 images of the batch
            render_depth = self.renderer.depth_batch_visualize(
                input_batch, out_summary
            )
            self.summary_writer.add_image("eval_render_depth", render_depth, self.step_count)

