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
from utils.vis.renderer import MeshRenderer


class Evaluator(CheckpointRunner):

    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

        self.check_best_metrics = [
            CheckBest('cd', 'best_test_loss_chamfer', is_loss=True),
            CheckBest('depth_loss', 'best_test_loss_depth', is_loss=True),
            CheckBest('r_gt_depth_loss', 'best_test_loss_rendered_vs_gt',
                      is_loss=True),
            CheckBest('f1_tau', 'best_test_f1_tau', is_loss=False),
            CheckBest('f1_2tau', 'best_test_f1_2tau', is_loss=False)
        ]
        self.upsampled_chamfer_loss = options.test.upsampled_chamfer_loss

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        if self.options.model.name == "pixel2mesh":
            # Renderer for visualization
            self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                         self.options.dataset.mesh_pos)
            # Initialize distance module
            self.chamfer = ChamferDist()
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
            # use weighted mean evaluation metrics or not
            self.weighted_mean = self.options.test.weighted_mean
        else:
            self.renderer = None
        self.num_classes = self.options.dataset.num_classes

        self.p2m_loss = P2MLoss(
            self.options.loss, self.ellipsoid,
            self.options.test.upsampled_chamfer_loss
        ).cuda()

        if shared_model is not None:
            self.model = shared_model
        else:
            if self.options.model.name == "pixel2mesh":
                # create model
                self.model = P2MModel(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos, self.options.train.freeze_cv)
            elif self.options.model.name == "classifier":
                self.model = Classifier(self.options.model, self.options.dataset.num_classes)
            else:
                raise NotImplementedError("Your model is not found")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0
        self.total_step_count = 0

    def models_dict(self):
        return {'model': self.model}

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)

    def evaluate_depth_loss(self, gt_depth, pred_depth,
                            mask, labels, depth_loss):
        batch_size = pred_depth.size(0)
        num_views = pred_depth.size(1)
        for i in range(batch_size):
            label = labels[i].cpu().item()
            loss = self.p2m_loss.depth_loss(
                gt_depth[i].unsqueeze(0), pred_depth[i].unsqueeze(0),
                mask[i].unsqueeze(0)
            )
            depth_loss[label].update(loss)

    def evaluate_chamfer_and_f1(self, pred_vertices, pred_faces,
                                gt_points, labels):
        # calculate accurate chamfer distance; ground truth points with different lengths;
        # therefore cannot be batched
        batch_size = pred_vertices.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length = gt_points[i].size(0)
            label = labels[i].cpu().item()
            if self.upsampled_chamfer_loss:
                d1, d2, i1, i2 = self.p2m_loss.upsampled_chamfer_dist(
                    pred_vertices[i].unsqueeze(0),
                    pred_faces.unsqueeze(0),
                    gt_points[i].unsqueeze(0)
                )
            else:
                d1, d2, i1, i2 = self.chamfer(pred_vertices[i].unsqueeze(0), gt_points[i].unsqueeze(0))
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()  # convert to millimeter
            self.chamfer_distance[label].update(np.mean(d1) + np.mean(d2))
            self.f1_tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 1E-4))
            self.f1_2tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 2E-4))

    def evaluate_accuracy(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        top_k = [1, 5]
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1.0 / batch_size)
            if k == 1:
                self.acc_1.update(acc)
            elif k == 5:
                self.acc_5.update(acc)

    def evaluate_step(self, input_batch):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            # Get ground truth

            # predict with model
            out = self.model(input_batch)

            if self.options.model.name == "pixel2mesh":
                pred_vertices = out["pred_coord"][-1]
                gt_points = input_batch["points_orig"]
                # gt_points = input_batch["points"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda() for pts in gt_points]
                self.evaluate_chamfer_and_f1(
                    pred_vertices,
                    self.ellipsoid.faces[-1],
                    gt_points, input_batch["labels"]
                )
                self.evaluate_depth_loss(
                    out["depths"], input_batch["depths"],
                    input_batch["masks"], input_batch["labels"],
                    self.depth_loss
                )

                all_valid_masks = torch.ones(
                    *(input_batch["masks"].size()), dtype=torch.uint8,
                    device=input_batch["masks"].device
                )
                # don't mask rendered_depth cuz
                # depth rendered outside the mask is also incorrect
                self.evaluate_depth_loss(
                    out["rendered_depths"][-1],
                    out["depths"] * input_batch["masks"],
                    all_valid_masks, input_batch["labels"],
                    self.rendered_vs_cv_depth_loss,
                )
                self.evaluate_depth_loss(
                    out["rendered_depths"][-1], input_batch["depths"],
                    all_valid_masks, input_batch["labels"],
                    self.rendered_vs_gt_depth_loss
                )
            elif self.options.model.name == "classifier":
                self.evaluate_accuracy(out, input_batch["labels"])

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

        if self.options.model.name == "pixel2mesh":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
            self.depth_loss = [AverageMeter() for _ in range(self.num_classes)]
            self.rendered_vs_cv_depth_loss = [
                AverageMeter() for _ in range(self.num_classes)
            ]
            self.rendered_vs_gt_depth_loss = [
                AverageMeter() for _ in range(self.num_classes)
            ]
        elif self.options.model.name == "classifier":
            self.acc_1 = AverageMeter()
            self.acc_5 = AverageMeter()

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
        if self.options.model.name == "pixel2mesh":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "depth_loss": self.average_of_average_meters(self.depth_loss),
                "r_cv_depth_loss":
                    self.average_of_average_meters(self.rendered_vs_cv_depth_loss),
                "r_gt_depth_loss": \
                    self.average_of_average_meters(self.rendered_vs_gt_depth_loss),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "classifier":
            return {
                "acc_1": self.acc_1,
                "acc_5": self.acc_5,
            }

    def evaluate_summaries(self, input_batch, out_summary):
        self.logger.info("Test Step %06d/%06d (%06d) " % (self.evaluate_step_count,
                                                          len(self.dataset) // (
                                                                  self.options.num_gpus * self.options.test.batch_size),
                                                          self.total_step_count,) \
                         + ", ".join([key + " " + (str(val) if isinstance(val, AverageMeter) else "%.6f" % val)
                                      for key, val in self.get_result_summary().items()]))

        self.summary_writer.add_histogram("eval_labels", input_batch["labels"].cpu().numpy(),
                                          self.total_step_count)
        if self.renderer is not None:
            # Do visualization for the first 2 images of the batch
            render_mesh = self.renderer.p2m_batch_visualize(input_batch, out_summary, self.ellipsoid.faces)
            self.summary_writer.add_image("eval_render_mesh", render_mesh, self.total_step_count)
