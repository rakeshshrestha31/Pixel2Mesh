import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.chamfer_wrapper import ChamferDist
from models.layers.sample_points import PointSampler
import numpy as np
import functools

import config

class P2MLoss(nn.Module):
    def __init__(self, options, ellipsoid, upsampled_chamfer_loss=False):
        super().__init__()
        self.options = options
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()
        self.laplace_idx = nn.ParameterList([
            nn.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx])
        self.edges = nn.ParameterList([
            nn.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges])
        self.ellipsoid = ellipsoid
        self.points_sampler = PointSampler(options.num_chamfer_upsample)
        self.upsampled_chamfer_loss = upsampled_chamfer_loss
        self.depth_loss_function = self.get_depth_loss_function(options.depth_loss_type)
        print('==> using depth loss', options.depth_loss_type)

    def get_depth_loss_function(self, loss_type):
        if loss_type == 'huber':
            return (lambda x, y, mask: self.huber_loss(x[mask], y[mask]))
        elif loss_type == 'berhu':
            return self.adaptive_berhu_loss
        elif loss_type == 'l1':
            return (lambda x, y, mask: self.l1_loss(x[mask], y[mask]))
        elif loss_type == 'l2':
            return (lambda x, y, mask: \
                        torch.sqrt(self.l2_loss(x[mask], y[mask])))
        else:
            print('unrecognized depth loss:', loss_type)

    def edge_regularization(self, pred, edges):
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        return self.l2_loss(pred[:, edges[:, 0]], pred[:, edges[:, 1]]) * pred.size(-1)

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """

        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices

        vertices = inputs[:, all_valid_indices]
        vertices[:, invalid_mask] = 0
        neighbor_sum = torch.sum(vertices, 2)
        neighbor_count = lap_idx[:, -1].float()
        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]

        return laplace

    def laplace_regularization(self, input1, input2, block_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """
        lap1 = self.laplace_coord(input1, self.laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, self.laplace_idx[block_idx])
        laplace_loss = self.l2_loss(lap1, lap2) * lap1.size(-1)
        move_loss = self.l2_loss(input1, input2) * input1.size(-1) if block_idx > 0 else 0
        return laplace_loss, move_loss

    def normal_loss(self, gt_normal, indices, pred_points, adj_list):
        edges = F.normalize(pred_points[:, adj_list[:, 0]] - pred_points[:, adj_list[:, 1]], dim=2)
        nearest_normals = torch.stack([t[i] for t, i in zip(gt_normal, indices.long())])
        normals = F.normalize(nearest_normals[:, adj_list[:, 0]], dim=2)
        cosine = torch.abs(torch.sum(edges * normals, 2))
        return torch.mean(cosine)

    def image_loss(self, gt_img, pred_img):
        rect_loss = F.binary_cross_entropy(pred_img, gt_img)
        return rect_loss

    def get_tensor_device(tensor):
        device = tensor.get_device()
        device = torch.device('cpu') if device < 0 else torch.device(device)
        return device

    ## BerHu (reverse huber loss)
    #  adapted from https://github.com/xanderchf/MonoDepth-FPN-PyTorch/blob/
    #                   9494306b59c1aa9e8ab85ecea48a65052ff509a5/main_fpn.py
    @staticmethod
    def adaptive_berhu_loss(depth_gt, depth_est, mask, threshold=0.2):
        mask = mask.type(depth_gt.dtype).to(depth_gt.device)
        diff = torch.abs(depth_gt * mask - depth_est * mask)
        delta = threshold * torch.max(diff).item()

        l1_part = -F.threshold(-diff, -delta, 0.)
        l2_part = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        l2_part = l2_part / (2.*delta)

        loss = l1_part + l2_part
        loss = torch.mean(loss)
        return loss

    @staticmethod
    def huber_loss(x, y):
        return F.smooth_l1_loss(x, y, reduction='mean')

    def depth_loss(self, depth_gt, depth_est, mask):
        if depth_est.size() != depth_gt.size():
            depth_gt = F.interpolate(
                depth_gt, size=depth_est.size()[-2:], mode='nearest'
            ).clone()

        mask = mask > 0.5
        if torch.all(mask == 0):
            # print('entire mask 0')
            return torch.tensor(0.0, dtype=depth_gt.dtype,
                                device=P2MLoss.get_tensor_device(depth_gt),
                                requires_grad=depth_gt.requires_grad)
        else:
            return self.depth_loss_function(depth_est, depth_gt, mask)

    ##
    #  @param pred_coord tensor of coords
    def upsample_coords(self, coords, faces):
        # make sure that all clouds have the same number of points
        self.points_sampler.point_num = self.options.num_chamfer_upsample \
                                      - coords.size(1)
        if coords.size(1) < self.points_sampler.point_num:
            upsampled_coords, _ = self.points_sampler(
                coords, faces
            )
            # add original coords too for good measure
            # note: the original coords should be before the upsampled ones
            # otherwise the normal loss will be messed up
            upsampled_coords = torch.cat((coords, upsampled_coords), dim=1)
        else:
            upsampled_coords = coords
        return upsampled_coords

    ##
    #  @param pred_coord list of coords at different resolution
    def get_upsampled_coords(self, pred_coord):
        upsampled_pred_coord = [None for _ in range(len(pred_coord))]
        for i in range(len(pred_coord)):
            faces = self.ellipsoid.faces[i] \
                        .unsqueeze(0) \
                        .repeat(pred_coord[i].size(0), 1, 1)
            upsampled_pred_coord[i] = self.upsample_coords(pred_coord[i], faces)
        return upsampled_pred_coord

    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """

        device = device=targets["images"].device
        chamfer_loss, edge_loss, normal_loss, lap_loss, move_loss, depth_loss = 0., 0., 0., 0., 0., 0.
        rendered_vs_cv_depth_loss = torch.tensor(0., device=device)
        rendered_vs_gt_depth_loss = torch.tensor(0., device=device)
        lap_const = [0.2, 1., 1.]

        gt_coord, gt_normal, \
            gt_images = targets["points"], targets["normals"], targets["images"]
        gt_depths, masks = targets["depths"], targets["masks"]
        pred_coord = outputs["pred_coord"]
        pred_coord_before_deform = outputs["pred_coord_before_deform"]
        pred_depths = outputs["depths"]
        rendered_depths, rendered_depths_before_deform = \
            outputs["rendered_depths"], outputs["rendered_depths_before_deform"]
        image_loss = 0.

        # TODO uncommit this line
        if outputs["reconst"] is not None and self.options.weights.reconst != 0:
            image_loss = self.image_loss(gt_images, outputs["reconst"])

        if self.upsampled_chamfer_loss:
            upsampled_pred_coord = self.get_upsampled_coords(pred_coord)
        else:
            upsampled_pred_coord = pred_coord

        for i in range(3):
            dist1, dist2, idx1, idx2 = self.chamfer_dist(
                gt_coord, upsampled_pred_coord[i]
            )
            chamfer_loss += self.options.weights.chamfer[i] * ( \
                torch.mean(dist1) \
                + self.options.weights.chamfer_opposite \
                  * torch.mean(dist2) \
            )
            # normal_loss = 0
            normal_loss += self.normal_loss(
                gt_normal, idx2,
                pred_coord[i], self.edges[i]
            )
            edge_loss += self.edge_regularization(
                pred_coord[i], self.edges[i]
            )
            lap, move = self.laplace_regularization(
                pred_coord_before_deform[i],
                pred_coord[i], i
            )
            lap_loss += lap_const[i] * lap
            move_loss += lap_const[i] * move

            all_valid_masks = torch.ones(
                *(masks.size()), dtype=torch.uint8, device=masks.device
            )
            rendered_vs_cv_depth_loss += \
                self.options.weights.rendered_vs_cv_depth[i] \
                    * self.depth_loss(rendered_depths[i],
                                      pred_depths * masks, all_valid_masks)
            rendered_vs_gt_depth_loss += \
                self.options.weights.rendered_vs_gt_depth[i] \
                    * self.depth_loss(rendered_depths[i], gt_depths,
                                      all_valid_masks)
        depth_loss += self.depth_loss(gt_depths, pred_depths, masks)

        #
        if self.options.only_depth_training:
            loss = depth_loss
        else:
            loss = chamfer_loss + image_loss * self.options.weights.reconst + \
                   self.options.weights.laplace * lap_loss + \
                   self.options.weights.move * move_loss + \
                   self.options.weights.edge * edge_loss + \
                   self.options.weights.normal * normal_loss + \
                   self.options.weights.depth * depth_loss + \
                   rendered_vs_gt_depth_loss + \
                   rendered_vs_cv_depth_loss

        # loss = depth_loss

        loss = loss * self.options.weights.constant
        loss_summary = {
            "loss": loss,
            "loss_chamfer": chamfer_loss / np.sum(self.options.weights.chamfer),
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            "loss_move": move_loss,
            "loss_normal": normal_loss,
            "loss_depth": depth_loss,
            "loss_rendered_vs_cv_depth": \
                rendered_vs_cv_depth_loss \
                    / np.sum(self.options.weights.rendered_vs_cv_depth),
            "loss_rendered_vs_gt_depth": \
                rendered_vs_gt_depth_loss \
                    / np.sum(self.options.weights.rendered_vs_gt_depth),
        }
        nan_losses = {
            key: value.item() for key, value in loss_summary.items()
            if torch.isnan(value)
        }
        nan_rendered_depths = [torch.any(torch.isnan(i)).item()
                              for i in rendered_depths]
        nan_pred_coord = [torch.any(torch.isnan(i)).item() for i in pred_coord]
        if np.any(nan_losses):
            print('nan_losses:', nan_losses)
            print('rendered losses:', rendered_vs_gt_depth_loss,
                                      rendered_vs_cv_depth_loss)
        if np.any(nan_rendered_depths):
            print('rendered depths nan:', nan_rendered_depths)
        if np.any(nan_pred_coord):
            print('pred coord nan:', nan_pred_coord)
        if torch.any(torch.isnan(pred_depths)):
            print('pred depth nan')
        return loss, loss_summary

