import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.chamfer_wrapper import ChamferDist
from models.layers.sample_points import PointSampler
from utils.depth_backprojection import get_points_from_depths
import numpy as np
import functools

import config

class P2MLoss(nn.Module):
    def __init__(self, options, ellipsoid, upsampled_chamfer_loss=False):
        super().__init__()
        self.options = options
        # MSELoss/L1Loss likely to give NaN gradients if input == output
        # self.l1_loss = nn.L1Loss(reduction='mean')
        # self.l2_loss = nn.MSELoss(reduction='mean')

        self.epsilon = 1e-10
        self.l1_loss = lambda x, y: (torch.abs(x - y) + self.epsilon).mean()
        self.l2_loss = lambda x, y: ((torch.abs(x - y) + self.epsilon) ** 2) \
                                        .mean()

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
        # Why so sparse? Only uses 1 edge incident on a vertex
        # to compute normal loss
        edges = F.normalize(pred_points[:, adj_list[:, 0]] - pred_points[:, adj_list[:, 1]], dim=2)
        nearest_normals = torch.stack([t[i] for t, i in zip(gt_normal, indices.long())])
        normals = F.normalize(nearest_normals[:, adj_list[:, 0]], dim=2)
        cosine = torch.abs(torch.sum(edges * normals, 2))
        return torch.mean(cosine)

    @staticmethod
    def upsampled_normal_loss(gt_normals, pred_normals, indices):
        nearest_gt_normals = torch.stack([
            t[i] for t, i in zip(gt_normals, indices.long())]
        )
        sine = torch.sum(nearest_gt_normals * pred_normals, dim=2)
        cosine = torch.sqrt(1 - (sine + 1e-10)**2 + 1e-10)
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
    def adaptive_huber_loss(input, target, beta=1./9, reduction='mean'):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if reduction == 'mean':
            return loss.mean()
        return loss.sum()

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
    # @param points num_points x 3
    # @return num_points x 4
    @staticmethod
    def make_homogeneous(points):
        return torch.cat((points, torch.ones_like(points[:, 0:1])), dim=-1)

    ##
    # @param points num_points x 4
    # @return num_points x 3
    @staticmethod
    def make_unhomogeneous(points):
        return points[:, :3]

    ##
    # @param points_old num_points x 3
    # @return num_points x 3
    @staticmethod
    def transform(T_new_old, points_old):
        points_old = P2MLoss.make_homogeneous(points_old)
        # p_new = T_new_old * p_old = p_old' * T_new_old'
        points_new = P2MLoss.make_unhomogeneous(
            torch.matmul(points_old,  T_new_old.transpose(0, 1))
        )
        return points_new

    def backprojected_depth_loss(self, pred_depths, pred_coord,
                                 intrinsics, extrinsics):
        depth_resize_factor = config.IMG_SIZE / pred_depths.size(-1)
        intrinsics = intrinsics[:2] / depth_resize_factor
        backprojected_depth_points = get_points_from_depths(
            pred_depths, intrinsics, extrinsics
        )
        batch_size = len(backprojected_depth_points)
        num_views = len(backprojected_depth_points[0])
        loss = 0.0
        num_points = 0
        # these operations cannot be batched since different batch/views
        # will have different number of points
        for batch_idx in range(batch_size):
            T_ref_world = extrinsics[batch_idx, 0]
            for view_idx in range(num_views):
                points_world = backprojected_depth_points[batch_idx][view_idx]
                points_ref = P2MLoss.transform(T_ref_world, points_world)
                # only backproj depth to pred coord loss cuz
                # the depth won't have all the 3D model points due to occlusion
                for resolution_idx in range(len(pred_coord)):
                    dist, _, _, _ = self.chamfer_dist(
                        points_ref.unsqueeze(0),
                        pred_coord[resolution_idx][batch_idx].unsqueeze(0)
                    )
                    loss += self.adaptive_huber_loss(
                        dist, torch.zeros_like(dist),
                        beta=self.options.weights.backprojected_depth_beta,
                        reduction='sum'
                    )
                    num_points += points_ref.size(0)
        loss = self.options.weights.backprojected_depth * loss / num_points
        return loss
    ##
    #  @param pred_coord tensor of coords
    def upsample_coords(self, coords, ellipsoid_idx):
        # make sure that all clouds have the same number of points
        self.points_sampler.point_num = self.options.num_chamfer_upsample \
                                      - coords.size(1)
        coords_normals = \
                self.ellipsoid.get_vertex_normals(coords, ellipsoid_idx)

        if coords.size(1) < self.points_sampler.point_num:
            faces = self.ellipsoid.faces[ellipsoid_idx] \
                        .unsqueeze(0).expand(coords.size(0), -1, -1)
            upsampled_coords, upsampled_normals = self.points_sampler(
                coords, faces
            )
            # add original coords too for good measure
            # note: the original coords should be before the upsampled ones
            # otherwise the normal loss will be messed up
            upsampled_coords = torch.cat((coords, upsampled_coords), dim=1)
            upsampled_normals = torch.cat(
                (coords_normals, upsampled_normals), dim=1
            )
        else:
            upsampled_coords = coords
            upsampled_normals = coords_normals
        return upsampled_coords, upsampled_normals

    ##
    #  @param pred_coord list of coords at different resolution
    def get_upsampled_coords(self, pred_coord):
        upsampled_pred_coord = [None for _ in range(len(pred_coord))]
        upsampled_normals = [None for _ in range(len(pred_coord))]
        for i in range(len(pred_coord)):
            upsampled_pred_coord[i], upsampled_normals[i] = \
                    self.upsample_coords(pred_coord[i], i)
        return upsampled_pred_coord, upsampled_normals

    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """

        device = device=targets["images"].device
        chamfer_loss, edge_loss, lap_loss, move_loss, depth_loss = 0., 0., 0., 0., 0.
        normal_loss = torch.tensor(0., device=device)
        rendered_vs_cv_depth_loss = torch.tensor(0., device=device)
        rendered_vs_gt_depth_loss = torch.tensor(0., device=device)
        backprojected_depth_loss = torch.tensor(0., device=device)
        lap_const = [0.2, 1., 1.]

        gt_coord, gt_normal, \
            gt_images = targets["points"], targets["normals"], targets["images"]
        gt_depths, masks = targets["depths"], targets["masks"]
        pred_coord = outputs["pred_coord"]
        pred_coord_before_deform = outputs["pred_coord_before_deform"]
        pred_depths = outputs["depths"]
        masked_pred_depths = pred_depths * masks
        rendered_depths, rendered_depths_before_deform = \
            outputs["rendered_depths"], outputs["rendered_depths_before_deform"]
        image_loss = 0.

        # TODO uncommit this line
        if outputs["reconst"] is not None and self.options.weights.reconst != 0:
            image_loss = self.image_loss(gt_images, outputs["reconst"])

        if self.upsampled_chamfer_loss:
            upsampled_pred_coord, upsampled_normals = self.get_upsampled_coords(pred_coord)
        else:
            upsampled_pred_coord = pred_coord
            upsampled_normals = [
                self.ellipsoid.get_vertex_normals(pred_coord[i], i)
                for i in range(len(pred_coord))
            ]

        for i in range(len(pred_coord)):
            dist1, dist2, idx1, idx2 = self.chamfer_dist(
                gt_coord, upsampled_pred_coord[i]
            )
            chamfer_loss += self.options.weights.chamfer[i] * ( \
                torch.mean(dist1) \
                + self.options.weights.chamfer_opposite \
                  * torch.mean(dist2) \
            )
            # normal_loss = 0
            if self.options.upsampled_normal_loss:
                tmp_loss = self.upsampled_normal_loss(
                    gt_normal, upsampled_normals[i], idx2
                )
                if not torch.any(torch.isnan(tmp_loss)):
                    normal_loss += tmp_loss
            else:
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
                                      masked_pred_depths, all_valid_masks)
            rendered_vs_gt_depth_loss += \
                self.options.weights.rendered_vs_gt_depth[i] \
                    * self.depth_loss(rendered_depths[i], gt_depths,
                                      all_valid_masks)

        if self.options.use_backprojected_depth_loss:
            extrinsics = targets['proj_matrices'][:, :, 0].contiguous()
            intrinsics = targets['proj_matrices'][0, 0, 1, :3, :3].contiguous()
            backprojected_depth_loss += self.backprojected_depth_loss(
                masked_pred_depths, upsampled_pred_coord, intrinsics, extrinsics
            )
        depth_loss += self.depth_loss(gt_depths, pred_depths, masks)

        #
        if self.options.only_depth_training:
            loss = depth_loss
        else:
            loss = chamfer_loss + \
                   backprojected_depth_loss + \
                   image_loss * self.options.weights.reconst + \
                   self.options.weights.laplace * lap_loss + \
                   self.options.weights.move * move_loss + \
                   self.options.weights.edge * edge_loss + \
                   self.options.weights.normal * normal_loss + \
                   self.options.weights.depth * depth_loss + \
                   rendered_vs_gt_depth_loss + \
                   rendered_vs_cv_depth_loss

        # loss = depth_loss

        loss = loss * self.options.weights.constant

        # avoid divisions by zero
        def get_weights_norm(weights):
            return np.sum(weights) if np.any(weights) else 1.0
        norm_rendered_vs_gt_depth_weights = \
                get_weights_norm(self.options.weights.rendered_vs_gt_depth)
        norm_rendered_vs_cv_depth_weights = \
                get_weights_norm(self.options.weights.rendered_vs_cv_depth)
        loss_summary = {
            "loss": loss,
            "loss_chamfer": chamfer_loss / np.sum(self.options.weights.chamfer),
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            "loss_move": move_loss,
            "loss_normal": normal_loss,
            "loss_depth": depth_loss,
            "loss_rendered_vs_cv_depth":
                rendered_vs_cv_depth_loss / norm_rendered_vs_cv_depth_weights,
            "loss_rendered_vs_gt_depth": \
                rendered_vs_gt_depth_loss / norm_rendered_vs_gt_depth_weights,
            "loss_backprojected_depth": \
                    backprojected_depth_loss \
                        / self.options.weights.backprojected_depth
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

