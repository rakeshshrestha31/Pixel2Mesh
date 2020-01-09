import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.chamfer_wrapper import ChamferDist


class P2MLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()

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

    def laplace_regularization(self, input1, input2,
                               block_idx, laplace_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """

        lap1 = self.laplace_coord(input1, laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, laplace_idx[block_idx])
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

    @staticmethod
    def depth_loss(depth_gt, depth_est, mask):
        mask = mask > 0.5
        return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

    def forward(self, outputs, targets, ellipsoids):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :param ellipsoids: initial ellipsoid objects
        :return: loss, loss_summary (dict)
        """

        chamfer_loss, edge_loss, normal_loss, lap_loss, move_loss, depth_loss = 0., 0., 0., 0., 0., 0.
        lap_const = [0.2, 1., 1.]

        gt_coord, gt_normal, gt_images, gt_depth, mask = targets["points"], targets["normals"], targets["images"], targets["depth"], targets["mask"]
        pred_coord, pred_coord_before_deform = outputs["pred_coord"], outputs["pred_coord_before_deform"]
        pred_depth = outputs["depth"]
        image_loss = 0.

        # TODO uncommit this line
        if outputs["reconst"] is not None and self.options.weights.reconst != 0:
            image_loss = self.image_loss(gt_images, outputs["reconst"])

        ellipsoids_info = extract_meshes_info(ellipsoids, gt_coord.get_device())

        for i in range(3):
            for batch_idx in range(gt_coord.size(0)):
                dist1, dist2, idx1, idx2 = self.chamfer_dist(
                    gt_coord[batch_idx:batch_idx+1],
                    pred_coord[i][batch_idx]
                )
                chamfer_loss += self.options.weights.chamfer[i] * ( \
                    torch.mean(dist1) \
                    + self.options.weights.chamfer_opposite \
                      * torch.mean(dist2) \
                )
                # normal_loss = 0
                normal_loss += self.normal_loss(
                    gt_normal[batch_idx:batch_idx+1], idx2,
                    pred_coord[i][batch_idx],
                    ellipsoids_info['edges'][batch_idx][i]
                )
                edge_loss += self.edge_regularization(
                    pred_coord[i][batch_idx],
                    ellipsoids_info['edges'][batch_idx][i]
                )
                lap, move = self.laplace_regularization(
                    pred_coord_before_deform[i][batch_idx],
                    pred_coord[i][batch_idx], i,
                    ellipsoids_info['laplace_indices'][batch_idx]
                )
                lap_loss += lap_const[i] * lap
                move_loss += lap_const[i] * move

        depth_loss += self.depth_loss(gt_depth, pred_depth, mask)
        #
        loss = chamfer_loss + image_loss * self.options.weights.reconst + \
               self.options.weights.laplace * lap_loss + \
               self.options.weights.move * move_loss + \
               self.options.weights.edge * edge_loss + \
               self.options.weights.normal * normal_loss + \
               self.options.weights.depth * depth_loss

        # loss = depth_loss

        loss = loss * self.options.weights.constant

        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            "loss_move": move_loss,
            "loss_normal": normal_loss,
            "loss_depth": depth_loss,
        }

def extract_meshes_info(meshes, device):
    """extract information (laplace indices, edges) from meshes
    :param: meshes list of meses
    :param: is_cuda boolean
    :return: dictionary of mesh_info"""
    laplace_indices = []
    edges = []
    for ellipsoid in meshes:
        laplace_indices.append(nn.ParameterList([
            nn.Parameter(idx, requires_grad=False)
            for idx in ellipsoid.laplace_idx]
        ))
        edges.append(nn.ParameterList([
            nn.Parameter(e, requires_grad=False)
            for e in ellipsoid.edges
        ]))

    if device >= 0:
        laplace_indices = [
            i.cuda(device) for i in laplace_indices
        ]
        edges = [i.cuda(device) for i in edges]

    return {
        "laplace_indices": laplace_indices,
        "edges": edges
    }
