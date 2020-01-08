import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection


class P2MModel(nn.Module):

    def __init__(self, options, camera_f, camera_c, mesh_pos):
        super(P2MModel, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                        activation=self.gconv_activation)
        ])

        self.unpooling = nn.ModuleList([
            GUnpooling(),
            GUnpooling()
        ])

        # if options.align_with_tensorflow:
        #     self.projection = GProjection
        # else:
        #     self.projection = GProjection
        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
                                      tensorflow_compatible=options.align_with_tensorflow)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim)

    def src2ref(self, pts, ref_proj, src_proj):
        with torch.no_grad():
            z_axis = torch.ones([pts.shape[0], pts.shape[1], 1], device=pts.device)
            pts = torch.cat([pts, z_axis], -1)
            src2ref_proj = torch.matmul(ref_proj[:, 0, :, :], torch.inverse(src_proj[:, 0, :, :]))
            pts_ref = torch.matmul(src2ref_proj, pts.permute(0,2,1)).permute(0,2,1)
            pts = pts_ref[:, :, :3] / pts_ref[:, :, 3:4]

        return  pts


    def forward(self, img, proj, ellipsoids, depth_values=None):
        batch_size = img.size(0)
        assert(batch_size, len(ellipsoids))
        init_pts = []
        device = next(self.parameters()).get_device()
        for ellipsoid in ellipsoids:
            # move stuffs to GPU if needed
            if device >= 0:
                ellipsoid.coord = ellipsoid.coord.cuda(device)
                for i in range(3):
                    ellipsoid.adj_mat[i] = ellipsoid.adj_mat[i].cuda(device)
            coord_param = nn.Parameter(ellipsoid.coord, requires_grad=False) \
                .unsqueeze(0).expand(1, -1, -1)
            init_pts.append(coord_param)

        #Multi-view start
        img_list = torch.unbind(img, 1)
        proj_list = torch.unbind(proj, 1)
        num_views = len(img_list)
        out_encoder = self.nn_encoder(img_list, proj_list, depth_values)
        feature_list = out_encoder["features"]#[self.nn_encoder(i) for i in img_list]
        ref_feature, src_feature_list = feature_list[0], feature_list[1:]
        ref_proj, src_proj_list       = proj_list[0], proj_list[1:]

        img_shape = self.projection.image_feature_shape(img[0])

        x1s = []
        x2s = []
        x3s = []
        x1_ups = []
        x2_ups = []

        def batch_from_list(lst, batch_idx):
            return [i[batch_idx:batch_idx+1] for i in lst]

        def batch_from_tensor(tensor, batch_idx):
            return tensor[batch_idx:batch_idx+1]

        # the ref_feature (and others) are list of feature tensors
        # with each element's first dimension being batch size
        for batch_idx in range(batch_size):
            x = self.projection(
                img_shape,
                batch_from_list(ref_feature, batch_idx),
                init_pts[batch_idx]
            )

            # x1 shape is torch.Size([16, 156, 3]), x_h
            x1, x_hidden = self.gcns[0](x, ellipsoids[batch_idx].adj_mat[0])

            # before deformation 2
            x1_up = self.unpooling[0](x1, ellipsoids[batch_idx].unpool_idx[0])

            # GCN Block 2
            x = self.projection(img_shape,
                                batch_from_list(ref_feature, batch_idx), x1)

            x = self.unpooling[0](
                torch.cat([x, x_hidden], 2),
                ellipsoids[batch_idx].unpool_idx[0]
            )

            # after deformation 2
            x2, x_hidden = self.gcns[1](x, ellipsoids[batch_idx].adj_mat[1])

            # before deformation 3
            x2_up = self.unpooling[1](x2, ellipsoids[batch_idx].unpool_idx[1])

            # GCN Block 3
            # x2 shape is torch.Size([16, 618, 3])
            x = self.projection(
                img_shape, batch_from_list(ref_feature, batch_idx), x2
            )

            x = self.unpooling[1](
                torch.cat([x, x_hidden], 2),
                ellipsoids[batch_idx].unpool_idx[1]
            )

            x3, _ = self.gcns[2](x, ellipsoids[batch_idx].adj_mat[2])
            if self.gconv_activation:
                x3 = F.relu(x3)
            # after deformation 3
            x3 = self.gconv(x3, ellipsoids[batch_idx].adj_mat[2])

            x1s.append(x1)
            x2s.append(x2)
            x3s.append(x3)
            x1_ups.append(x1_up)
            x2_ups.append(x2_up)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(ref_feature)
        else:
            reconst = None
        # multi-view end

        return {
            "pred_coord": [x1s, x2s, x3s],
            "pred_coord_before_deform": [init_pts, x1_ups, x2_ups],
            "reconst": reconst,
            "depth": out_encoder["depth"],
        }
