import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection


class P2MModel(nn.Module):

    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
        super(P2MModel, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[0], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[1], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                        ellipsoid.adj_mat[2], activation=self.gconv_activation)
        ])

        self.unpooling = nn.ModuleList([
            GUnpooling(ellipsoid.unpool_idx[0]),
            GUnpooling(ellipsoid.unpool_idx[1])
        ])

        # if options.align_with_tensorflow:
        #     self.projection = GProjection
        # else:
        #     self.projection = GProjection
        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
                                      tensorflow_compatible=options.align_with_tensorflow)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])
    def src2ref(self, pts, ref_proj, src_proj):
        with torch.no_grad():
            z_axis = torch.ones([pts.shape[0], pts.shape[1], 1], device=pts.device)
            pts = torch.cat([pts, z_axis], -1)
            src2ref_proj = torch.matmul(ref_proj[:, 0, :, :], torch.inverse(src_proj[:, 0, :, :]))
            pts_ref = torch.matmul(src2ref_proj, pts.permute(0,2,1)).permute(0,2,1)
            pts = pts_ref[:, :, :3] / pts_ref[:, :, 3:4]

        return  pts



    def forward(self, img, proj):
        batch_size = img.size(0)

        #Multi-view start
        img_list = torch.unbind(img, 1)
        proj_list = torch.unbind(proj, 1)
        num_views = len(img_list)
        feature_list = [self.nn_encoder(i) for i in img_list]
        ref_feature, src_feature_list = feature_list[0], feature_list[1:]
        ref_proj, src_proj_list       = proj_list[0], proj_list[1:]

        img_shape = self.projection.image_feature_shape(img[0])
        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        x = self.projection(img_shape, ref_feature, init_pts)
        for src_feature, src_proj in zip(src_feature_list, src_proj_list):
            init_pts_src = self.src2ref(init_pts, ref_proj, src_proj)
            x_src = self.projection(img_shape, src_feature, init_pts_src)
            x += x_src
        x = x / num_views

        x1, x_hidden = self.gcns[0](x)  # x1 shape is torch.Size([16, 156, 3]), x_h

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, ref_feature, x1)
        for src_feature, src_proj in zip(src_feature_list, src_proj_list):
            x1_src = self.src2ref(x1, ref_proj, src_proj)
            x_src = self.projection(img_shape, src_feature, x1_src)
            x += x_src
        x = x / num_views
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))

        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)  #

        # GCN Block 3
        x = self.projection(img_shape, ref_feature, x2)  # x2 shape is torch.Size([16, 618, 3])
        for src_feature, src_proj in zip(src_feature_list, src_proj_list):
            x2_src = self.src2ref(x2, ref_proj, src_proj)
            x_src = self.projection(img_shape, src_feature, x2_src)
            x += x_src
        x = x / num_views
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(ref_feature)
        else:
            reconst = None
        # multi-view end

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst
        }
