import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.costvolume import MVSNet
from models.backbones.vgg16 import VGG16P2M
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.layers.gprojection_xyz import GProjection as GProjectionXYZ

import config

class P2MModel(nn.Module):

    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos, freeze_cv=False):
        super(P2MModel, self).__init__()
        self.freeze_cv = freeze_cv
        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation
        self.gconv_skip_connection = options.gconv_skip_connection.lower()

        self.mvsnet = MVSNet(freeze_cv=self.freeze_cv)
        self.vgg = VGG16P2M(n_classes_input=1, pretrained=False)
        # x3 for three views
        self.features_dim = (self.vgg.features_dim + self.coord_dim) * 3
        print("===> number of P2MModel features:", self.features_dim)

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
        self.projection = GProjection(
            mesh_pos, camera_f, camera_c, bound=options.z_threshold,
            tensorflow_compatible=options.align_with_tensorflow
        )

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])

        if self.gconv_skip_connection == 'concat':
            self.gconv1 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[0])
            self.gconv2 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[1])
            self.gconv3 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[2])


    @staticmethod
    def get_tensor_device(tensor):
        return tensor.get_device() if tensor.is_cuda else torch.device('cpu')

    @staticmethod
    def flatten_batch_view(tensor):
        return tensor.view(-1, *(tensor.size()[2:]))

    @staticmethod
    def unflatten_batch_view(tensor, batch_size):
        return tensor.view(batch_size, -1, *(tensor.size()[1:]))

    @staticmethod
    def transform_points(pts, T):
        return P2MModel.make_points_unhomogeneous(torch.bmm(
            P2MModel.make_points_homogeneous(pts), T.transpose(-1, -2)
        ))

    @staticmethod
    def make_points_homogeneous(pts):
        return torch.cat(
            (
                pts,
                torch.ones( *(pts.size()[:-1]), 1,
                            dtype=pts.dtype, device=pts.device )
            ), dim=-1
        )

    @staticmethod
    def make_points_unhomogeneous(pts):
        return pts[:, :, :3]

    ##
    #  @param img_feats list with elements batch x view x channel x height x width
    #  @param pts batch x num_points x 3
    #  @return view pooled tensor of size batch x total_channels x height x width
    def cross_view_feature_pooling(self, img_shape, img_feats, pts, proj_mat):
        T_ref_world = proj_mat[:, 0, 0]
        T_world_ref = torch.inverse(T_ref_world)
        num_views = img_feats[0].size(1)
        transformed_features = []

        for view_idx in range(num_views):
            T_view_world = proj_mat[:, view_idx, 0]
            T_view_ref = torch.bmm(T_view_world, T_world_ref)
            pts_view = self.transform_points(pts, T_view_ref)
            view_features = [ i[:, view_idx].contiguous() for i in img_feats ]
            x = self.projection(img_shape, view_features, pts_view)
            transformed_features.append(x)
        return torch.cat(transformed_features, dim=-1)


    def forward(self, input_batch):
        img = input_batch["images"]
        proj_mat = input_batch["proj_matrices"]
        depth_values = input_batch["depth_values"]
        masks = input_batch["masks"]
        batch_size = img.size(0)

        out_mvsnet = self.mvsnet(input_batch)
        # unsqeeze 1 to simulate channels (single channel)
        vgg_input = self.flatten_batch_view(
            out_mvsnet["depths"] * masks
        ).unsqueeze(1)
        vgg_input = F.interpolate(
            vgg_input, size=[config.IMG_SIZE, config.IMG_SIZE], mode='nearest'
        )
        vgg_feats = self.vgg(vgg_input)
        img_feats = [
            self.unflatten_batch_view(i, batch_size) for i in vgg_feats
        ]

        img_shape = self.projection.image_feature_shape(img[0])
        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        # GCN Block 1
        # x1 shape is torch.Size([16, 156, 3]), x_h
        x = self.cross_view_feature_pooling(
            img_shape, img_feats, init_pts, proj_mat
        )
        x1, x_hidden = self.gcns[0](x)

        if self.gconv_skip_connection == 'concat':
            x1 = self.gconv1(torch.cat((x1, init_pts), -1))
        elif self.gconv_skip_connection == 'add':
            x1 = x1 + init_pts

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.cross_view_feature_pooling(img_shape, img_feats, x1, proj_mat)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))

        # after deformation 2
        x2, x_hidden = self.gcns[1](x)
        if self.gconv_skip_connection == 'concat':
            x2 = self.gconv2(torch.cat((x2, x1_up), -1))
        elif self.gconv_skip_connection == 'add':
            x2 = x2 + x1_up

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        # x2 shape is torch.Size([16, 618, 3])
        x = self.cross_view_feature_pooling(img_shape, img_feats, x2, proj_mat)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))

        x3, _ = self.gcns[2](x)

        if self.gconv_activation:
            x3 = F.relu(x3)

        # after deformation 3
        x3 = self.gconv(x3)
        if self.gconv_skip_connection == 'concat':
            x3 = self.gconv3(torch.cat((x3, x2_up), -1))
        elif self.gconv_skip_connection == 'add':
            x3 = x3 + x2_up

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": None,
            "depths": out_mvsnet["depths"],
        }
