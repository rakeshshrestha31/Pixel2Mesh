import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.layers.gprojection_xyz import GProjection as GProjectionXYZ

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

        self.nn_encoder, self.nn_decoder = get_backbone(options, self.freeze_cv)
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
        self.projection = GProjectionXYZ(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
                                      tensorflow_compatible=options.align_with_tensorflow)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])

        if self.gconv_skip_connection == 'concat':
            self.gconv1 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[0])
            self.gconv2 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[1])
            self.gconv3 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[2])

    def src2ref(self, pts, ref_proj, src_proj):
        with torch.no_grad():
            z_axis = torch.ones([pts.shape[0], pts.shape[1], 1], device=pts.device)
            pts = torch.cat([pts, z_axis], -1)
            src2ref_proj = torch.matmul(ref_proj[:, 0, :, :], torch.inverse(src_proj[:, 0, :, :]))
            pts_ref = torch.matmul(src2ref_proj, pts.permute(0,2,1)).permute(0,2,1)
            pts = pts_ref[:, :, :3] / pts_ref[:, :, 3:4]

        return  pts


    @staticmethod
    def get_tensor_device(tensor):
        return tensor.get_device() if tensor.is_cuda else torch.device('cpu')


    ## projects points' assigned features
    def assigned_projection(self, points, features, points_assignments,
                            img_shape, depth_values):
        num_views = len(features)
        batch_size, num_points, num_coords = points.size()
        proj_features = torch.zeros(
            (batch_size, num_points, self.features_dim),
            dtype=points.dtype, device = self.get_tensor_device(points)
        )

        ## Note: Since the number of selected points from each batch
        ## can be different, we have to do this flattening stuff
        # batch and num_points combined to one dim
        flattened_points = points.contiguous().view(-1, num_coords)

        for view_idx in range(num_views):
            selected_indices = (points_assignments == view_idx)
            # check how many points of each batch selected

            selected_points = flattened_points[selected_indices.view(-1)]
            # batch and channels combined to make batch size look like 1
            flattened_features = [i.view(-1, *(i.shape[2:])).unsqueeze(0)
                                  for i in features[view_idx][0]]
            selected_proj = self.projection(
                img_shape, flattened_features,
                selected_points.unsqueeze(0), depth_values
            )
            proj_features += self.unflatten_features(
                selected_proj, selected_indices, num_points, batch_size
            )

        return proj_features

    ##
    #  @param featues tensor of size
    #       1 x (batch1_indices + batch2_indices ...) x
    #       (num_features*batch_size + 3)
    #       Note: all features aren't necessary, only corresponding features
    #       from selected_indices
    #  @return tensor fo size batch_size x num_points x (features + 3)
    def unflatten_features(self, features, selected_indices,
                           num_points, batch_size):
        unflattened_features = torch.zeros(
            (batch_size, num_points, self.features_dim),
            dtype=features.dtype, device = self.get_tensor_device(features)
        )
        batch_selected_count = selected_indices.sum(dim=1)
        batch_selected_cumsum = torch.cumsum(batch_selected_count, 0)
        batch_selected_cumsum = [0] + batch_selected_cumsum.tolist()

        for batch in range(batch_size):
            batch_start, batch_end = batch_selected_cumsum[batch:batch+2]
            feature_start, feature_end = [i * self.nn_encoder.features_dim
                                          for i in [batch, batch+1] ]
            batch_proj = torch.cat([
                # feature from cost volume
                features[:, batch_start:batch_end, feature_start:feature_end],
                # 3D coordinates
                features[:, batch_start:batch_end, -self.coord_dim:]
            ], dim=-1)
            unflattened_features[batch, selected_indices[batch]] += \
                    batch_proj.squeeze(0)
        return unflattened_features


    def forward(self, img, proj, depth_values, points_assignments):
        batch_size = img.size(0)

        #Multi-view start
        img_list = torch.unbind(img, 1)
        proj_list = torch.unbind(proj, 1)
        num_views = len(img_list)
        out_encoder = self.nn_encoder(img, proj_list, depth_values)
        feature_list = out_encoder["features"][0] #[self.nn_encoder(i) for i in img_list]
        ref_feature, src_feature_list = feature_list[0], feature_list[1:]
        ref_proj, src_proj_list       = proj_list[0], proj_list[1:]

        img_shape = self.projection.image_feature_shape(img[0])
        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        x = self.assigned_projection(
            init_pts, out_encoder["features"], points_assignments[0],
            img_shape, depth_values
        )

        # x1 shape is torch.Size([16, 156, 3]), x_h
        x1, x_hidden = self.gcns[0](x)

        if self.gconv_skip_connection == 'concat':
            x1 = self.gconv1(torch.cat((x1, init_pts), -1))
        elif self.gconv_skip_connection == 'add':
            x1 = x1 + init_pts

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.assigned_projection(
            x1, out_encoder["features"], points_assignments[0],
            img_shape, depth_values
        )

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
        x = self.assigned_projection(
            x2, out_encoder["features"], points_assignments[1],
            img_shape, depth_values
        )

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

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(ref_feature)
        else:
            reconst = None
        # multi-view end

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst,
            "depths": out_encoder["depths"],
        }
