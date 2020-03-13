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
from models.layers.attention_feature_pooling import AttentionFeaturePooling
import config

import neural_renderer as nr
import cv2
import numpy as np
import functools

class P2MModel(nn.Module):

    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos,
                 freeze_cv=False, mvsnet_checkpoint=''):
        super(P2MModel, self).__init__()
        self.freeze_cv = freeze_cv
        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation
        self.gconv_skip_connection = options.gconv_skip_connection.lower()
        self.options = options

        self.mvsnet = MVSNet(freeze_cv=self.freeze_cv,
                             checkpoint=mvsnet_checkpoint)
        if options.use_rgb_features:
            self.rgb_vgg = VGG16P2M(n_classes_input=3, pretrained=False)
        self.depth_vgg = VGG16P2M(n_classes_input=2, pretrained=False)
        # features: RGB-features + Depth features + cost-volume features
        #               + Depth + local point coordinates
        if self.options.feature_fusion_method == 'attention':
            pre_fusion_features_dim = \
                (self.rgb_vgg.features_dim if options.use_rgb_features else 0) \
                + self.depth_vgg.features_dim \
                + self.mvsnet.features_dim + 1
            post_fusion_features_dim = pre_fusion_features_dim
            # TODO: the num_heads is too high. But it can't be lowered
            # since it needs to be a factor of the features dim
            self.attention_model = AttentionFeaturePooling(
                pre_fusion_features_dim, post_fusion_features_dim,
                num_heads=43, max_views=3
            )
            # local point coordinates for all views
            self.features_dim = post_fusion_features_dim + (3 * 3)
        else:
            self.features_dim = ( \
                (self.rgb_vgg.features_dim if options.use_rgb_features else 0) \
                + self.depth_vgg.features_dim
                + self.mvsnet.features_dim \
                + 1 + 3 \
            ) * 3
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
        self.projection_2d = GProjection(
            mesh_pos, camera_f, camera_c, bound=options.z_threshold,
            tensorflow_compatible=options.align_with_tensorflow
        )
        self.projection_3d = GProjectionXYZ(
            mesh_pos, camera_f, camera_c, bound=options.z_threshold,
            tensorflow_compatible=options.align_with_tensorflow
        )

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])

        if self.gconv_skip_connection == 'concat':
            self.gconv1 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[0])
            self.gconv2 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[1])
            self.gconv3 = GConv(in_features=6, out_features=3, adj_mat=ellipsoid.adj_mat[2])

        self.ellipsoid = ellipsoid
        self.init_renderer(camera_f, camera_c)

    def init_renderer(self, camera_f, camera_c):
        self.renderer = nr.Renderer(
            camera_mode='projection', near=1e-4, far=1.5,
            anti_aliasing=False
        )
        self.camera_k = torch.tensor(
            [[camera_f[0], 0, camera_c[0]],
             [0, camera_f[1], camera_c[1]],
             [0, 0, 1]],
            dtype=torch.float32
        )
        self.dist_coeffs = torch.zeros(5, dtype=torch.float32)

        # conversion from shapenet convention (East-Up_South)
        # to renderer convention (East-Down-North)
        # final rotation: R_renderer_shapenet * extrinsics
        # inverse y and z, equivalent to inverse x, but gives positive z
        rvec = np.array([np.pi, 0., 0.], dtype=np.float32)
        R = cv2.Rodrigues(rvec)[0]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        self.T_renderer_shapenet = torch.from_numpy(T)
        self.T_shapenet_renderer = torch.inverse(self.T_renderer_shapenet)

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
    def flatten_2d_list(list_2d):
        return [j for i in list_2d for j in i]

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

    @staticmethod
    # convenience function for projecting features of a view
    def project_view_features(features, points, view_idx,
                              img_shape, projection_functor):
        view_features = [ i[:, view_idx].contiguous() for i in features ]
        proj_features = projection_functor(img_shape, view_features, points)
        return proj_features

    ##
    #  @param img_feats list with elements batch x view x channel x height x width
    #  @param depth_feats list with elements batch x view x channel x height x width
    #  @param costvolume_feats list with elements
    #               batch x view x channel x z x height x width
    #  @param pts batch x num_points x 3
    #  @return view pooled tensor of size batch x total_channels x height x width
    def cross_view_feature_pooling(
            self, img_shape, img_feats, depth_feats,
            costvolume_feats, depth, pts, proj_mat, depth_values
    ):
        T_ref_world = proj_mat[:, 0, 0]
        T_world_ref = torch.inverse(T_ref_world)
        num_views = img_feats[0].size(1)
        transformed_features = []

        projection_3d = functools.partial(self.projection_3d,
                                          depth_values=depth_values)
        pts_coordinates = []
        for view_idx in range(num_views):
            T_view_world = proj_mat[:, view_idx, 0]
            T_view_ref = torch.bmm(T_view_world, T_world_ref)
            pts_view = self.transform_points(pts, T_view_ref)
            # bound functions for easy features projection
            project_2d_features = functools.partial(
                self.project_view_features, points=pts_view, view_idx=view_idx,
                img_shape=img_shape, projection_functor=self.projection_2d
            )
            project_3d_features = functools.partial(
                self.project_view_features, points=pts_view, view_idx=view_idx,
                img_shape=img_shape, projection_functor=projection_3d
            )

            proj_feats = []
            pts_coordinates.append(pts_view)
            # features from RGB image
            if self.options.use_rgb_features:
                x_img_feats = project_2d_features(features=img_feats)
                proj_feats.append(x_img_feats)

            # features from depths
            x_depth_feats = project_2d_features(features=depth_feats)
            proj_feats.append(x_depth_feats)

            # features from costvolume
            x_costvolume_feats = project_3d_features(features=costvolume_feats)
            proj_feats.append(x_costvolume_feats)

            # features from predicted depth
            x_depth = project_2d_features(features=[depth.unsqueeze(2)])
            proj_feats.append(x_depth)

            transformed_features.append(torch.cat(proj_feats, dim=-1))
        return self.fuse_features(pts_coordinates, transformed_features)

    ##
    #  @param coords list of points coordinates in each views' frame
    #  @param features list of features of size
    #           batch x num_points x num_channels for each view
    def fuse_features(self, coords, features):
        if self.options.feature_fusion_method == 'concat':
            # the zip and flatten are just for compatibility with checkpoints
            # generated by previous version of code
            # otherwise simply adding is okay
            joint_features = self.flatten_2d_list(zip(coords, features))
            return torch.cat(joint_features, dim=-1)
        elif self.options.feature_fusion_method == 'stats':
            features_stats = self.get_features_stats(features)
            return torch.cat(coords + [features_stats], dim=-1)
        elif self.options.feature_fusion_method == 'attention':
            features_attn = self.attention_fusion(features)
            return torch.cat(coords + [features_attn], dim=-1)
        else:
            print('unknown method %r' % self.options.feature_fusion_method)
            exit(0)

    def attention_fusion(self, features):
        num_views = len(features)
        batch_size, num_points, num_features = features[0].size()

        # the batch number of points should be flattened
        # Each point acts independently regarless of the batch it belongs to
        # TODO: verify if this is the right approach
        flattened_features = torch.stack(features, dim=0) \
                                  .view(num_views, -1, num_features)
        features_attn, weights_attn = self.attention_model(flattened_features)
        features_attn = features_attn.squeeze(0) \
                                     .view(batch_size, num_points, num_features)
        weights_attn = weights_attn.squeeze(1) \
                                   .view(batch_size, num_points, num_views)
        return features_attn

    #  @param features list of features of size
    #           batch x num_points x num_channels for each view
    @staticmethod
    def get_features_stats(features):
        joint_features = torch.stack(features, dim=-1)
        max_features = torch.max(joint_features, dim=-1)[0]
        mean_features = torch.mean(joint_features, dim=-1)
        var_features = torch.var(joint_features, dim=-1, unbiased=False)
        # calculating std using torch methods give NaN gradients
        # var will have different unit that mean/max, hence std desired
        std_features = torch.sqrt(var_features + 1e-8)
        return torch.cat((max_features, mean_features, std_features), dim=-1)

    def get_rgb_features(self, img):
        if self.options.use_rgb_features:
            batch_size = img.size(0)
            vgg_input = self.flatten_batch_view(img)
            vgg_feats = self.rgb_vgg(vgg_input)

            img_feats = [
                self.unflatten_batch_view(i, batch_size)
                for i in vgg_feats
            ]
            return img_feats
        else:
            return []

    ##
    #  @param pred_depth  batch x view x h x w
    #  @param rendered_depth batch x view x h x w
    def get_depth_features(self, pred_depth, rendered_depth):
        batch_size = pred_depth.size(0)
        # unsqeeze 1 to simulate channels (single channel)
        pred_depth = P2MModel.flatten_batch_view(pred_depth).unsqueeze(1)
        rendered_depth = self.flatten_batch_view(rendered_depth).unsqueeze(1)
        joint_input = torch.cat((pred_depth, rendered_depth), dim=1)
        joint_features = self.depth_vgg(joint_input)
        return [
            self.unflatten_batch_view(i, batch_size) for i in joint_features
        ]

    ##
    #  @param depth_features 2D list of size num_views x num_features
    #  @return 1D list of size num_features
    #           each item batch x view x channel x w x h
    @staticmethod
    def group_costvolume_features(costvolume_features):
        # list of size num_features x num_views
        costvolume_features = [
            [costvolume_features[i][j] for i in range(len(costvolume_features))]
            for j in range(len(costvolume_features[0]))
        ]
        costvolume_features = [torch.stack(i, dim=1) for i in costvolume_features]
        # reverse to make similar to image features (lower channel first)
        return costvolume_features

    def forward(self, input_batch):
        img = input_batch["images"]
        proj_mat = input_batch["proj_matrices"]
        depth_values = input_batch["depth_values"]
        masks = input_batch["masks"]
        batch_size = img.size(0)
        img_shape = self.projection_2d.image_feature_shape(img[0])

        rendered_depths_before_deform = [None for _ in range(3)]
        rendered_depths = [None for _ in range(3)]
        batched_faces = [
            self.ellipsoid.faces[i].unsqueeze(0).expand(batch_size, -1, -1)
            for i in range(3)
        ]
        unflatten_batch_view = functools.partial(
            self.unflatten_batch_view, batch_size=batch_size
        )

        out_mvsnet = self.mvsnet(input_batch)
        costvolume_features = out_mvsnet["features"]
        costvolume_features = \
                self.group_costvolume_features(costvolume_features)
        masked_pred_depth = out_mvsnet["depths"] * masks
        masked_pred_depth = F.interpolate(
            masked_pred_depth, [config.IMG_SIZE, config.IMG_SIZE], mode='nearest'
        )

        rgb_features = self.get_rgb_features(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        rendered_depths_before_deform[0] = self.mesh_to_depth(
            init_pts, batched_faces[0], proj_mat, img_shape
        )
        depth_features = self.get_depth_features(
            masked_pred_depth, rendered_depths_before_deform[0]
        )

        # GCN Block 1
        # x1 shape is torch.Size([16, 156, 3]), x_h
        x = self.cross_view_feature_pooling(
            img_shape, rgb_features, depth_features, costvolume_features,
            masked_pred_depth, init_pts, proj_mat, depth_values
        )
        x1, x_hidden = self.gcns[0](x)

        if self.gconv_skip_connection == 'concat':
            x1 = self.gconv1(torch.cat((x1, init_pts), -1))
        elif self.gconv_skip_connection == 'add':
            x1 = x1 + init_pts

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        rendered_depths_before_deform[1] = self.mesh_to_depth(
            x1, batched_faces[0], proj_mat, img_shape
        )
        depth_features = self.get_depth_features(
            masked_pred_depth, rendered_depths_before_deform[1]
        )

        # GCN Block 2
        x = self.cross_view_feature_pooling(
            img_shape, rgb_features, depth_features, costvolume_features,
            masked_pred_depth, x1, proj_mat, depth_values
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

        rendered_depths_before_deform[2] = self.mesh_to_depth(
            x2, batched_faces[1], proj_mat, img_shape
        )
        depth_features = self.get_depth_features(
            masked_pred_depth, rendered_depths_before_deform[2]
        )

        # GCN Block 3
        # x2 shape is torch.Size([16, 618, 3])
        x = self.cross_view_feature_pooling(
            img_shape, rgb_features, depth_features, costvolume_features,
            masked_pred_depth, x2, proj_mat, depth_values
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

        pred_coord = [x1, x2, x3]
        pred_coord_before_deform = [init_pts, x1_up, x2_up]

        # render depth
        rendered_depths = [
            self.mesh_to_depth(pred_coord[i], batched_faces[i],
                               proj_mat, img_shape)
            for i in range(len(pred_coord))
        ]

        return {
            "pred_coord": pred_coord,
            "pred_coord_before_deform": pred_coord_before_deform,
            "reconst": None,
            "depths": out_mvsnet["depths"],
            "rendered_depths": rendered_depths,
            "rendered_depths_before_deform": rendered_depths_before_deform
        }

    def mesh_to_depth(self, coords, faces, proj_mat, image_shape):
        """
        takes care of creating proper sized tensors for efficient
        batched rendering
        @param pred_coords: (batch x vertices x 3) tensor
        @param faces: (batch x faces x 3) tensor
        @param image_shape shape of the depth image to be rendered
        @param proj_mat: (batch x view x 2 x 4 x 4) tensor
        @return depth tensor batch x view x height x width
        """
        batch_size = proj_mat.size(0)
        num_views = proj_mat.size(1)
        # augment views: size = (batch x view x vertices x 3)
        coords_augmented = coords.unsqueeze(1).expand(-1, num_views, -1, -1) \
                                    .contiguous()
        # size = (batch x view x faces` x 3)
        faces_augmented = faces.unsqueeze(1).expand(-1, num_views, -1, -1) \
                                    .contiguous()

        T_ref_world = proj_mat[:, 0, 0]
        T_world_ref = torch.inverse(T_ref_world)
        transformations = []
        for view_idx in range(num_views):
            T_view_world = proj_mat[:, view_idx, 0]
            T_view_ref = torch.bmm(T_view_world, T_world_ref)
            transformations.append(T_view_ref)
        transformations_augmented = torch.stack(transformations, dim=1)
        depth_flattened = self.render_depth(
            self.flatten_batch_view(coords_augmented),
            self.flatten_batch_view(faces_augmented),
            self.flatten_batch_view(transformations_augmented),
            image_shape
        )
        return self.unflatten_batch_view(depth_flattened, batch_size)

    def transform_to_renderer_frame(self, T_view_world):
        """
        @param T_view_world: (batch x 4 x 4) transformation
                in shapenet coordinates (East-Up-South)
        @return (batch x 4 x 4) transformation in renderer frame (East-Down-North)
        """
        batch_size = T_view_world.size(0)
        device = T_view_world.device

        self.T_renderer_shapenet = self.T_renderer_shapenet.to(device)
        self.T_shapenet_renderer = self.T_shapenet_renderer.to(device)

        # change to correct shape (batched)
        T_renderer_shapenet = self.T_renderer_shapenet \
                                  .unsqueeze(0) .expand(batch_size, -1, -1)
        T_shapenet_renderer = self.T_shapenet_renderer \
                                  .unsqueeze(0).expand(batch_size, -1, -1)

        # inverse y and z, equivalent to inverse x, but gives positive z
        T_view_world = torch.bmm(T_renderer_shapenet, T_view_world)
        return T_view_world

    def render_depth(self, coords, faces, T_view_world, image_shape):
        """
        renders a batch of depths
        @param pred_coords: (batch x vertices x 3) tensor
        @param faces: (batch x faces x 3) tensor
        @param image_shape shape of the depth image to be rendered
        @param T_view_world: (batch x 4 x 4) transformation
                in shapenet coordinates (EUS)
        """
        image_size = image_shape.max()
        # This is not thread safe!
        self.renderer.image_size = image_size
        batch_size, num_points = coords.size()[:2]

        # move to correct device
        device = coords.device
        self.camera_k = self.camera_k.to(device)
        self.dist_coeffs = self.dist_coeffs.to(device)
        faces = faces.type(torch.int32).to(device)

        # change to correct shape (batches)
        dist_coeffs = self.dist_coeffs.unsqueeze(0).expand(batch_size, -1)

        # transformation stuffs
        T_view_world = self.transform_to_renderer_frame(T_view_world)
        R = T_view_world[:, :3, :3]
        t = T_view_world[:, :3, 3].unsqueeze(1)
        depth = self.renderer(
            vertices=coords, faces=faces, mode='depth',
            K=self.camera_k.unsqueeze(0), dist_coeffs=dist_coeffs,
            R=R, t=t, orig_size=image_size
        )
        depth[depth <= self.renderer.near] = 0
        depth[depth >= self.renderer.far] = 0
        return depth

