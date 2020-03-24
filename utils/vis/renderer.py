import cv2
import neural_renderer as nr
# import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F

def _process_render_result(img, height, width):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 2:
        # assuming single channel image
        img = np.expand_dims(img, axis=0)
    if img.shape[-1] == 3:
        # assuming [height, width, rgb]
        img = np.moveaxis(img, -1, 0)
    # return 3 * width * height or width * height, in range [0, 1]
    return np.clip(img[:height, :width], 0, 1)


def _mix_render_result_with_image(rgb, alpha, image):
    alpha = np.expand_dims(alpha, 0)
    return alpha * rgb + (1 - alpha) * image


class MeshRenderer(object):

    def __init__(self, camera_f, camera_c, mesh_pos):
        self.colors = {'pink': np.array([.9, .7, .7]),
                       'light_blue': np.array([0.65098039, 0.74117647, 0.85882353]),
                       'light_green': np.array([165., 216., 168.]) / 255,
                       'purple': np.array([216., 193., 165.]) / 255,
                       'orange': np.array([216., 165., 213.]) / 255,
                       'light_yellow': np.array([213., 216., 165.]) / 255,
                       }
        self.camera_f, self.camera_c, self.mesh_pos = camera_f, camera_c, mesh_pos
        self.renderer = nr.Renderer(camera_mode='projection',
                                    light_intensity_directional=.8,
                                    light_intensity_ambient=.3,
                                    background_color=[1., 1., 1.],
                                    light_direction=[0., 0., -1.])

    def _render_mesh(self, vertices: np.ndarray, faces: np.ndarray, width, height,
                     camera_k, camera_dist_coeffs, rvec, tvec, color=None):
        # render a square image, then crop
        img_size = max(height, width)

        # This is not thread safe!
        self.renderer.image_size = img_size

        vertices = torch.tensor(vertices, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.int32)

        if color is None:
            color = 'light_blue'
        color = self.colors[color]
        texture_size = 2
        textures = torch.tensor(color, dtype=torch.float32) \
            .repeat(faces.size(0), texture_size, texture_size, texture_size, 1)

        camera_k = torch.tensor(camera_k, dtype=torch.float32)
        rotmat = torch.tensor(cv2.Rodrigues(rvec)[0], dtype=torch.float32)
        tvec = torch.tensor(tvec, dtype=torch.float32)
        camera_dist_coeffs = torch.tensor(camera_dist_coeffs, dtype=torch.float32)

        rgb, _, alpha = self.renderer.render(vertices.unsqueeze(0).cuda(),
                                             faces.unsqueeze(0).cuda(),
                                             textures.unsqueeze(0).cuda(),
                                             K=camera_k.unsqueeze(0).cuda(),
                                             R=rotmat.unsqueeze(0).cuda(),
                                             t=tvec.unsqueeze(0).cuda(),
                                             dist_coeffs=camera_dist_coeffs.unsqueeze(0).cuda(),
                                             orig_size=img_size)
        # use the extra dimension of alpha for broadcasting
        alpha = _process_render_result(alpha[0], height, width)
        rgb = _process_render_result(rgb[0], height, width)

        return rgb, alpha

    def _render_pointcloud(self, vertices: np.ndarray, width, height,
                           camera_k, camera_dist_coeffs, rvec, tvec, colors=None):
        if colors is None:
            colors = 'pink'
            colors = np.tile(
                np.expand_dims(self.colors['pink'], axis=0),
                (vertices.shape[0], 1)
            )

        vertices = vertices.astype(np.float64)
        rvec = rvec.astype(np.float64)
        tvec = tvec.astype(np.float64)
        camera_k = camera_k.astype(np.float64)
        camera_dist_coeffs = camera_dist_coeffs.astype(np.float64)

        # return pointcloud
        vertices_2d = cv2.projectPoints(np.expand_dims(vertices, -1),
                                        rvec, tvec, camera_k, camera_dist_coeffs)[0]
        vertices_2d = np.reshape(vertices_2d, (-1, 2))
        alpha = np.zeros((height, width), np.float)
        rgb = np.zeros((height, width, 3), np.float)
        whiteboard = np.ones((3, height, width), np.float)
        if np.isnan(vertices_2d).any():
            return whiteboard, alpha
        for vertex_idx, (x, y) in enumerate(vertices_2d):
            try:
                cv2.circle(
                    alpha, (int(x), int(y)), radius=1,
                    color=1., thickness=-1
                )
                cv2.circle(
                    rgb, (int(x), int(y)), radius=1,
                    color=colors[vertex_idx], thickness=-1
                )
            except OverflowError as e:
                print("can't render (%r, %r):" % (x, y))
                # raise(e)
        rgb = _process_render_result(rgb, height, width)
        alpha = _process_render_result(alpha, height, width)
        rgb = _mix_render_result_with_image(rgb, alpha[0], whiteboard)
        return rgb, alpha

    def _1to3channel(self, img):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        img = np.repeat(np.expand_dims(img, 0), 3, 0)
        return img

    @staticmethod
    def transform_points(T, pts):
        return MeshRenderer.make_points_unhomogeneous(
            np.dot(MeshRenderer.make_points_homogeneous(pts), T.transpose()
        ))

    @staticmethod
    def make_points_homogeneous(pts):
        return np.concatenate(
            (pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)),
            axis=-1
        )

    @staticmethod
    def make_points_unhomogeneous(pts):
        return pts[:, :3]

    ##
    #  @param extrinsics numpy array of size views x 4 x 4
    def visualize_reconstruction(self, gt_coord, coord, faces, images,
                                 pred_depths, gt_depths, rendered_depth, masks,
                                 view_weights, extrinsics,
                                 mesh_only=False, **kwargs):
        camera_k = np.array([[self.camera_f[0], 0, self.camera_c[0]],
                             [0, self.camera_f[1], self.camera_c[1]],
                             [0, 0, 1]])
        # inverse y and z, equivalent to inverse x, but gives positive z
        rvec = np.array([np.pi, 0., 0.], dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        mesh, _ = self._render_mesh(coord, faces, images[0].shape[2], images[0].shape[1],
                                    camera_k, dist_coeffs, rvec, tvec, **kwargs)
        if mesh_only:
            return mesh

        T_ref_world = extrinsics[0]
        T_world_ref = np.linalg.inv(T_ref_world)
        num_views = len(pred_depths)
        gt_pcs = [None for _ in range(num_views)]
        pred_pcs = [None for _ in range(num_views)]
        for view_idx in range(num_views):
            T_view_world = extrinsics[view_idx]
            T_view_ref = np.dot(T_view_world, T_world_ref)
            gt_coord_view = self.transform_points(T_view_ref, gt_coord)
            pred_coord_view = self.transform_points(T_view_ref, coord)
            dummy_view_weights = np.zeros(
                (gt_coord.shape[0], num_views), dtype=np.float64
            )
            dummy_view_weights[:, view_idx] = 1.0

            gt_pcs[view_idx], _ = self._render_pointcloud(
                gt_coord_view, images[0].shape[2], images[0].shape[1],
                camera_k, dist_coeffs, rvec, tvec, dummy_view_weights,
                **kwargs
            )
            pred_pcs[view_idx], _ = self._render_pointcloud(
                pred_coord_view, images[0].shape[2], images[0].shape[1],
                camera_k, dist_coeffs, rvec, tvec, view_weights,
                **kwargs
            )
        # get all views from the depths
        pred_depths = [self._1to3channel(i) for i in pred_depths]
        gt_depths = [self._1to3channel(i) for i in gt_depths]
        masks = [self._1to3channel(i) for i in masks]
        rendered_depths = [
            self._1to3channel(rendered_depth[view_idx])
            for view_idx in range(rendered_depth.shape[0])
        ]

        # return np.concatenate((images[0], gt_pc, pred_pc), 2)
        return np.concatenate((
            *images, *gt_depths, *pred_depths, *rendered_depths,
            *gt_pcs, *pred_pcs, mesh
        ), 2)

    def p2m_batch_visualize(self, batch_input, batch_output, faces, atmost=3):
        """
        Every thing is tensor for now, needs to move to cpu and convert to numpy
        """
        batch_size = min(batch_input["images_orig"].size(0), atmost)
        images_stack = []
        mesh_pos = np.array(self.mesh_pos)
        for i in range(batch_size):
            images = batch_input["images_orig"][i].cpu().numpy()
            gt_depths = batch_input["depths"][i].cpu().numpy()
            pred_depths = batch_output["depths"][i].cpu().numpy()
            masks = batch_input["masks"][i].cpu().numpy()
            pred_depths *= masks
            depth_max = np.max(gt_depths)
            gt_depths = gt_depths / depth_max
            pred_depths = pred_depths / depth_max
            gt_points = batch_input["points"][i].cpu().numpy() + mesh_pos

            pred_depths = list(pred_depths)
            gt_depths = list(gt_depths)
            masks = list(masks)

            # write_point_cloud(gt_points, '/tmp/{}_gt.ply'.format(i))
            for j in range(3):
                for k in (["pred_coord_before_deform", "pred_coord"] if j == 0 else ["pred_coord"]):
                    coord = batch_output[k][j][i].cpu().numpy() + mesh_pos
                    depth_key = k.replace("pred_coord", "rendered_depths")
                    rendered_depth = batch_output[depth_key][j][i].detach() \
                                        .cpu().numpy()
                    if batch_output["view_weights"][j] is not None:
                        view_weights = F.normalize(
                            batch_output["view_weights"][j][i].squeeze(-1),
                            dim=-1
                        ).cpu().numpy().astype(np.float64)
                    else:
                        view_weights = None
                    extrinsics = batch_input["proj_matrices"][i, :, 0] \
                                        .cpu().numpy()
                    images_stack.append(self.visualize_reconstruction(
                        gt_points, coord, faces[j].cpu().numpy(), images,
                        pred_depths, gt_depths, rendered_depth,
                        masks, view_weights, extrinsics
                    ))
                    # write_point_cloud(coord, '/tmp/{}_{}_{}.ply'.format(i, j, k))

        return torch.from_numpy(np.concatenate(images_stack, 1))

def write_point_cloud(coord, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    o3d.io.write_point_cloud(filename, pcd)
