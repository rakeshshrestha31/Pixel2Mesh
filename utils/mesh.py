import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

import trimesh
from scipy.sparse import coo_matrix
import itertools

import config


def torch_sparse_tensor(indices, value, size):
    coo = coo_matrix((value, (indices[:, 0], indices[:, 1])), shape=size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, shape)


class Ellipsoid(object):

    def __init__(self, mesh_pos, file=config.ELLIPSOID_PATH):
        with open(file, "rb") as fp:
            fp_info = pickle.load(fp, encoding='latin1')

        # shape: n_pts * 3
        self.coord = torch.tensor(fp_info[0]) - torch.tensor(mesh_pos, dtype=torch.float)

        # edges & faces & lap_idx
        # edge: num_edges * 2
        # faces: num_faces * 4
        # laplace_idx: num_pts * 10
        self.edges, self.laplace_idx = [], []

        for i in range(3):
            self.edges.append(torch.tensor(fp_info[1 + i][1][0], dtype=torch.long))
            self.laplace_idx.append(torch.tensor(fp_info[7][i], dtype=torch.long))

        # unpool index
        # num_pool_edges * 2
        # pool_01: 462 * 2, pool_12: 1848 * 2
        self.unpool_idx = [torch.tensor(fp_info[4][i], dtype=torch.long) for i in range(2)]

        # loops and adjacent edges
        self.adj_mat = []
        for i in range(1, 4):
            # 0: np.array, 2D, pos
            # 1: np.array, 1D, vals
            # 2: tuple - shape, n * n
            adj_mat = torch_sparse_tensor(*fp_info[i][1])
            self.adj_mat.append(adj_mat)

        ellipsoid_dir = os.path.dirname(file)
        self.faces = []
        self.obj_fmt_faces = []
        # faces: f * 3, original ellipsoid, and two after deformations
        for i in range(1, 4):
            face_file = os.path.join(ellipsoid_dir, "face%d.obj" % i)
            faces = np.loadtxt(face_file, dtype='|S32')
            self.obj_fmt_faces.append(faces)
            self.faces.append(torch.tensor(faces[:, 1:].astype(np.int) - 1))

        self.update_adjacent_faces()

    ## make uniform tensor by padding zeroes to a 2D list
    # https://stackoverflow.com/questions/40569220/efficiently-convert-uneven-list-of-lists-to-minimal-containing-array-padded-with
    @staticmethod
    def get_padded_tensor(list_obj, dtype=torch.float32, pad_value=0):
        return torch.tensor(
            list(itertools.zip_longest(*list_obj, fillvalue=0)),
            dtype=dtype
        ).transpose(0, 1).contiguous()

    ## find faces adjacent to each vertex
    #   Used for calculating vertex normals
    def update_adjacent_faces(self):
        self.adj_faces = []
        # since the number of faces a vertex can be adjacent to can vary
        self.adj_faces_mask = []
        for i in range(3):
            num_vertices = self.adj_mat[i].size(0)
            adj_faces = [[] for _ in range(num_vertices)]

            for face_idx, vertex_indices in enumerate(self.faces[i].unbind(0)):
                for vertex_idx in vertex_indices:
                    adj_faces[vertex_idx].append(face_idx)

            adj_faces_mask = [ [1.0 for _ in range(len(i))] for i in adj_faces ]

            self.adj_faces.append(
                self.get_padded_tensor(adj_faces, dtype=torch.long)
            )
            self.adj_faces_mask.append(self.get_padded_tensor(adj_faces_mask))

    ## get vertex normals
    #  @param coords coordinates of the vertices. batch x num_points x 3 tensor
    def get_vertex_normals(self, coords, resolution_idx):
        assert(coords.size(1) == self.adj_faces[resolution_idx].size(0))
        batch_size = coords.size(0)
        faces = self.faces[resolution_idx]
        adj_faces = self.adj_faces[resolution_idx]
        adj_faces_mask = self.adj_faces_mask[resolution_idx]
        num_faces = faces.size(0)
        num_vertices = coords.size(1)

        # vertices of all faces: batch_size x num_faces x 3 x 3
        # the first 3 is for 3 vertices in a face , the last 3 for 3D coords
        face_vertices = coords[:, faces.view(-1)] \
                            .view(batch_size, num_faces, 3, 3)
        # cross products of all faces: batch_size x num_faces x 3
        face_cross_products = torch.cross(
            face_vertices[:, :, 0] - face_vertices[:, :, 1],
            face_vertices[:, :, 0] - face_vertices[:, :, 2],
            dim=-1
        )
        # cross product of vertices for each adjacent face
        # batch_size x num_vertices x max_adjacent_faces x 3
        vertex_cross_products = face_cross_products[:, adj_faces.view(-1)] \
                                    .view(batch_size, num_vertices, -1, 3)
        face_mask = adj_faces_mask.unsqueeze(0).unsqueeze(-1) \
                                  .expand(batch_size, -1, -1, 3) \
                                  .to(coords.device)
        masked_vertex_cross_products = vertex_cross_products * face_mask
        # summing up adjacent faces' cross products is
        # equivalent to weighted sum of normal where weight = area of face
        # https://www.iquilezles.org/www/articles/normals/normals.htm
        vertex_normals = masked_vertex_cross_products.sum(dim=2)
        vertex_normals = F.normalize(vertex_normals, dim=-1)

        return vertex_normals

##
#  @param vertices num_points x 3
#  @param faces num_faces x 3 long tensor
#  @param vertex_normals num_points x 3
def visualize_vertex_normals(vertices, faces, vertex_normals, filename):
    # augment with normal points
    augmented_vertices = torch.cat((
        vertices, vertices + 0.001,
        vertices + vertex_normals * 0.01
    ), dim=0).numpy()
    # fake faces corresponding to normals
    normal_faces = torch.stack((
        torch.arange(vertices.size(0)),
        torch.arange(vertices.size(0)) + vertices.size(0),
        torch.arange(vertex_normals.size(0)) + (2 * vertices.size(0))
    ), dim=-1)
    augmented_faces = torch.cat(
        (faces, normal_faces), dim=0
    ).numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(augmented_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(augmented_faces)
    o3d.io.write_triangle_mesh(filename, mesh)

def test_normals():
    ellipsoid = Ellipsoid([0, 0, 0])
    vertex_normals = ellipsoid.get_vertex_normals(ellipsoid.coord.unsqueeze(0), 0)
    print('vertex normals:', vertex_normals.size())
    visualize_vertex_normals(
        ellipsoid.coord, ellipsoid.faces[0], vertex_normals[0],
        '/tmp/ellipsoid_normal.ply'
    )

def test_upsampled_normals():
    ellipsoid = Ellipsoid([0, 0, 0])
    coords = ellipsoid.coord.unsqueeze(0)
    faces = ellipsoid.faces[0].unsqueeze(0)
    normals = ellipsoid.get_vertex_normals(coords, 0)

    from models.layers.sample_points import PointSampler
    points_sampler = PointSampler(4000)
    upsampled_coords, upsampled_normals = points_sampler(coords, faces)
    upsampled_coords = torch.cat((coords, upsampled_coords), dim=1)
    upsampled_normals = torch.cat((normals, upsampled_normals), dim=1)
    print('upsampled vertex normals:', upsampled_normals.size())
    visualize_vertex_normals(
        upsampled_coords[0], faces[0], upsampled_normals[0],
        '/tmp/ellipsoid_normal_upsampled.ply'
    )

if __name__ == "__main__":
    test_normals()
    test_upsampled_normals()
