# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os
import subprocess
import pickle
import numpy as np
import open3d as o3d
import sklearn.preprocessing
import trimesh
import multiprocessing as mp
import time
import contextlib

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')

parser.add_argument('shapenet_dir', type=str,
                    help='directory of the original shapenetv1 dataset')
parser.add_argument('rendering_dir', type=str,
                    help='Directory of the rendering meta data.')
parser.add_argument('xms_exec', type=str,
                    help='xms executable for rendering depth')
parser.add_argument('--objects-categories-file', type=str, default='',
                    help='xms executable for rendering depth')

args = parser.parse_args()

P2M_SCALE_FACTOR = 0.57
P2M_FOCAL_LENGTH = 250
P2M_IMG_SIZE = (224, 224)
P2M_PRINCIPAL_POINT = (112, 112)
P2M_MIN_POINT_DEPTH = 0.3
P2M_MAX_POINT_DEPTH = 1.0

def normal(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return np.divide(v, norm)


def camera_mat(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
    return cam_mat, cam_pos


def original_obj_transform(obj_path, view_path):
    mesh_list = list(trimesh.load_mesh(obj_path, 'obj').geometry.values())
    # if not isinstance(mesh_list, list):
    #     mesh_list = [mesh_list]

    area_sum = 0
    for mesh in mesh_list:
        area_sum += np.sum(mesh.area_faces)

    sample = np.zeros((0, 3), dtype=np.float32)
    normal = np.zeros((0, 3), dtype=np.float32)
    for mesh in mesh_list:
        number = int(round(16384 * np.sum(mesh.area_faces) / area_sum))
        if number < 1:
            continue
        # points, index = trimesh.sample.sample_surface_even(mesh, number)
        points, index = trimesh.sample.sample_surface(mesh, number)
        sample = np.append(sample, points, axis=0)

        triangles = mesh.triangles[index]
        pt1 = triangles[:, 0, :]
        pt2 = triangles[:, 1, :]
        pt3 = triangles[:, 2, :]
        norm = np.cross(pt3 - pt1, pt2 - pt1)
        norm = sklearn.preprocessing.normalize(norm, axis=1)
        normal = np.append(normal, norm, axis=0)

    # 2 tranform to camera view
    position = sample * 0.57

    cam_params = np.loadtxt(view_path)
    for index, param in enumerate(cam_params):
        # camera tranform
        cam_mat, cam_pos = camera_mat(param)

        pt_trans = np.dot(position - cam_pos, cam_mat.transpose())
        nom_trans = np.dot(normal, cam_mat.transpose())
        train_data = np.hstack((pt_trans, nom_trans))

        p2m_pcd = o3d.geometry.PointCloud()
        p2m_pcd.points = o3d.utility.Vector3dVector(pt_trans)
        o3d.io.write_point_cloud("/tmp/shapenet_to_p2m_original.ply", p2m_pcd)
        break

def projection_mat(R, t):
    T = np.eye(4, 4)
    T[:3, :3] = R
    T[:3, 3] = -np.matmul(R, t)
    # the negative ones are for compatibility with xms renderer
    P = np.asarray([
        [+1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0]
    ])
    return np.matmul(P, T)

def render_object(obj_category):
    obj, category = obj_category
    original_mesh_file = os.path.join(
        args.shapenet_dir, obj, category, 'model.obj'
    )
    scaled_mesh_file = '/tmp/model_scaled_{}_{}.obj'.format(obj, category)
    normal_mesh_file = '/tmp/model_normal_{}_{}.obj'.format(obj, category)
    rendering_metadata_file = '{}/{}/{}/rendering/rendering_metadata.txt' \
                                .format(args.rendering_dir, obj, category)
    depth_dir = '{}/{}/{}/rendering_depth' \
                    .format(args.rendering_dir, obj, category)
    os.makedirs(depth_dir, exist_ok=True)

    # if len(os.listdir(depth_dir)):
    #     return

    # avoid already (recently) rendered files
    depth_file = os.path.join(depth_dir, '00.png')
    if os.path.isfile(depth_file): # and time.time() - os.path.getmtime(depth_file) < 27500:
        return

    print('Working with ', obj, category)
    print('mesh file ', original_mesh_file)

    with contextlib.suppress(FileNotFoundError):
        os.remove(normal_mesh_file)
        os.remove(scaled_mesh_file)

    try:
        shapenet_model = o3d.io.read_triangle_mesh(original_mesh_file)
        # scale it to fit P2M
        shapenet_model.scale(P2M_SCALE_FACTOR)
        o3d.io.write_triangle_mesh(scaled_mesh_file, shapenet_model)

        # shapenet_model = trimesh.load_mesh(original_mesh_file)
        # shapenet_model.apply_scale(P2M_SCALE_FACTOR)
        # shapenet_model.export(scaled_mesh_file)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print('[Error] Unable to load mesh', original_mesh_file, str(e))
        return

    # find mesh with normal
    subprocess.run([
        'meshlabserver', '-i', scaled_mesh_file, '-o', normal_mesh_file,
        # options to select properties to save
        '-m', 'vn'
    ]) # , stdout=subprocess.DEVNULL)

    if not os.path.isfile(normal_mesh_file):
        print('[Error] Unable to generate mesh with normal', normal_mesh_file)

    rendering_metadata = np.loadtxt(rendering_metadata_file)

    for view_id, extrinsics in enumerate(rendering_metadata):
        R, t = camera_mat(extrinsics)
        proj_mat = projection_mat(R, t)
        proj_mat_str = np.char.mod('%f', proj_mat.flatten()).tolist()

        depth_file_prefix = os.path.join(
            depth_dir, '{0:02}'.format(view_id)
        )
        print('writing', depth_file_prefix)
        subprocess.run([
            'python', args.xms_exec,
            '-texture', 'simpledepthmap',
            normal_mesh_file, depth_file_prefix,
            '--intrinsics',
            str(P2M_FOCAL_LENGTH),
            str(P2M_PRINCIPAL_POINT[0]), str(P2M_PRINCIPAL_POINT[1]),
            '--extrinsics', *proj_mat_str,
            '--image_size', str(P2M_IMG_SIZE[0]), str(P2M_IMG_SIZE[1]),
            '--min_point_depth', str(P2M_MIN_POINT_DEPTH),
            '--max_point_depth', str(P2M_MAX_POINT_DEPTH),
        ], cwd=os.path.dirname(args.xms_exec)) # , stdout=subprocess.DEVNULL)

        if os.path.isfile(depth_file_prefix + '.png'):
            print('written', depth_file_prefix)
        else:
            print('[Error] unable to render', depth_file_prefix)

        break

    os.remove(normal_mesh_file)
    os.remove(scaled_mesh_file)

# multi-processing stuffs
NCORE = 8

if __name__ == '__main__':
    if args.objects_categories_file:
        shapenet_objects_categories = np.loadtxt(
            args.objects_categories_file, dtype=str
        )
    else:
        shapenet_objects = (
            i for i in os.listdir(args.shapenet_dir)
            if os.path.isdir(os.path.join(args.shapenet_dir, i))
        )
        shapenet_objects_categories = (
            (obj, category)
            for obj in shapenet_objects
            for category in os.listdir(os.path.join(args.shapenet_dir, obj))
            if os.path.isdir(os.path.join(args.shapenet_dir, obj, category))
        )

    with mp.Pool(NCORE) as p:
        p.map(render_object, shapenet_objects_categories)

    # for i in shapenet_objects_categories:
    #     render_object(i)
