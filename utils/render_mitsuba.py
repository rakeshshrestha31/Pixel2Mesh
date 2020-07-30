import argparse, sys, os
import subprocess
import pickle
import cv2
import numpy as np
import open3d as o3d
import sklearn.preprocessing
import trimesh
import multiprocessing as mp
import time
import contextlib
import functools

def parse_args():
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')

    parser.add_argument('shapenet_dir', type=str,
                        help='directory of the original shapenetv1 dataset')
    parser.add_argument('rendering_dir', type=str,
                        help='Directory of the rendering meta data.')
    parser.add_argument('--objects-categories-file', type=str, default='',
                        help='file with models to render (either train_list*.txt or test_list*.txt')

    args = parser.parse_args()
    return args

P2M_SCALE_FACTOR = 0.57
P2M_FOCAL_LENGTH = 250
P2M_FOV = 25
P2M_IMG_SIZE = (224, 224)
P2M_PRINCIPAL_POINT = (112, 112)
P2M_MIN_POINT_DEPTH = 0.1
P2M_MAX_POINT_DEPTH = 1.3
RENDERING_RESIZE_FACTOR = 1 # 5
RENDERING_DEPTH_SCALE = 1000

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

def transformation_mat(R, t):
    T = np.eye(4, 4)
    T[:3, :3] = R
    T[:3, 3] = -np.matmul(R, t)
    return T

def remove_mesh_file(mesh_filename):
    os.remove(mesh_filename)
    with contextlib.suppress(FileNotFoundError):
        os.remove(mesh_filename.replace('.obj', '.mtl'))
        os.remove(mesh_filename.replace('.obj', '.png'))

def write_rendering_params(rendering_params_files, rendering_metadata,
                           obj_file):
    assert(RENDERING_RESIZE_FACTOR == 1)
    mitsuba_template_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'mitsuba_template.xml'
    )
    with open(mitsuba_template_file, 'r') as f:
        mitsuba_template = f.read()
    rendering_size = [i * RENDERING_RESIZE_FACTOR for i in P2M_IMG_SIZE]
    for view_idx, extrinsics in enumerate(rendering_metadata):
        extrinsics_mat = transformation_mat(*(camera_mat(extrinsics)))
        extrinsics_list = extrinsics_mat.flatten().tolist()
        extrinsics_list = [str(i) for i in extrinsics_list]
        mitsuba_xml = mitsuba_template.format(
            obj_file=obj_file,
            img_width=rendering_size[0],
            img_height=rendering_size[1],
            fov=P2M_FOV,
            transformation_world_cam=" ".join(extrinsics_list),
            origin_x=extrinsics_mat[0, 3],
            origin_y=extrinsics_mat[1, 3],
            origin_z=extrinsics_mat[2, 3],
        )
        with open(rendering_params_files[view_idx], 'w') as f:
            f.write(mitsuba_xml)

def render_object(obj_category, args, return_depths):
    obj, category = obj_category
    original_mesh_file = os.path.join(
        args.shapenet_dir, obj, category, 'model.obj'
    )
    scaled_mesh_file = '/tmp/model_scaled_{}_{}.obj'.format(obj, category)
    normal_mesh_file = '/tmp/model_normal_{}_{}.obj'.format(obj, category)
    rendering_metadata_file = '{}/{}/{}/rendering/rendering_metadata.txt' \
                                .format(args.rendering_dir, obj, category)

    rendering_metadata = np.loadtxt(rendering_metadata_file)

    rendering_params_files = [
        '/tmp/rendering_params_{}_{}_{}.xml'.format(obj, category, view_idx)
        for view_idx in range(len(rendering_metadata))
    ]

    depth_dir = '{}/{}/{}/rendering_depth' \
                    .format(args.rendering_dir, obj, category)
    os.makedirs(depth_dir, exist_ok=True)

    # if len(os.listdir(depth_dir)):
    #     return

    # avoid already (recently) rendered files
    depth_files = [
        os.path.join(depth_dir, '{0:02}.png'.format(i))
        for i in range(len(rendering_metadata))
    ]
    # max_creation_time = 24 * 3600.
    # depth_files_available = [
    #     os.path.isfile(depth_file) \
    #             and (time.time() - os.path.getmtime(depth_file)) \
    #                 < max_creation_time
    #     for depth_file in depth_files
    # ]
    # if np.all(depth_files_available):
    #     print('skipping', depth_dir)
    #     return

    print('Working with ', obj, category)
    print('mesh file ', original_mesh_file)

    with contextlib.suppress(FileNotFoundError):
        remove_mesh_file(normal_mesh_file)
        remove_mesh_file(scaled_mesh_file)

    write_rendering_params(rendering_params_files, rendering_metadata, scaled_mesh_file)
    print('writing', depth_dir)

    meshlabserver_command = [
        'meshlabserver', '-i', original_mesh_file, '-o', normal_mesh_file,
        # options to select properties to save
        '-om', 'vn'
    ]
    print('running :', ' '.join(meshlabserver_command))

    # find mesh with normal
    subprocess.run(meshlabserver_command) # , stdout=subprocess.DEVNULL)

    print('done meshlab')

    if not os.path.isfile(normal_mesh_file):
        print('[Error] Unable to generate mesh with normal', normal_mesh_file)
        return

    try:
        shapenet_model = o3d.io.read_triangle_mesh(normal_mesh_file)
        # scale it to fit P2M
        shapenet_model.scale(P2M_SCALE_FACTOR, center=False)
        # find mesh with normal (meshlab server may not always succeed)
        # but we need meshlab server nonetheless
        # some meshes are not readable otherwise
        shapenet_model.compute_vertex_normals()
        o3d.io.write_triangle_mesh(scaled_mesh_file, shapenet_model)

        # shapenet_model = trimesh.load_mesh(original_mesh_file)
        # shapenet_model.apply_scale(P2M_SCALE_FACTOR)
        # shapenet_model.export(scaled_mesh_file)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print('[Error] Unable to load mesh', normal_mesh_file, str(e))
        return

    for view_idx, param_file in enumerate(rendering_params_files):
        xms_command = [
            'mitsuba',
            param_file, '-o', os.path.join(depth_dir, '%d.exr' % view_idx)
        ]
        print('xms command:', ' '.join(xms_command))
        subprocess.run(xms_command) # , stdout=subprocess.DEVNULL)

    if return_depths:
        depths = []
        for view_idx in range(24):
            depth_file = os.path.join(depth_dir, '{:02}.png'.format(view_idx))
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            depth[depth == P2M_MAX_POINT_DEPTH * RENDERING_DEPTH_SCALE] = 0
            print(depth.dtype, np.min(depth), np.max(depth), np.mean(depth))
            depth = cv2.resize(depth, P2M_IMG_SIZE, interpolation=cv2.INTER_NEAREST)
            print(depth.dtype, np.min(depth), np.max(depth), np.mean(depth))
            cv2.imwrite(depth_file, depth)
            depths.append(depth)
    else:
        depths = None

    [os.remove(i) for i in rendering_params_files]
    remove_mesh_file(scaled_mesh_file)
    remove_mesh_file(normal_mesh_file)
    return depths

# multi-processing stuffs
NCORE = 8

if __name__ == '__main__':
    args = parse_args()
    if args.objects_categories_file:
        shapenet_objects_categories = np.loadtxt(
            args.objects_categories_file, dtype=str
        ).tolist()
        shapenet_objects_categories = (
            tuple(i.split('_')[:2]) for i in shapenet_objects_categories
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

    if NCORE >= 2:
        with mp.Pool(NCORE) as p:
            p.map(functools.partial(render_object, args=args, return_depths=False),
                  shapenet_objects_categories)
    else:
        for i in shapenet_objects_categories:
            print(i)
            render_object(i, args=args, return_depths=False)

    # for i in shapenet_objects_categories:
    #     render_object(i, args=args, return_depths=False)

