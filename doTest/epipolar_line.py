'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import torch

import models.backbones.costvolume as costvolume
import config
from datasets.data_io import read_pfm

img1_idx = 16
img2_idx = 3
DATASET_DIR = '/home/zhiwen/projects/Pixel2Mesh/datasets/data/shapenet/data_tf/'

# Load the left and right images in gray scale
imgLeft = cv2.imread(
    os.path.join(
        DATASET_DIR,
        'Rectified_resized/scan2_train/rect_{:0>3}_0_r5000.png'.format(img1_idx)
    ), 0
)
imgRight = cv2.imread(
    os.path.join(
        DATASET_DIR,
        'Rectified_resized/scan2_train/rect_{:0>3}_0_r5000.png'.format(img2_idx)
    ), 0
)

proj_mat1_filename = os.path.join(
    DATASET_DIR, 'Cameras/train_resized/{:0>8}_cam.txt'.format(img1_idx)
)
proj_mat2_filename = os.path.join(
    DATASET_DIR, 'Cameras/train_resized/{:0>8}_cam.txt'.format(img2_idx)
)


def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1]) * 4 * 1.06
    return intrinsics, extrinsics, depth_min, depth_interval


def get_projection_matrix(proj_mat_filename):
    intrinsics, extrinsics, depth_min, depth_interval = read_cam_file(proj_mat_filename)
    proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
    proj_mat[0, :4, :4] = extrinsics
    proj_mat[1, :3, :3] = intrinsics
    proj_mat = torch.tensor(proj_mat, dtype=torch.float32)
    proj_mat = proj_mat.unsqueeze(0)

    # proj_mat_new = proj_mat[:, 0].clone()
    # proj_mat_new[:, :3, :4] = torch.matmul(proj_mat[:, 1, :3, :3], proj_mat[:, 0, :3, :4])

    return proj_mat, depth_min, depth_interval


proj_mat1, _, _ = get_projection_matrix(proj_mat1_filename)
proj_mat2, _, _ = get_projection_matrix(proj_mat2_filename)

K1 = proj_mat1[0, 1, :3, :3].cpu().numpy()
K2 = proj_mat2[0, 1, :3, :3].cpu().numpy()
K1[:2] *= 4
K2[:2] *= 4

# Detect the SIFT key points and compute the descriptors for the two images
sift = cv2.xfeatures2d.SIFT_create()
keyPointsLeft, descriptorsLeft = sift.detectAndCompute(imgLeft, None)
keyPointsRight, descriptorsRight = sift.detectAndCompute(imgRight, None)

# Create FLANN matcher object
FLANN_INDEX_KDTREE = 1
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=100)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# Match the descriptors (this crashes in OpenCV3.1)
# See https://github.com/Itseez/opencv/issues/5667
matches = flann.knnMatch(descriptorsLeft, descriptorsRight, k=2)

# Apply ratio test
goodMatches = []
ptsLeft = []
ptsRight = []

for m, n in matches:
    if m.distance < 0.8 * n.distance:
        goodMatches.append([m])
        ptsLeft.append(keyPointsLeft[m.queryIdx].pt)
        ptsRight.append(keyPointsRight[m.trainIdx].pt)

ptsLeft = np.int32(ptsLeft)
ptsRight = np.int32(ptsRight)


E, mask = cv2.findEssentialMat(ptsLeft.astype(np.float32), ptsRight.astype(np.float32), K1, cv2.RANSAC)
# F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_RANSAC)
F = np.linalg.multi_dot(
    (np.linalg.inv(K2.transpose()), E, np.linalg.inv(K1))
)
# We select only inlier points
ptsLeft = ptsLeft[mask.ravel() == 1]
ptsRight = ptsRight[mask.ravel() == 1]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for pt_idx, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        # color = tuple(np.random.randint(0, 255, 3).tolist())
        color = cv2.applyColorMap(
            np.array([int(float(pt_idx) / len(pts1) * 255)]).astype(np.uint8),
            cv2.COLORMAP_COOL
        )[0][0].tolist()
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def make_points_homogeneous(pts):
    return  np.concatenate(
        (pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)),
        axis=1
    )


def compute_transform(F, K1, K2, pts1, pts2):
    E = np.linalg.multi_dot((K1.transpose(), F, K2))
    pts1 = make_points_homogeneous(pts1)
    pts2 = make_points_homogeneous(pts2)
    pts1 = np.matmul(pts1, np.linalg.inv(K1).transpose())
    pts2 = np.matmul(pts2, np.linalg.inv(K2).transpose())
    points, R, t, mask = cv2.recoverPose(E, pts1[:, :2], pts2[:, :2], K1)
    # for i in range(pts1.shape[0]):
    #     print('epipolar constraint\n', np.linalg.multi_dot((pts2[i,:], E, pts1[i,:].transpose())))
    print(cv2.decomposeEssentialMat(E))
    print('K1:\n', K1)
    print('K2:\n', K2)
    print('R:\n', R.transpose())
    print('t:\n', -np.dot(R.transpose(), t))


def cv_epiploar_lines(imgLeft, imgRight, ptsLeft, ptsRight, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    linesLeft = cv2.computeCorrespondEpilines(ptsRight, 2, F)
    linesLeft = linesLeft.reshape(-1, 3)
    img5, img6 = drawlines(imgLeft, imgRight, linesLeft, ptsLeft, ptsRight)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2), 1, F)
    linesRight = linesRight.reshape(-1, 3)
    img3, img4 = drawlines(imgRight, imgLeft, linesRight, ptsRight, ptsLeft)


    plt.subplot(223), plt.imshow(img5)
    plt.subplot(224), plt.imshow(img3)
    # plt.show()


def read_depth(filename):
    # read pfm depth file
    depth = np.array(read_pfm(filename)[0], dtype=np.float32)
    h, w = depth.shape
    depth = cv2.resize(depth, (56, 56), interpolation=cv2.INTER_NEAREST)
    return depth


def cost_volume_epipolar_lines(img1_idx, img2_idx,
                               img1_orig, img2_orig, pts1_orig, pts2_orig):
    global DATASET_DIR
    ndepths = 500

    proj_mat1_filename = os.path.join(
        DATASET_DIR, 'Cameras/train_resized/{:0>8}_cam.txt'.format(img1_idx)
    )
    proj_mat2_filename = os.path.join(
        DATASET_DIR, 'Cameras/train_resized/{:0>8}_cam.txt'.format(img2_idx)
    )

    proj_mat1, _, _ = get_projection_matrix(proj_mat1_filename)
    proj_mat2, depth_min, depth_interval = get_projection_matrix(proj_mat2_filename)
    depth_min = 0.1

    # resize to size actually used by cost volume implementation
    img_size = (config.IMG_SIZE // 4, config.IMG_SIZE // 4)
    img1 = cv2.resize(img1_orig, img_size, cv2.INTER_AREA)
    img2 = cv2.resize(img2_orig, img_size, cv2.INTER_AREA)

    # Note: matrix dim 0 and 1 correspond to axes y and x (not x and y)
    pts1 = np.asarray([
        [i[0] * img_size[1] / img1_orig.shape[1], i[1] * img_size[0] / img1_orig.shape[0]]
        for i in pts1_orig
    ], dtype=np.float32)
    pts2 = np.asarray([
        (i[0] * img_size[1] / img2_orig.shape[1], i[1] * img_size[0] / img2_orig.shape[0])
        for i in pts2_orig
    ], dtype=np.float32)

    depth_filename1 = os.path.join(DATASET_DIR, 'Depths/scan2_train/depth_map_{:0>4}.pfm'.format(img1_idx))
    depth_filename2 = os.path.join(DATASET_DIR, 'Depths/scan2_train/depth_map_{:0>4}.pfm'.format(img2_idx))
    depth_img1 = read_depth(depth_filename1)
    depth_img2 = read_depth(depth_filename2)

    depth_values = np.arange(depth_min, depth_interval * ndepths + depth_min,
                             depth_interval, dtype=np.float32)
    depth_values = torch.tensor(depth_values).unsqueeze(0)

    # pts1 = pts1[5:]
    # pts2 = pts2[5:]
    x1 = torch.tensor(pts1[:, 0])
    y1 = torch.tensor(pts1[:, 1])
    x2 = torch.tensor(pts2[:, 0])
    y2 = torch.tensor(pts2[:, 1])

    d1 = depth_img1[int(pts1[0, 1]), int(pts1[0, 0])] * config.DTU_RESCALE_FACTOR
    d2 = depth_img2[int(pts2[0, 1]), int(pts2[0, 0])] * config.DTU_RESCALE_FACTOR
    print(pts1[0], pts2[0])
    print(d1, d2)

    proj_xy = costvolume.project_pixel_coords(x2, y2, depth_values, proj_mat1, proj_mat2, 1)
    img3, img4 = draw_points(img1, img2, proj_xy, pts1, pts2)

    proj_xy = costvolume.project_pixel_coords(x1, y1, depth_values, proj_mat2, proj_mat1, 1)
    img5, img6 = draw_points(img2, img1, proj_xy, pts2, pts1)

    # plt.figure()
    plt.subplot(221), plt.imshow(img3)
    plt.subplot(222), plt.imshow(img5)
    # plt.show()
    return {'proj_mat1': proj_mat1, 'proj_mat2':proj_mat2}

def draw_points(img1, img2, proj_xy, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for pt_idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        # color = tuple(np.random.randint(0, 255, 3).tolist())
        color = cv2.applyColorMap(
            np.array([int(float(pt_idx) / len(pts1) * 255)]).astype(np.uint8),
            cv2.COLORMAP_COOL
        )[0][0].tolist()
        img1 = cv2.circle(img1, tuple(map(int, pt1)), 2, color, -1)
        img2 = cv2.circle(img2, tuple(map(int, pt2)), 2, color, -1)

        for hypot_idx in range(proj_xy.size(2)):
            hypot_pt = proj_xy[0, :, hypot_idx, pt_idx].tolist()
            print(hypot_pt)
            # img1 = cv2.circle(img1, tuple(map(int, hypot_pt)), 1, color, 1)
            if 0 < hypot_pt[1] < img1.shape[0] and 0 < hypot_pt[0] < img1.shape[1]:
                img1[int(hypot_pt[1]), int(hypot_pt[0])] = color
    return img1, img2

plt.figure()
cv_epiploar_lines(imgLeft, imgRight, ptsLeft, ptsRight, F)

output = cost_volume_epipolar_lines(img1_idx, img2_idx,
                                    imgLeft, imgRight, ptsLeft, ptsRight)

compute_transform(F, K1, K2, ptsLeft, ptsRight)
plt.show()