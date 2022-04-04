import numpy as np
# import pandas as pd
import os
import cv2
from collections import Counter
import pickle
import argparse
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
gfile = tf.gfile

CMAP = 'plasma'

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../../../jsy/odom_wxr/', type=str, help='input root dir')
parser.add_argument('--output_dir', default='../../data', type=str, help='output dir')
parser.add_argument('--depth_threshold', type=float, default=30, help='depth threshold for pointcloud')
parser.add_argument('--rm_ground', type=float, default=-1.2, help='the depth of the removing ground')

FLAGS = parser.parse_args()

INPUT_DIR = FLAGS.input_dir
OUTPUT_DIR = FLAGS.output_dir
DEPTH_THRESHOLD = FLAGS.depth_threshold
RM_GROUND = FLAGS.rm_ground

def main():  # generate the standard format of data
    for sequence in range(11):  # range of sequence has been changed
        sequence = str(sequence).zfill(2)
        dirname = os.path.join(OUTPUT_DIR, 'velo2cam_no_duplicate_depth/dataset/sequences', sequence)  # notice the directory
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        velo_dir = os.path.join(INPUT_DIR, 'data_odometry_velodyne/sequences', sequence)

        velo_files = gfile.Glob(os.path.join(velo_dir, 'velodyne', '*.bin'))
        velo_files = sorted(velo_files)

        calib_name = os.path.join(INPUT_DIR, 'data_odometry_calib/dataset/sequences', sequence, 'calib.txt')
        calib_file = read_calib_file(calib_name)

        num_frames = len(velo_files)

        pc2 = generate_point_cloud(calib_file, velo_files[0])

        for index in range(1, num_frames):
            fname = os.path.join(dirname, str(index - 1).zfill(6) + '.npz')

            pc1 = pc2
            pc2 = generate_point_cloud(calib_file, velo_files[index])

            within_depth1 = np.logical_and((pc1[:, 2] < DEPTH_THRESHOLD), (pc1[:, 2] > -DEPTH_THRESHOLD))
            within_depth2 = np.logical_and((pc2[:, 2] < DEPTH_THRESHOLD), (pc2[:, 2] > -DEPTH_THRESHOLD))
            within_lr1 = np.logical_and((pc1[:, 0] < DEPTH_THRESHOLD), (pc1[:, 0] > -DEPTH_THRESHOLD))
            within_lr2 = np.logical_and((pc2[:, 0] < DEPTH_THRESHOLD), (pc2[:, 0] > -DEPTH_THRESHOLD))
            within_surround1 = np.logical_and(within_depth1, within_lr1)
            within_surround2 = np.logical_and(within_depth2, within_lr2)
            is_ground1 = (pc1[:, 1] < RM_GROUND)
            not_ground1 = np.logical_not(is_ground1)
            is_ground2 = (pc2[:, 1] < RM_GROUND)
            not_ground2 = np.logical_not(is_ground2)
            sampled_indices1 = np.logical_and(within_surround1, not_ground1)
            sampled_indices2 = np.logical_and(within_surround2, not_ground2)

            pc1 = pc1[sampled_indices1]
            pc2 = pc2[sampled_indices2]

            num1 = pc1.shape[0]
            num2 = pc2.shape[0]

            color1 = np.tile([0,0,0],[num1,1])
            color2 = np.tile([0,0,0],[num2,1])

            print('pc1:', pc1.shape, 'pc2:', pc2.shape,'color1:', color1.shape, 'color2:', color2.shape)
            if (num1 >= 8192) & (num2 >= 8192):
                print(fname)
                np.savez_compressed(fname, points1=pc1, \
                                    points2=pc2, \
                                    color1=color1, \
                                    color2=color2)


def load_velodyne_points(file_name):  # no need to change
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):  # changed
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1

def generate_point_cloud(calib_file, velo_file_name):
    # load calibration files

    h = 376
    w = 1241
    velo2cam = calib_file['Tr']
    velo2cam = velo2cam.reshape(3, 4)
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    cam2img = calib_file['P2'].reshape(3, 4)


    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)  # (n,4)

    # project the points to the camera
    cam = np.dot(velo2cam, velo.T)  # (4,n)
    img = np.dot(cam2img, cam).T #(n,3)
    img[:, :2] = img[:, :2] / img[:, 2][..., np.newaxis]
    img[:, 0] = np.round(img[:, 0]) - 1
    img[:, 1] = np.round(img[:, 1]) - 1
    img[:,2] = velo[:,0]
    pc = cam.T[:, :3]
    mask1 = ((img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h) & (pc[:, 2] > 0))  # cam1投影到正图像平面内
    img = img[mask1,:]
    pc = pc[mask1,:]

    mask2 = np.ones((pc.shape[0],),dtype=np.bool)
    # find the duplicate points and choose the closest depth
    inds = sub2ind((h,w), img[:, 1].astype(np.int), img[:, 0].astype(np.int))
    dupe_inds = [item for item, count in Counter(inds).items() if
                 count > 1]

    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        min_depth = img[pts, 2].min()
        for i in pts:
            if img[i,2] > min_depth:
                mask2[i] = 0

    pc = pc[mask2,:]

    return pc




if __name__ == '__main__':
    main()


