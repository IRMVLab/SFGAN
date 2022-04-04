import numpy as np
# import pandas as pd
import os
import cv2
from collections import Counter
import pickle
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
gfile = tf.gfile


CMAP = 'plasma'
          
def main():                          #generate the standard format of data
    cam = 2
    depth_threshold = 30
    for sequence in range(11):            #range of sequence has been changed
        sequence = str(sequence).zfill(2)
        velo_dir = os.path.join(BASE_DIR,'data_odometry_velodyne/sequences',sequence)
        velo_files = gfile.Glob(os.path.join(velo_dir,'velodyne','*.bin'))
        velo_files= sorted(velo_files)

        calib_name = os.path.join(BASE_DIR,'data_odometry_calib/dataset/sequences',sequence,'calib.txt')
        calib_file = read_calib_file(calib_name)
        P_rect = calib_file['P2'].reshape(3, 4)

        image_dir = os.path.join(BASE_DIR,'data_odometry_color/dataset/sequences',sequence)
        images = gfile.Glob(os.path.join(image_dir,'image_2','*.png'))
        images = sorted(images)

        num_frames = len(velo_files)
        
        dirname = os.path.join(BASE_DIR,'data/depth_img_new_noresize/dataset/sequences',sequence,'image2')    #notice the directory
        if not os.path.exists(dirname):
          os.makedirs(dirname)

        poses_file = os.path.join(BASE_DIR,'data_odometry_poses/dataset/poses',sequence + '.txt')
        poses = np.loadtxt(poses_file)
        poses = poses.reshape(-1,12)

        pc2_img = cv2.imread(images[0])
        pc2_img = cv2.cvtColor(pc2_img, cv2.COLOR_BGR2RGB)  # rgb
        pc2_shape = pc2_img.shape[:2]
        depth2 = generate_depth_map(calib_file, P_rect, velo_files[0], pc2_shape, cam, False, True)
        pc2 = generate_point_cloud(depth2,P_rect)
        mask2 = (depth2 > 0) & (depth2 < depth_threshold)
        #depth2 = resize_depth_map(depth2)
        Ti = poses[0].reshape(3, 4)
        Ti = np.vstack((Ti, np.array([0, 0, 0, 1.0])))

        for index in range(1,num_frames):
            fname = os.path.join(dirname,str(index-1).zfill(6)+'.npz')
            #pc1_name = os.path.join(dirname,str(index).zfill(6)+'pc1.ply')
            #pc2_name = os.path.join(dirname,str(index).zfill(6)+'pc2.ply')
            #pc2_cam1_name = os.path.join(dirname,str(index).zfill(6)+'pc2_cam1.ply')
            #depth_path = os.path.join(dirname, str(index).zfill(6) + '.png')

            pc1_img = pc2_img
            pc1_shape = pc2_shape
            mask1 = mask2
            pc1 = pc2
            depth1 = depth2
            Ti_1 = np.mat(Ti)

            pc2_img = cv2.imread(images[index])
            pc2_img = cv2.cvtColor(pc2_img, cv2.COLOR_BGR2RGB) #rgb
            pc2_shape = pc2_img.shape[:2]
            #image =np.transpose(image, (2,0,1))
            depth2 = generate_depth_map(calib_file, P_rect, velo_files[index], pc2_shape, cam, False, True)
            pc2 = generate_point_cloud(depth2, P_rect)
            h,w = depth2.shape
            mask2 = (depth2 > 0) & (depth2 < depth_threshold)
            #depth2 = resize_depth_map(depth2)
            Ti = poses[index].reshape(3,4)
            Ti = np.vstack((Ti, np.array([0, 0, 0, 1.0])))
            T2_1 = np.dot(Ti_1.I,Ti)
            #print('transpose of ',index,'\n',T2_1)

            pc1 = pc1.reshape(h*w,3)
            pc2 = pc2.reshape(h*w,3)
            color1 = pc1_img.reshape(h*w,3)
            color2 = pc2_img.reshape(h*w,3)
            mask1 = mask1.astype(np.int16)
            mask1 = mask1.reshape(h*w,)
            mask2 = mask2.astype(np.int16)
            mask2 = mask2.reshape(h*w,)
            #print('the valid point of pc1 is ',np.sum(mask1))      #output the number of the valid point
            index1 = np.array(np.where(mask1 == 1))[0]
            index2 = np.array(np.where(mask2 == 1))[0]
            pc1_n = pc1[index1,:]
            color1 = color1[index1,:]
            pc2_n = pc2[index2,:]
            color2 = color2[index2,:]
            
            is_ground1 = (pc1_n[:, 1] < -1.2)
            not_ground1 = np.logical_not(is_ground1)
            is_ground2 = (pc2_n[:, 1] < -1.2)
            not_ground2 = np.logical_not(is_ground2)

            pc1_n = pc1_n[not_ground1, :]
            color1 = color1[not_ground1, :]
            pc2_n = pc2_n[not_ground2, :]
            color2 = color2[not_ground2, :]

            num1 = pc1_n.shape[0]
            num2 = pc2_n.shape[0]

            img1 = pc1_img
            img2 = pc2_img

            print('img1.shape:', img1.shape, 'img2.shape:', img2.shape, 'depth1.shape:', depth1.shape, 'depth2.shape', depth2.shape)
            print('pc1:', pc1_n.shape, 'pc2:', pc2_n.shape, 'color1:', color1.shape, 'color2:', color2.shape)
            if(num1 >= 8192) & (num2 >= 8192):
                print(fname)
                np.savez_compressed(fname, points1=pc1_n, \
                                    points2=pc2_n, \
                                    color1=color1, \
                                    color2=color2, \
                                    shape1=pc1_shape, \
                                    shape2=pc2_shape, \
                                    P_rect=P_rect, \
                                    img1=img1, \
                                    img2=img2, \
                                    depth1=depth1, \
                                    depth2=depth2, \
                                    flow=T2_1)
                # generate_ply_pc1(fname,pc1_name)      #
                # generate_ply_pc2(fname,pc2_name)
                # generate_ply_pc2_cam1(fname,pc2_cam1_name)




def load_velodyne_points(file_name):                #no need to change
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def load_depth_map(fname):
    with open(fname, 'rb') as fp:
        data = np.load(fp)
        depth1 = data['depth1']
        depth2 = data['depth2']

    return depth1,depth2

def cam2_cam1(pc2_cam2,pose):
    n,_ = pc2_cam2.shape
    ones = np.ones(shape=(n,1),dtype=np.float32)
    pc2_cam2 = np.concatenate([pc2_cam2,ones],axis = 1)
    pc2_cam2 = np.transpose(pc2_cam2,[1,0])
    pc2_cam1 = np.dot(pose,pc2_cam2)
    pc2_cam1 = np.transpose(pc2_cam1,[1,0])
    pc2_cam1 = pc2_cam1[:,:3]
    return pc2_cam1
    
def resize_depth_map(gt_depth):
    max_depth = 35
    min_depth = 1e-3
    gt_height, gt_width = gt_depth.shape
    for q in range(0, 2):
        for i in range(1, gt_height - 1):
            for j in range(1, gt_width - 1):
                if gt_depth[i, j] == 0:
                    if gt_depth[i, j - 1] != 0 and gt_depth[i, j + 1] == 0 and gt_depth[i - 1, j] == 0 and \
                                    gt_depth[i + 1, j] == 0:
                        gt_depth[i, j] = gt_depth[i, j - 1]
                    if gt_depth[i, j - 1] == 0 and gt_depth[i, j + 1] != 0 and gt_depth[i - 1, j] == 0 and \
                                    gt_depth[i + 1, j] == 0:
                        gt_depth[i, j] = gt_depth[i, j + 1]
                    if gt_depth[i, j - 1] == 0 and gt_depth[i, j + 1] == 0 and gt_depth[i - 1, j] != 0 and \
                                    gt_depth[i + 1, j] == 0:
                        gt_depth[i, j] = gt_depth[i - 1, j]
                    if gt_depth[i, j - 1] == 0 and gt_depth[i, j + 1] == 0 and gt_depth[i - 1, j] == 0 and \
                                    gt_depth[i + 1, j] != 0:
                        gt_depth[i, j] = gt_depth[i + 1, j]

                    if gt_depth[i, j - 1] != 0 and gt_depth[i, j + 1] != 0 and gt_depth[i - 1, j] == 0 and \
                                    gt_depth[i + 1, j] == 0:
                        gt_depth[i, j] = (gt_depth[i, j - 1] + gt_depth[i, j + 1]) / 2.0
                    if gt_depth[i, j - 1] != 0 and gt_depth[i, j + 1] == 0 and gt_depth[i - 1, j] != 0 and \
                                    gt_depth[i + 1, j] == 0:
                        gt_depth[i, j] = (gt_depth[i, j - 1] + gt_depth[i - 1, j]) / 2.0
                    if gt_depth[i, j - 1] != 0 and gt_depth[i, j + 1] == 0 and gt_depth[i - 1, j] == 0 and \
                                    gt_depth[i + 1, j] != 0:
                        gt_depth[i, j] = (gt_depth[i, j - 1] + gt_depth[i + 1, j]) / 2.0
                    if gt_depth[i, j - 1] == 0 and gt_depth[i, j + 1] != 0 and gt_depth[i - 1, j] != 0 and \
                                    gt_depth[i + 1, j] == 0:
                        gt_depth[i, j] = (gt_depth[i, j + 1] + gt_depth[i - 1, j]) / 2.0
                    if gt_depth[i, j - 1] == 0 and gt_depth[i, j + 1] != 0 and gt_depth[i - 1, j] == 0 and \
                                    gt_depth[i + 1, j] != 0:
                        gt_depth[i, j] = (gt_depth[i, j + 1] + gt_depth[i + 1, j]) / 2.0
                    if gt_depth[i, j - 1] == 0 and gt_depth[i, j + 1] == 0 and gt_depth[i - 1, j] != 0 and \
                                    gt_depth[i + 1, j] != 0:
                        gt_depth[i, j] = (gt_depth[i - 1, j] + gt_depth[i + 1, j]) / 2.0

                    if gt_depth[i, j - 1] != 0 and gt_depth[i, j + 1] != 0 and gt_depth[i - 1, j] != 0 and \
                                    gt_depth[i + 1, j] == 0:
                        gt_depth[i, j] = (gt_depth[i, j - 1] + gt_depth[i, j + 1] + gt_depth[i - 1, j]) / 3.0
                    if gt_depth[i, j - 1] != 0 and gt_depth[i, j + 1] != 0 and gt_depth[i - 1, j] == 0 and \
                                    gt_depth[i + 1, j] != 0:
                        gt_depth[i, j] = (gt_depth[i, j - 1] + gt_depth[i, j + 1] + gt_depth[i + 1, j]) / 3.0
                    if gt_depth[i, j - 1] != 0 and gt_depth[i, j + 1] == 0 and gt_depth[i - 1, j] != 0 and \
                                    gt_depth[i + 1, j] != 0:
                        gt_depth[i, j] = (gt_depth[i, j - 1] + gt_depth[i - 1, j] + gt_depth[i + 1, j]) / 3.0
                    if gt_depth[i, j - 1] == 0 and gt_depth[i, j + 1] != 0 and gt_depth[i - 1, j] != 0 and \
                                    gt_depth[i + 1, j] != 0:
                        gt_depth[i, j] = (gt_depth[i, j + 1] + gt_depth[i - 1, j] + gt_depth[i + 1, j]) / 3.0

                    if gt_depth[i, j - 1] != 0 and gt_depth[i, j + 1] != 0 and gt_depth[i - 1, j] != 0 and \
                                    gt_depth[i + 1, j] != 0:
                        gt_depth[i, j] = (gt_depth[i, j - 1] + gt_depth[i, j + 1] + gt_depth[i - 1, j] +
                                          gt_depth[i + 1, j]) / 4.0

    for i in range(1, gt_height - 1):
        for j in range(1, gt_width - 1):
            if gt_depth[i, j] == 0:
                gt_depth[i, j] = max_depth

    gt_depth[gt_depth > max_depth] = max_depth
    gt_depth[gt_depth < min_depth] = min_depth

    gt_depth = (cv2.resize(gt_depth, (416, 128), interpolation=cv2.INTER_LINEAR))

    '''gt_depth = gt_depth / 10.
    gt_depth = (cv2.resize(gt_depth, (416, 128), interpolation=cv2.INTER_LINEAR))
    colored_map = _normalize_depth_for_display(gt_depth, pc=95, crop_percent=1, cmap=CMAP)
    scipy.misc.imsave(depth_path, colored_map)'''

    return gt_depth


def read_calib_file(path):                          #changed
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


def _gray2rgb(im, cmap=CMAP):
  cmap = plt.get_cmap(cmap)
  rgba_img = cmap(im.astype(np.float32))
  rgb_img = np.delete(rgba_img, 3, 2)
  return rgb_img


def _normalize_depth_for_display(depth,
                                 pc=95,
                                 crop_percent=1,
                                 normalizer=None,
                                 cmap=CMAP):
  """Converts a depth map to an RGB image."""
  # Convert to disparity.
  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = _gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[keep_h:]
  return disp


def generate_depth_map(calib_file, P_rect, velo_file_name,im_shape, cam=2, interp=False, vel_depth=False):
    # load calibration files
    velo2cam =  calib_file['Tr']  
    velo2cam = velo2cam.reshape(3,4)
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))  

    # compute projection matrix velodyne->image plane
    P_velo2im = np.dot(P_rect, velo2cam)  #(3,4)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)  #(n,4)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T  #(n,3)
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis] # normalization xy/z

    velo_pts_im[:, 2] = velo[:, 0]  # depth

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if
                 count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth        #(h,w)


def generate_point_cloud(depth, P_rect,px=None, py=None):
    focal_x = P_rect[0,0]
    focal_y = P_rect[1,1]
    const_x = P_rect[0,2] * depth + P_rect[0,3]
    const_y = P_rect[1,2] * depth + P_rect[1,3]
    #focal_length_pixel = P_rect[0,0]
    height,width = depth.shape
    if px is None:
        px = np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1))
    if py is None:
        py = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    
    x = ((px * (depth + P_rect[2,3]) - const_x) / focal_x)
    y = ((py * (depth + P_rect[2,3]) - const_y) / focal_y)

    pc = np.stack((x, y, depth), axis=-1)
    pc[..., :2] *= -1.
    return pc


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def generate_ply_pc1(input_file,output_file):
    with open(input_file, 'rb') as fp:
        data = np.load(fp)
        pc = data['points1']  # shape nx3
        n = pc.shape[0]
        color = np.tile([220,20,60],[n,1])
        #color = data['color1']
    create_output(pc, color, output_file)
    
def generate_ply_pc2(input_file,output_file):
    with open(input_file, 'rb') as fp:
        data = np.load(fp)
        pc = data['points2']  # shape nx3
        n = pc.shape[0]
        color = np.tile([34,139,34],[n,1])
        #color = data['color2']
    create_output(pc, color, output_file)
    
    
def generate_ply_pc2_cam1(input_file,output_file):
    with open(input_file, 'rb') as fp:
        data = np.load(fp)
        pc = data['pc2_cam1']  # shape nx3
        n = pc.shape[0]
        color = np.tile([255,255,255],[n,1])
        #color = data['color2']
    create_output(pc, color, output_file)

if __name__ == '__main__':
    main()


