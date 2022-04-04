"""
    FlowNet3D model with up convolution
"""
from collections import Counter
import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
from pointnet_util import *
from tf_grouping import knn_point,group_point


def placeholder_inputs(batch_size, num_point):
    pc1 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pc2 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    color1 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    color2 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    img1 = tf.placeholder(tf.float32,shape=(batch_size,128,416,3))
    img2 = tf.placeholder(tf.float32,shape=(batch_size,128,416,3))
    depth1 = tf.placeholder(tf.float32,shape=(batch_size,128,416))
    depth2 = tf.placeholder(tf.float32,shape=(batch_size,128,416))
    P_rect=tf.placeholder(tf.float32,shape=(batch_size,3,4))
    dense_pc1 = tf.placeholder(tf.float32, shape=(batch_size, 20000, 3))
    dense_color1 = tf.placeholder(tf.float32, shape=(batch_size, 20000, 3))
    dense_pc2 = tf.placeholder(tf.float32, shape=(batch_size, 20000, 3))
    dense_color2 = tf.placeholder(tf.float32, shape=(batch_size, 20000, 3))
    pose = tf.placeholder(tf.float32, shape=(batch_size, 4, 4))
    return pc1, pc2, color1, color2,img1,img2,depth1,depth2,P_rect,dense_pc1, dense_color1, dense_pc2, dense_color2,pose


def placeholder_inputs_eval(batch_size, num_point):
    pc1 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pc2 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    color1 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    color2 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    flow = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pc1, pc2, color1, color2, flow


def get_model(pc1, pc2, color1, color2, is_training, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    # batch_size =pc1.get_shape()[0].value
    num_point = pc1.get_shape()[1].value

    l0_xyz_f1 = pc1
    l0_points_f1 = color1
    l0_xyz_f2 = pc2
    l0_points_f2 = color2

    RADIUS1 = 0.5
    RADIUS2 = 1.0
    RADIUS3 = 2.0
    RADIUS4 = 4.0
    with tf.variable_scope('sa1') as scope:
        # Frame 1, Layer 1
        l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1, npoint=num_point/2, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        end_points['l1_indices_f1'] = l1_indices_f1

        # Frame 1, Layer 2
        l2_xyz_f1, l2_points_f1, l2_indices_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=num_point/8, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        end_points['l2_indices_f1'] = l2_indices_f1

        scope.reuse_variables()
        # Frame 2, Layer 1
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=num_point/2, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        # Frame 2, Layer 2
        l2_xyz_f2, l2_points_f2, l2_indices_f2 = pointnet_sa_module(l1_xyz_f2, l1_points_f2, npoint=num_point/8, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    _, l2_points_f1_new = flow_embedding_module(l2_xyz_f1, l2_xyz_f2, l2_points_f1, l2_points_f2, radius=10.0, nsample=64, mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding', bn=True, pooling='max', knn=True, corr_func='concat')

    # Layer 3
    l3_xyz_f1, l3_points_f1, l3_indices_f1 = pointnet_sa_module(l2_xyz_f1, l2_points_f1_new, npoint=num_point/32, radius=RADIUS3, nsample=8, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    end_points['l3_indices_f1'] = l3_indices_f1

    # Layer 4
    l4_xyz_f1, l4_points_f1, l4_indices_f1 = pointnet_sa_module(l3_xyz_f1, l3_points_f1, npoint=num_point/128, radius=RADIUS4, nsample=8, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    end_points['l4_indices_f1'] = l4_indices_f1

    # Feature Propagation
    l3_feat_f1 = set_upconv_module(l3_xyz_f1, l4_xyz_f1, l3_points_f1, l4_points_f1, nsample=8, radius=2.4, mlp=[], mlp2=[256,256], scope='up_sa_layer1', is_training=is_training, bn_decay=bn_decay, knn=True)
    l2_feat_f1 = set_upconv_module(l2_xyz_f1, l3_xyz_f1, tf.concat(axis=-1, values=[l2_points_f1, l2_points_f1_new]), l3_feat_f1, nsample=8, radius=1.2, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer2', is_training=is_training, bn_decay=bn_decay, knn=True)
    l1_feat_f1 = set_upconv_module(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_feat_f1, nsample=8, radius=0.6, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer3', is_training=is_training, bn_decay=bn_decay, knn=True)
    l0_feat_f1 = pointnet_fp_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_feat_f1, [256,256], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_feat_f1, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


'''
    def knn_point(k, xyz1, xyz2):
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

'''
def smooth_loss(pc1, forward_sf,nsample=10):
    val,idx = knn_point(nsample, pc1, pc1)
    neighbor_sf = group_point(forward_sf,idx)
    forward_sf = tf.expand_dims(forward_sf,axis=2)
    neighbor_weight = tf.exp(-val)
    neighbor_weight = tf.expand_dims(neighbor_weight,-1)
    loss = tf.abs(forward_sf - neighbor_sf) * neighbor_weight/(tf.abs(forward_sf)+tf.abs(neighbor_sf))
    loss = tf.reduce_mean(loss)
    
    return loss
'''

def robust_loss_matrix(x, eps, q):
    loss = tf.pow(tf.abs(x) + eps, q)
    loss = tf.reduce_mean(loss, axis=-1)
    return loss

def norm2_loss(x):
    loss = tf.reduce_sum(x * x, axis=-1) / 2.0
    return loss

def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(losses,axis=-1)

def smooth_loss(pc1,forward_sf,color1,nsample,radius,eps,q,loss_type,delta):
    idx,pts_cnt = query_ball_point(radius,nsample,pc1,pc1)
    b,n = pts_cnt.shape
    flag = tf.tile(tf.reshape(tf.range(nsample), (1,nsample)), [b * n, 1]) < tf.tile(tf.reshape(pts_cnt, (-1, 1)), [1, nsample])
    mask_w = tf.reshape(tf.cast(flag,dtype=tf.float32),[b,n,nsample])
    mask = tf.cast((pts_cnt > 1),dtype=tf.float32)
    neighbor_pc = group_point(pc1,idx)
    pc1 = tf.expand_dims(pc1,axis=2)
    neighbor_sf = group_point(forward_sf, idx)
    forward_sf = tf.expand_dims(forward_sf, axis=2)
    neighbor_color = group_point(color1, idx)
    color1 = tf.expand_dims(color1,axis=2)
    l2_dist = tf.norm(pc1-neighbor_pc,ord=2,axis=-1)
    l2_color = tf.norm(color1-neighbor_color,ord=2,axis=-1)
    neighbor_weight = tf.exp(-(l2_dist+l2_color))
    neighbor_weight = neighbor_weight * mask_w
    if(loss_type=='robust'):
        loss = robust_loss_matrix((forward_sf - neighbor_sf),eps,q) * neighbor_weight
    elif (loss_type=='huber'):
        loss = huber_loss((forward_sf - neighbor_sf),delta) * neighbor_weight
    else:
        loss = norm2_loss(forward_sf - neighbor_sf) * neighbor_weight
    #loss = tf.abs(forward_sf - neighbor_sf) * neighbor_weight / (tf.abs(forward_sf) + tf.abs(neighbor_sf))
    loss = tf.reduce_sum(loss,axis=-1)
    pts_cnt1 = tf.to_float(pts_cnt)
    loss = loss / pts_cnt1
    mask = tf.stop_gradient(mask)
    loss = loss * mask
    #num = tf.reduce_sum(mask,axis=1)
    #loss = tf.reduce_sum(loss,axis=1)/ num
    #loss = tf.reduce_mean(loss)
    num = tf.reduce_sum(mask)
    loss = tf.reduce_sum(loss) / (num + 1e-6)

    return loss,num

def sf_loss(overall_sf, static_sf, dynamic_sf, eps, q,loss_type,delta):
    # pc1 = pixel2xyz(depth1,intrinsics)
    # pc2_cam2 = pixel2xyz(depth2,intrinsics)
    # depth:BxHxW   pose:Bx6  intrinsics:Bx3x3

    differ = overall_sf - static_sf - dynamic_sf
    if (loss_type == 'robust'):
        sf_loss = robust_loss_matrix(differ,eps,q)
    elif (loss_type=='huber'):
        sf_loss = huber_loss(differ,delta)
    else:
        sf_loss = norm2_loss(differ)
    sf_loss = tf.reduce_mean(sf_loss)
    tf.summary.scalar('sf loss', sf_loss)
    tf.add_to_collection('losses', sf_loss)

    return sf_loss


def reconstruction_loss(pc2_infered,img2,P_rect,color1, mask, eps, q,loss_type,delta,stop_z):

    if(stop_z):
        depth = pc2_infered[:,:,2]
        depth = tf.stop_gradient(depth)
        depth = tf.expand_dims(depth,-1)
        pc2_infered = tf.concat([pc2_infered[:,:,:2],depth],2)

    color2_infered = linear_interpolation(pc2_infered,img2,P_rect)
    if (loss_type == 'robust'):
        rec_loss = robust_loss_matrix(color2_infered - color1, eps, q)
    elif (loss_type=='huber'):
        rec_loss = huber_loss((color2_infered - color1),delta)
    else:
        rec_loss = norm2_loss(color2_infered - color1)
    mask_n = tf.to_float(pc2_infered[:,:,2]>0)
    mask = mask * mask_n
    mask = tf.stop_gradient(mask)
    rec_loss = mask * rec_loss
    num = tf.reduce_sum(mask)
    rec_loss = tf.reduce_sum(rec_loss) / (num+1e-6)
    # rec_loss = tf.reduce_mean(tf.reduce_sum((color2_infered - color1) * (color2_infered - color1), axis=2) / 2.0)
    tf.summary.scalar('rec loss', rec_loss)
    tf.add_to_collection('losses', rec_loss)

    return rec_loss,color2_infered

def reconstruction_loss_3d(pc2_infered,pc2,color2,color1, mask, eps, q,interp,radius,k,loss_type,delta,stop_z):
    if (stop_z):
        depth = pc2_infered[:, :, 2]
        depth = tf.stop_gradient(depth)
        depth = tf.expand_dims(depth, -1)
        pc2_infered = tf.concat([pc2_infered[:,:,:2],depth],2)

    if interp=='xyz':
        color2_infered, mask_k = xyz_interpolation(pc2_infered, pc2, color2, radius, k)
    elif interp=='dist':
        color2_infered, mask_k = dist_interpolation(pc2_infered, pc2, color2, radius, k)
    else:
        color2_infered, mask_k = reci_d_interpolation(pc2_infered, pc2, color2, radius, k)

    if (loss_type == 'robust'):
        rec_loss = robust_loss_matrix(color2_infered - color1, eps, q)
    elif(loss_type=='huber'):
        rec_loss = huber_loss((color2_infered - color1),delta)
    else:
        rec_loss = norm2_loss(color2_infered - color1)
    mask =  mask_k
    mask = tf.stop_gradient(mask)
    rec_loss = mask * rec_loss
    num = tf.reduce_sum(mask)
    rec_loss = tf.reduce_sum(rec_loss) / (num+1e-6)
    # rec_loss = tf.reduce_mean(tf.reduce_sum((color2_infered - color1) * (color2_infered - color1), axis=2) / 2.0)
    tf.summary.scalar('rec loss', rec_loss)
    tf.add_to_collection('losses', rec_loss)

    return rec_loss,num

def depth_rec_loss(color1,color2_infered,pc2_infered,depth2,P_rect,pc1, mask, eps, q,loss_type,delta,normalization,color_weight,stop_uv,stop_color):
    depth2 = tf.expand_dims(depth2,axis=-1)
    depth2_infered = linear_interpolation(pc2_infered,depth2,P_rect,stop_uv)
    if (normalization):
        differ_depth = tf.abs(depth2_infered - tf.expand_dims(pc2_infered[:, :, 2], axis=-1))/ tf.abs(depth2_infered + tf.expand_dims(pc2_infered[:, :, 2], axis=-1))
    else:
        differ_depth = tf.abs(depth2_infered - tf.expand_dims(pc2_infered[:, :, 2], axis=-1))

    if(color_weight):
        color2_infered = (color2_infered+1)/2
        color1 = (color1+1)/2
        weight = 1 - tf.abs(color1 - color2_infered)
        weight = tf.reduce_mean(weight,axis=2,keep_dims=True)
        if(stop_color):
            weight = tf.stop_gradient(weight)
        differ_depth = tf.multiply(weight,differ_depth)

    if (loss_type=='robust'):
        loss = robust_loss_matrix(differ_depth, eps, q)
    elif (loss_type=='huber'):
        loss = huber_loss(differ_depth, delta)
    else:
        loss = norm2_loss(differ_depth)
    mask_n = tf.to_float((pc2_infered[:, :, 2] > 0))
    mask = mask * mask_n
    mask = tf.stop_gradient(mask)
    loss = mask * loss
    num = tf.reduce_sum(mask,-1)
    loss = tf.reduce_sum(loss,-1) / (num+1e-6)
    loss = tf.reduce_mean(loss)
    # rec_loss = tf.reduce_mean(tf.reduce_sum((color2_infered - color1) * (color2_infered - color1), axis=2) / 2.0)
    tf.summary.scalar('rec loss', loss)
    tf.add_to_collection('losses', loss)

    return loss

def depth_rec_loss_3d(color1,color2_infered,pc2_infered,pc2,pc1, mask, eps, q,interp,P_rect,radius,k,loss_type,delta,normalization,color_weight,stop_uv,stop_color,xy_infered):
    pts2_img_infered = projection2img(pc2_infered,P_rect)
    if (stop_uv):
        pts2_img_infered = tf.stop_gradient(pts2_img_infered)
    pts2_img = projection2img(pc2,P_rect)
    depth2 = tf.expand_dims(pc2[:,:,2],axis=-1)

    if interp=='xyz':
        depth2_infered, mask_k = xyz_interpolation(pts2_img_infered, pts2_img, depth2, radius, k)
    elif interp=='dist':
        depth2_infered, mask_k = dist_interpolation(pts2_img_infered, pts2_img, depth2, radius, k)
    else:
        depth2_infered, mask_k = reci_d_interpolation(pts2_img_infered, pts2_img, depth2, radius, k)
    mask_n = (tf.to_float(pc2_infered[:, :, 2] > 0))
    mask = mask * mask_k * mask_n

    if (normalization):
        differ = tf.abs(depth2_infered - tf.expand_dims(pc2_infered[:, :, 2], axis=-1))/ tf.abs(depth2_infered + tf.expand_dims(pc2_infered[:, :, 2], axis=-1))
    else:
        differ = tf.abs(depth2_infered - tf.expand_dims(pc2_infered[:, :, 2], axis=-1))

    if(xy_infered):
        x2 = tf.expand_dims(pc2[:,:,0],axis=-1)
        y2 = tf.expand_dims(pc2[:,:,1],axis=-1)
        if interp == 'xyz':
            x2_infered, _ = xyz_interpolation(pts2_img_infered, pts2_img, x2, radius, k)
            y2_infered, _ = xyz_interpolation(pts2_img_infered, pts2_img, y2, radius, k)
        elif interp == 'dist':
            x2_infered, _ = dist_interpolation(pts2_img_infered, pts2_img, x2, radius, k)
            y2_infered, _ = dist_interpolation(pts2_img_infered, pts2_img, y2, radius, k)
        else:
            x2_infered, _ = reci_d_interpolation(pts2_img_infered, pts2_img, x2, radius, k)
            y2_infered, _ = reci_d_interpolation(pts2_img_infered, pts2_img, y2, radius, k)

        if (normalization):
            differ_x = tf.abs(x2_infered - tf.expand_dims(pc2_infered[:, :, 0], axis=-1)) / tf.abs(x2_infered + tf.expand_dims(pc2_infered[:, :, 0], axis=-1))
            differ_y = tf.abs(y2_infered - tf.expand_dims(pc2_infered[:, :, 1], axis=-1)) / tf.abs(y2_infered + tf.expand_dims(pc2_infered[:, :, 1], axis=-1))
        else:
            differ_x = tf.abs(x2_infered - tf.expand_dims(pc2_infered[:, :, 0], axis=-1))
            differ_y = tf.abs(y2_infered - tf.expand_dims(pc2_infered[:, :, 1], axis=-1))

        differ = (differ_x + differ_y +differ)/3

    if (color_weight):
        color2_infered = (color2_infered + 1) / 2
        color1 = (color1 + 1) / 2
        weight = 1 - tf.abs(color1 - color2_infered)
        weight = tf.reduce_mean(weight, axis=2,keep_dims=True)
        if (stop_color):
            weight = tf.stop_gradient(weight)
        differ = tf.multiply(weight, differ)

    if (loss_type=='robust'):
        loss = robust_loss_matrix(differ, eps, q)
    elif(loss_type=='huber'):
        loss = huber_loss(differ,delta)
    else:
        loss = norm2_loss(differ)

    mask = tf.stop_gradient(mask)
    loss = mask * loss
    num = tf.reduce_sum(mask,-1)
    loss = tf.reduce_sum(loss,-1) / (num+1e-6)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('rec loss', loss)
    tf.add_to_collection('losses', loss)

    return loss,num


def consistency_loss(pc2, forward_sf, backward_sf, pc2_infered,interp,mask, eps, q,radius,k,constraint,alpha1,alpha2,loss_type,delta):
    if interp=='xyz':
        backward_infered, mask_k = xyz_interpolation(pc2_infered, pc2, backward_sf, radius, k)
    elif interp=='dist':
        backward_infered, mask_k = dist_interpolation(pc2_infered, pc2, backward_sf, radius, k)
    else:
        backward_infered, mask_k = reci_d_interpolation(pc2_infered, pc2, backward_sf, radius, k)

    num_k = tf.reduce_sum(mask_k)
    if (constraint):
        left = tf.reduce_sum((forward_sf + backward_infered) * (forward_sf + backward_infered), axis=2)
        right = alpha1 * (tf.reduce_sum(forward_sf * forward_sf, axis=2) + tf.reduce_sum(backward_infered * backward_infered, axis=2)) + alpha2
        mask_lr = tf.cast((left < right), dtype=tf.float32)
        num_lr = tf.reduce_sum(mask_lr)
        num_k = tf.cast(num_k,'float32')
        num_lr = tf.cast(num_lr,'float32')
        num_k = tf.stack([num_k,num_lr])
        mask = mask * mask_k * mask_lr
    else:
        mask = mask * mask_k

    mask = tf.stop_gradient(mask)

    if (loss_type=='robust'):
        loss = mask * robust_loss_matrix(forward_sf + backward_infered, eps, q)
    elif (loss_type=='huber'):
        loss = mask * huber_loss(forward_sf + backward_infered, delta)
    else:
        loss = mask * norm2_loss(forward_sf + backward_infered)
    num = tf.reduce_sum(mask)
    loss = tf.reduce_sum(loss) / (num+1e-6)
    
    # loss = tf.reduce_sum(loss)
    return loss,num


'''def cam2_cam1(pc2_cam2,pose):
    b,n,_ = pc2_cam2.shape
    rotation = pose[:,:3,:3]
    translation = pose[:,:3,-1]
    pc2_cam2 = tf.transpose(pc2_cam2,[0,2,1])    #(b,3,n)
    rotated_pc2 = tf.matmul(rotation,pc2_cam2)
    translation = tf.expand_dims(translation,2)
    pc2_cam1 = rotated_pc2 + translation
    pc2_cam1 = tf.transpose(pc2_cam1,[0,2,1])
    #pc2_cam1 = tf.stop_gradient(pc2_cam1)
    return pc2_cam1'''


def cam2_cam1(pc2_cam2, pose):
    b, n, _ = pc2_cam2.shape
    ones = tf.ones(shape=(b, n, 1), dtype=tf.float32)
    pc2_cam2 = tf.concat([pc2_cam2, ones], axis=2)
    pc2_cam2 = tf.transpose(pc2_cam2, [0, 2, 1])
    pc2_cam1 = tf.matmul(pose, pc2_cam2)
    pc2_cam1 = tf.transpose(pc2_cam1, [0, 2, 1])
    pc2_cam1 = pc2_cam1[:, :, :3]
    return pc2_cam1


'''def cam2_cam1(pc2_cam2,pose):
    b,n,_ = pc2_cam2.shape
    ones = tf.ones(shape=(b,n,1),dtype=tf.float32)
    pc2_cam2 = tf.concat([pc2_cam2,ones],axis = 2)
    pc2_cam2 = tf.transpose(pc2_cam2,[0,2,1])
    pose_inv = tf.matrix_inverse(pose)
    pc2_cam1 = tf.matmul(pose_inv,pc2_cam2)
    pc2_cam1 = tf.transpose(pc2_cam1,[0,2,1])
    pc2_cam1 = pc2_cam1[:,:,:3]
    return pc2_cam1'''


def cam1_cam2(pc1_cam1, pose):
    b, n, _ = pc1_cam1.shape
    ones = tf.ones(shape=(b, n, 1), dtype=tf.float32)
    pc1_cam1 = tf.concat([pc1_cam1, ones], axis=2)
    pc1_cam1 = tf.transpose(pc1_cam1, [0, 2, 1])
    pose_inverse = tf.matrix_inverse(pose)
    pc1_cam2 = tf.matmul(pose_inverse,pc1_cam1)
    pc1_cam2 = tf.transpose(pc1_cam2, [0, 2, 1])
    pc1_cam2 = pc1_cam2[:, :, :3]
    return pc1_cam2


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        if grads != []:
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def generate_batch_mask(pc,P_rect,overlap,img_shape=(376,1241)):
    bsize,n,_ = pc.shape
    for i in range(bsize):
        #h,w = tf.slice(img_shape,[i,0,0],[1,1,2])
        h = img_shape[0]
        w = img_shape[1]
        focal_x = P_rect[i, 0, 0]
        focal_y = P_rect[i, 1, 1]
        cx = P_rect[i, 0, 2]
        cy = P_rect[i, 1, 2]
        const_x = P_rect[i,0,3]
        const_y = P_rect[i,1,3]
        const_z = P_rect[i,2,3]


        #pts_img = tf.zeros([n, 3], dtype=np.float32)
        pts_imgx = tf.expand_dims((focal_x * pc[i,:, 0] + cx * pc[i,:, 2] + const_x)/(pc[i,:,2]+const_z),1)
        pts_imgy = tf.expand_dims((focal_y * pc[i,:, 1] + cy * pc[i,:, 2] + const_y)/(pc[i,:,2]+const_z),1)
        pts_imgz = tf.expand_dims((pc[i,:, 2]),1)
        pts_imgx = tf.round(pts_imgx) - 1
        pts_imgy = tf.round(pts_imgy) - 1
        mask1 = (pts_imgx >= 0) & (pts_imgy >= 0) & (pts_imgx < w) & (pts_imgy < h)
        mask1 = tf.squeeze(mask1)
        mask1 = tf.expand_dims(mask1,0)
        pts_img = tf.concat((pts_imgx, pts_imgy), axis=-1)
        pts_img = tf.concat((pts_img, pts_imgz), axis=-1)

        unique_pts,idx = tf.unique(tf.squeeze(pts_imgy * w + pts_imgx))
        min_depth = tf.unsorted_segment_min(pts_img[:,2], idx,tf.shape(unique_pts)[0])
        min_depth_pts = tf.gather(min_depth, idx)
        mask2 = tf.where(tf.less_equal(pts_img[:,2], min_depth_pts), tf.ones_like(pts_img[:,2]),tf.zeros_like(pts_img[:,2]))
        mask2 = tf.expand_dims(mask2,0)
        if (overlap):
            mask0 = tf.cast(mask1,dtype=tf.float32)
        else:
            mask0 = tf.multiply(tf.cast(mask1,dtype=tf.float32),mask2)

        if i == 0:
            mask = mask0
        else:
            mask = tf.concat((mask,mask0),0)
    mask = tf.stop_gradient(mask)
    return mask

def projection2img(pc,P_rect):
    bsize, n, _ = pc.shape
    for i in range(bsize):
        focal_x = P_rect[i, 0, 0]
        focal_y = P_rect[i, 1, 1]
        cx = P_rect[i, 0, 2]
        cy = P_rect[i, 1, 2]
        const_x = P_rect[i, 0, 3]
        const_y = P_rect[i, 1, 3]
        const_z = P_rect[i, 2, 3]

        x = tf.expand_dims((focal_x * pc[i, :, 0] + cx * pc[i, :, 2] + const_x) / (pc[i, :, 2] + const_z), 1)
        y = tf.expand_dims((focal_y * pc[i, :, 1] + cy * pc[i, :, 2] + const_y) / (pc[i, :, 2] + const_z), 1)
        z = tf.zeros([n,1],dtype=tf.float32)

        pts_imgi = tf.concat((x, y), axis=-1)
        pts_imgi = tf.concat((pts_imgi, z), axis=-1)
        pts_imgi =  tf.expand_dims(pts_imgi,0)

        if i == 0:
            pts_img = pts_imgi
        else:
            pts_img = tf.concat((pts_img,pts_imgi),0)


    return pts_img

def get_batch_2d_flow(pc1, pc2, predicted_pc2, paths):
    focallengths = []
    cxs = []
    cys = []
    constx = []
    consty = []
    constz = []
    for path in paths:
        fname = os.path.split(path)[-1]
        calib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'utils/calib_cam_to_cam',
            fname[:6] + '.txt')
        with open(calib_path) as fd:
            lines = fd.readlines()
            P_rect_left = \
                np.array([float(item) for item in
                          [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                         dtype=np.float32).reshape(3, 4)
            focallengths.append(-P_rect_left[0, 0])
            cxs.append(P_rect_left[0, 2])
            cys.append(P_rect_left[1, 2])
            constx.append(P_rect_left[0, 3])
            consty.append(P_rect_left[1, 3])
            constz.append(P_rect_left[2, 3])
    focallengths = np.expand_dims(np.array(focallengths),1)
    cxs = np.expand_dims(np.array(cxs),1)
    cys = np.expand_dims(np.array(cys),1)
    constx = np.expand_dims(np.array(constx),1)
    consty = np.expand_dims(np.array(consty),1)
    constz = np.expand_dims(np.array(constz),1)

    px1, py1 = project_3d_to_2d(pc1, f=focallengths, cx=cxs, cy=cys,
                                constx=constx, consty=consty, constz=constz)
    px2, py2 = project_3d_to_2d(predicted_pc2, f=focallengths, cx=cxs, cy=cys,
                                constx=constx, consty=consty, constz=constz)
    px2_gt, py2_gt = project_3d_to_2d(pc2, f=focallengths, cx=cxs, cy=cys,
                                      constx=constx, consty=consty, constz=constz)

    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1
    flow_y_gt = py2_gt - py1

    print('----------------focallengths-----------', focallengths.shape)
    print('-----------------px1-------------------', px1.shape)
    print('-----------------flow_x-------------------', flow_x.shape)

    flow_pred = np.stack((flow_x, flow_y), axis=2)
    flow_gt = np.stack((flow_x_gt, flow_y_gt), axis=2)
    return flow_pred, flow_gt


def project_3d_to_2d(pc, f=-1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[:,:, 0] * f + cx * pc[:,:, 2] + constx) / (pc[:,:, 2] + constz)
    y = (pc[:,:, 1] * f + cy * pc[:,:, 2] + consty) / (pc[:,:, 2] + constz)

    return x, y

if __name__ == '__main__':
    with tf.Graph().as_default():
        # inputs = tf.zeros((32,1024*2,6))
        pc1 = tf.zeros((32, 1024, 3))
        pc2 = tf.zeros((32, 1024, 3))
        color1 = tf.zeros((32, 1024, 3))
        color2 = tf.zeros((32, 1024, 3))
        pose = tf.ones((32, 4, 4))
        outputs = get_model(pc1, pc2, color1, color2, tf.constant(True))
        pc2_cam2 = cam2cam(pc2, pose)
        print(pc2_cam2)
