"""
    FlowNet3D model with up convolution
"""

from tensorflow.examples.tutorials.mnist import input_data
from collections import Counter
import tensorflow as tf
import numpy as np
import math
import sys
import os
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))


import utils.tf_util as tf_util
from utils.pointnet_util import *
from tf_grouping import knn_point,group_point
from tf_interpolate import three_nn, three_interpolate

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * 2, 6))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, masks_pl


def Generator(point_cloud, is_training, bn_decay=None):
    """ FlowNet3D, for training
        input: Bx(N1+N2)x3,
        output: BxN1x3 """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // 2

    l0_xyz_f1 = point_cloud[:, :num_point, 0:3]
    l0_points_f1 = point_cloud[:, :num_point, 3:]
    l0_xyz_f2 = point_cloud[:, num_point:, 0:3]
    l0_points_f2 = point_cloud[:, num_point:, 3:]

    RADIUS1 = 0.5
    RADIUS2 = 1.0
    RADIUS3 = 2.0
    RADIUS4 = 4.0
    
    with tf.variable_scope('sa1', reuse=tf.AUTO_REUSE) as scope:
        # Frame 1, Layer 1
        l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1, npoint=1024, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        end_points['l1_indices_f1'] = l1_indices_f1

        # Frame 1, Layer 2
        l2_xyz_f1, l2_points_f1, l2_indices_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        end_points['l2_indices_f1'] = l2_indices_f1

        scope.reuse_variables()
        # Frame 2, Layer 1
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=1024, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        # Frame 2, Layer 2
        l2_xyz_f2, l2_points_f2, l2_indices_f2 = pointnet_sa_module(l1_xyz_f2, l1_points_f2, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    _, l2_points_f1_new = flow_embedding_module(l2_xyz_f1, l2_xyz_f2, l2_points_f1, l2_points_f2, radius=10, nsample=64, mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding', bn=True, pooling='max', knn=True, corr_func='concat')

    # Layer 3
    l3_xyz_f1, l3_points_f1, l3_indices_f1 = pointnet_sa_module(l2_xyz_f1, l2_points_f1_new, npoint=64, radius=RADIUS3, nsample=8, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    end_points['l3_indices_f1'] = l3_indices_f1

    # Layer 4
    l4_xyz_f1, l4_points_f1, l4_indices_f1 = pointnet_sa_module(l3_xyz_f1, l3_points_f1, npoint=16, radius=RADIUS4, nsample=8, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    end_points['l4_indices_f1'] = l4_indices_f1

    # Feature Propagation
    l3_feat_f1 = set_upconv_module(l3_xyz_f1, l4_xyz_f1, l3_points_f1, l4_points_f1, nsample=8, radius=2.4, mlp=[], mlp2=[256,256], scope='up_sa_layer1', is_training=is_training, bn_decay=bn_decay, knn=True)
    l2_feat_f1 = set_upconv_module(l2_xyz_f1, l3_xyz_f1, tf.concat(axis=-1, values=[l2_points_f1, l2_points_f1_new]), l3_feat_f1, nsample=8, radius=1.2, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer2', is_training=is_training, bn_decay=bn_decay, knn=True)
    l1_feat_f1 = set_upconv_module(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_feat_f1, nsample=8, radius=0.6, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer3', is_training=is_training, bn_decay=bn_decay, knn=True)
    l0_feat_f1 = pointnet_fp_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_feat_f1, [256,256], is_training, bn_decay, bn=True, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_feat_f1, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points

def Discriminator(point_cloud, flow, is_training, bn_decay=None):
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // 2

    l0_xyz_f1 = point_cloud[:, :num_point, 0:3]
    l0_points_f1 = point_cloud[:, :num_point, 3:]
    l0_xyz_f2 = point_cloud[:, num_point:, 0:3]
    l0_points_f2 = point_cloud[:, num_point:, 3:]

    l0_xyz_fake = tf.add(l0_xyz_f1, flow[:, :, 0:3])
    l0_points_fake = l0_points_f1

    #l0_xyz = tf.concat([l0_xyz_f2[:, :1024, 0:3], l0_xyz[:, 1024:, 0:3]], axis=1)
    #l0_points = tf.concat([l0_points_f2[:, :1024, :], l0_points[:, 1024:, :]], axis=1)

    l0_xyz_real = l0_xyz_f2
    l0_points_real = l0_points_f2

    '''
    l0_xyz = tf.concat([l0_xyz_f2[:, :, :], l0_xyz[:, :, :]], axis=1)
    l0_points = tf.concat([l0_points_f2[:, :, :], l0_points[:, :, :]], axis=1)
    '''
    

    RADIUS1 = 0.5
    RADIUS2 = 1.0
    RADIUS3 = 2.0
    RADIUS4 = 4.0


    with tf.variable_scope('dis', reuse=tf.AUTO_REUSE) as scope_1:
        with tf.variable_scope('sa1', reuse=tf.AUTO_REUSE) as scope_2:
            #Layer1
            l1_xyz_fake, l1_points_fake, l1_indices_fake = pointnet_sa_module(l0_xyz_fake, l0_points_fake, npoint=1024, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

            #Layer2
            l2_xyz_fake, l2_points_fake, l2_indices_fake = pointnet_sa_module(l1_xyz_fake, l1_points_fake, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

            scope_2.reuse_variables()

            #Layer1
            l1_xyz_real, l1_points_real, l1_indices_real = pointnet_sa_module(l0_xyz_real, l0_points_real, npoint=1024, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

            #Layer2
            l2_xyz_real, l2_points_real, l2_indices_real = pointnet_sa_module(l1_xyz_real, l1_points_real, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

            scope_2.reuse_variables()

            #Layer1
            l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1, npoint=1024, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

            #Layer2
            l2_xyz_f1, l2_points_f1, l2_indices_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

        with tf.variable_scope('sa3', reuse=tf.AUTO_REUSE) as scope_4:
            _, l2_points_f1_fake = flow_embedding_module(l2_xyz_real, l2_xyz_fake, l2_points_real, l2_points_fake, radius=10, nsample=64, mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, bn=True, scope='flow_embedding', pooling='max', knn=True, corr_func='concat')
            scope_4.reuse_variables()
            _, l2_points_f1_real = flow_embedding_module(l2_xyz_fake, l2_xyz_real, l2_points_fake, l2_points_real, radius=10, nsample=64, mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, bn=True, scope='flow_embedding', pooling='max', knn=True, corr_func='concat')

        with tf.variable_scope('sa2', reuse=tf.AUTO_REUSE) as scope_3:
            #Layer3
            l3_xyz_fake, l3_points_fake, l3_indices_fake = pointnet_sa_module(l2_xyz_f1, l2_points_f1_fake, npoint=32, radius=RADIUS3, nsample=8, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='d_layer3')
            #end_points['l1_indices'] = l3_indices_fake

            #Layer4
            l4_xyz_fake, l4_points_fake, l4_indices_fake = pointnet_sa_module(l3_xyz_fake, l3_points_fake, npoint=8, radius=RADIUS4, nsample=8, mlp=[256, 256, 512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='d_layer4')
            #end_points['l1_indices'] = l4_indices

            #l4 = tf.concat([l4_xyz, l4_points], 2)
            #flatten = tf.layers.flatten(l4_points)

            logits_fake = tf_util.conv1d(l4_points_fake, 256, 1, padding='VALID', is_training=is_training, scope='d_fc1', bn=True, bn_decay=bn_decay)
            logits_fake = tf_util.conv1d(logits_fake, 32, 1, padding='VALID', is_training=is_training, scope='d_fc2', bn=True, bn_decay=bn_decay)
            logits_fake = tf.reshape(logits_fake, (-1, 1, 256))
            #logits = tf_util.conv1d(logits, 4, 1, padding='VALID', bn=True, is_training=is_training, scope='d_fc3', bn_decay=bn_decay)
            #logits = tf.layers.dense(logits, 4)
            logits_fake = tf_util.conv1d(logits_fake, 1, 1, padding='VALID', activation_fn=None, scope='d_fc', is_training=is_training)
            out_fake = tf.nn.sigmoid(logits_fake)

            scope_3.reuse_variables()

            #Layer3
            l3_xyz_real, l3_points_real, l3_indices_real = pointnet_sa_module(l2_xyz_f1, l2_points_f1_real, npoint=32, radius=RADIUS3, nsample=8, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='d_layer3')
            #end_points['l1_indices'] = l3_indices_fake

            #Layer4
            l4_xyz_real, l4_points_real, l4_indices_real = pointnet_sa_module(l3_xyz_real, l3_points_real, npoint=8, radius=RADIUS4, nsample=8, mlp=[256, 256, 512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='d_layer4')
            #end_points['l1_indices'] = l4_indices

            #l4 = tf.concat([l4_xyz, l4_points], 2)
            #flatten = tf.layers.flatten(l4_points)

            logits_real = tf_util.conv1d(l4_points_real, 256, 1, padding='VALID', is_training=is_training, scope='d_fc1', bn=True, bn_decay=bn_decay)
            logits_real = tf_util.conv1d(logits_real, 32, 1, padding='VALID', is_training=is_training, scope='d_fc2', bn=True, bn_decay=bn_decay)
            logits_real = tf.reshape(logits_real, (-1, 1, 256))
            #logits = tf_util.conv1d(logits, 4, 1, padding='VALID', bn=True, is_training=is_training, scope='d_fc3', bn_decay=bn_decay)
            #logits = tf.layers.dense(logits, 4)
            logits_real = tf_util.conv1d(logits_real, 1, 1, padding='VALID', activation_fn=None, scope='d_fc', is_training=is_training)
            out_real = tf.nn.sigmoid(logits_real)

    return out_fake, logits_fake, out_real, logits_real


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

def cam2_cam1(pc2_cam2, pose):
    b, n, _ = pc2_cam2.shape
    ones = tf.ones(shape=(b, n, 1), dtype=tf.float32)
    pc2_cam2 = tf.concat([pc2_cam2, ones], axis=2)
    pc2_cam2 = tf.transpose(pc2_cam2, [0, 2, 1])
    pc2_cam1 = tf.matmul(pose, pc2_cam2)
    pc2_cam1 = tf.transpose(pc2_cam1, [0, 2, 1])
    pc2_cam1 = pc2_cam1[:, :, :3]
    return pc2_cam1

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


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)


def get_loss(pred, label):
    """ pred: BxNx3,
        label: BxNx3,
        mask: BxN
    """
    batch_size = pred.get_shape()[0].value
    num_point = pred.get_shape()[1].value
    l2_loss = tf.reduce_mean(tf.reduce_sum((pred-label) * (pred-label), axis=2) / 2.0)
    tf.summary.scalar('l2 loss', l2_loss)
    tf.add_to_collection('losses', l2_loss)
    return l2_loss


def GANLoss(d_model_real, d_logits_real, d_model_fake, d_logits_fake, is_training):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                    labels=tf.ones_like(d_model_real)*0.9))

    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
     
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                labels= tf.ones_like(d_model_fake)))
     
    d_loss = d_loss_real + d_loss_fake
     
    return d_loss, g_loss


def GLoss(d_model_real, d_logits_real, d_model_fake, d_logits_fake, is_training):
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                labels= tf.ones_like(d_model_fake)))
    return g_loss


def DLoss(d_model_real, d_logits_real, d_model_fake, d_logits_fake, is_training):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                    labels=tf.ones_like(d_model_real)))

    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
     
    d_loss = d_loss_real + d_loss_fake
     
    return d_loss


def NN_loss(pc1,pc2,forward_sf):
    pred_pc2 = pc1 + forward_sf
    val,idx = knn_point(1,pc2,pred_pc2)
    neighbor_pc2 = group_point(pc2,idx)
    neighbor_pc2 = tf.squeeze(neighbor_pc2, axis=2)
    loss = tf.reduce_mean(tf.reduce_sum((pred_pc2 - neighbor_pc2)*(pred_pc2 - neighbor_pc2),axis=2))

    return loss

def chamfer_loss(pc1,pc2,forward_sf):
    pred_pc2 = pc1 + forward_sf

    val,idx1 = knn_point(1,pc2,pred_pc2)
    neighbor_pc2 = group_point(pc2,idx1)
    neighbor_pc2 = tf.squeeze(neighbor_pc2, axis=2)
    loss1 = tf.reduce_mean(tf.reduce_sum((pred_pc2 - neighbor_pc2)*(pred_pc2 - neighbor_pc2),axis=2))

    val,idx2 = knn_point(1,pred_pc2,pc2)
    neighbor_pred_pc2 = group_point(pred_pc2,idx2)
    neighbor_pred_pc2 = tf.squeeze(neighbor_pred_pc2, axis=2)
    loss2 = tf.reduce_mean(tf.reduce_sum((pc2 - neighbor_pred_pc2)*(pc2 - neighbor_pred_pc2),axis=2))

    return loss1+loss2

def cycle_consistency_loss(forward_sf,backward_sf):
    loss = tf.reduce_mean(tf.reduce_sum((forward_sf + backward_sf) * (forward_sf + backward_sf), axis=2))

    return loss

'''def cylinder_projection(pc):
    delta_phi = 0.4
    delta_theta = 0.2
     w ='''

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


# def smooth_loss(pc1,forward_sf,color1,nsample,radius,eps,q,loss_type,delta):
def smooth_loss(pc1,forward_sf,color1,nsample=16,radius=0.5):
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
    # if(loss_type=='robust'):
    #     loss = robust_loss_matrix((forward_sf - neighbor_sf),eps,q) * neighbor_weight
    # elif (loss_type=='huber'):
    #     loss = huber_loss((forward_sf - neighbor_sf),delta) * neighbor_weight
    # else:
    #     loss = norm2_loss(forward_sf - neighbor_sf) * neighbor_weight
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

def curvature(pc):
    val, index = knn_point(10, pc, pc)
    grouped_pc = group_point(pc, index)
    #grouped_pc = tf.squeeze(grouped_pc, axis=2)
    pc_curvature = tf.reduce_sum(grouped_pc - tf.expand_dims(pc, 2), axis=2) / 9.0
    return pc_curvature

def curvatureWarp(pc, warp_pc):
    val, index = knn_point(10, pc, pc)
    grouped_pc = group_point(warp_pc, index)
    #grouped_pc = tf.squeeze(grouped_pc, axis=2)
    pc_curvature = tf.reduce_sum(grouped_pc - tf.expand_dims(warp_pc, 2), axis=2) / 9.0
    return pc_curvature

def interpolateCurvature(pc1_warp, pc2, pc2_curvature):
    dist, index = three_nn(pc1_warp, pc2)
    weight = tf.ones_like(dist)/3.0
    interpolated_pc2_curvature = three_interpolate(pc2_curvature, index, weight)
    return interpolated_pc2_curvature

def curvature_loss(pc1, pc2, forward_sf):
    pc1_warp = pc1 + forward_sf

    pc1_warp_curvature = curvatureWarp(pc1, pc1_warp)
    pc2_curvature = curvature(pc2)
    interpolated_pc2_curvature = interpolateCurvature(pc1_warp, pc2, pc2_curvature)

    loss = tf.reduce_mean(
        tf.reduce_sum((interpolated_pc2_curvature - pc1_warp_curvature) * (interpolated_pc2_curvature - pc1_warp_curvature),
        axis=2))
    return loss

def computesmooth(pc1, pred):
    _, kidx = knn_point(9, pc1, pc1)
    grouped_flow = group_point(pred, kidx)
    diff_flow = tf.reduce_mean(
                        
                            tf.reduce_sum(
                                tf.reduce_sum((grouped_flow - tf.expand_dims(pred, 2)) * (grouped_flow - tf.expand_dims(pred, 2)),
                                 axis=3), axis=2) / 8.0)
    return diff_flow



if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = Generator(inputs, tf.constant(True))
        print(outputs)
