'''
    Single-GPU training code
'''

import argparse
import math
from datetime import datetime
from model_concat_upsa2 import chamfer_loss, computesmooth
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import flying_things_dataset
import kitti_dataset
import kitti_dataset_self_supervised_cycle
import pickle
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
from pointnet_util import *
from tf_grouping import knn_point,group_point

#import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=3, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa2', help='Model name [default: model_concat_upsa]')
parser.add_argument('--data', default='data_preprocessing/data_processed_maxcut_35_20k_2k_8192', help='Dataset directory [default: data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--data_kitti', default='../kitti_self_supervised_flow')
parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=151, help='Epoch to run [default: 151]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate_g', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--learning_rate_d', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--nn_decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--pre_trained_dir', default='', help='pretrained_model')
parser.add_argument('--lamda', type=float, default=0.5, help='Weight for anchor point [default: 0.5]')
parser.add_argument('--gd_frequency', type=float, default=1.0, help='g:d')
parser.add_argument('--weight_loss', type=float, default=50.0)
parser.add_argument('--weight_g', type=float, default=1.0)
parser.add_argument('--weight_cc',  type=float, default=1.0)
parser.add_argument('--weight_smooth',  type=float, default=10.0)
parser.add_argument('--weight_chamfer',  type=float, default=1.0)
parser.add_argument('--weight_cur',  type=float, default=1.0)
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

EPOCH_CNT = 0
EPOCH_TRAIN_CNT = 0

NUMBER_GPUS = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATA = FLAGS.data
DATA_KITTI = FLAGS.data_kitti
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE_G = FLAGS.learning_rate_g
BASE_LEARNING_RATE_D = FLAGS.learning_rate_d
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
PRETRAINED_DIR = FLAGS.pre_trained_dir
LAMDA = FLAGS.lamda
GD_F = FLAGS.gd_frequency
WEIGHT_LOSS = FLAGS.weight_loss
NN = FLAGS.nn_decay_step
WEIGHT_G = FLAGS.weight_g
WEIGHT_CC = FLAGS.weight_cc
WEIGHT_CHAMFER = FLAGS.weight_chamfer
WEIGHT_SMOOTH = FLAGS.weight_smooth
WEIGHT_CUR = FLAGS.weight_cur

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
#os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (FLAGS.model+'.py', LOG_DIR))
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure

# if DATA=='kitti_rm_ground':
#     os.system('cp %s %s' % ('kitti_dataset.py', LOG_DIR)) # bkp of dataset file
# else:
#     os.system('cp %s %s' % ('flying_things_dataset.py', LOG_DIR)) # bkp of dataset file
os.system('cp %s %s' % ('flying_things_dataset.py', LOG_DIR))
os.system('cp %s %s' % ('kitti_dataset.py', LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


# TRAIN_DATASET_KITTI = flying_things_dataset.SceneflowDataset(DATA, npoints=NUM_POINT, train=0)
# TEST_DATASET = flying_things_dataset.SceneflowDataset(DATA, npoints=NUM_POINT, train=1)
# TRAIN_DATASET_KITTI = kitti_dataset.SceneflowDataset(DATA, npoints=NUM_POINT, train=True)
# TEST_DATASET_KITTI = kitti_dataset.SceneflowDataset(DATA_KITTI, npoints=NUM_POINT, train=False)
TRAIN_DATASET_KITTI = kitti_dataset_self_supervised_cycle.SceneflowDataset(DATA_KITTI, npoints=NUM_POINT, train=True)
TEST_DATASET_KITTI = kitti_dataset.SceneflowDataset(DATA_KITTI, npoints=NUM_POINT, train=False)




def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate_g(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE_G,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_learning_rate_d(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE_D,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def get_nn(batch):
    w_nn = tf.train.exponential_decay(
                        1.0,  
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        NN,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    return w_nn

def train():
    tf.reset_default_graph()
    with tf.Graph().as_default():       #默认图
        with tf.device('/gpu:'+str(GPU_INDEX)):     #制定GPU
            with tf.variable_scope('const', reuse=tf.AUTO_REUSE) as scope:
                pointclouds_pl, labels_pl, masks_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
                is_training_pl_g = tf.placeholder(tf.bool, shape=())
                is_training_pl_d = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                #pointclouds_pl_split = tf.split(pointclouds_pl,NUMBER_GPUS)


            with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:

                print("--- Get model and loss")
                # Get model and loss
                pred, end_points = MODEL.Generator(pointclouds_pl, is_training_pl_g, bn_decay=bn_decay)
                d_model_fake, d_logits_fake, d_model_real, d_logits_real = MODEL.Discriminator(pointclouds_pl, pred, is_training=is_training_pl_d, bn_decay=bn_decay)

                
                pc1 = pointclouds_pl[:,:NUM_POINT,0:3]
                color1 = pointclouds_pl[:, :NUM_POINT, 3:]
                pc2 = pointclouds_pl[:, NUM_POINT:, 0:3]
                color2 = pointclouds_pl[:, NUM_POINT:, 3:]

                pred_pc2 = pc1 + pred
                val, idx = knn_point(1, pc2, pred_pc2)
                neighbor_pc2 = group_point(pc2, idx)
                neighbor_pc2 = tf.squeeze(neighbor_pc2, axis=2)
                neighbor_color2 = group_point(color2, idx)
                neighbor_color2 = tf.squeeze(neighbor_color2, axis=2)

                anchor_pc2 = LAMDA * pred_pc2 + (1 - LAMDA) * neighbor_pc2
                anchor_color2 = LAMDA * color1 + (1 - LAMDA) * neighbor_color2
                anchor_p2 = tf.concat((anchor_pc2, anchor_color2), axis=2)
                p1 = tf.concat((pc1, color1), axis=2)
                pointclouds_b = tf.concat((anchor_p2, p1), axis=1)

                backward_sf, end_points_b = MODEL.Generator(pointclouds_b, is_training_pl_g, bn_decay=bn_decay)
                

            # w_nn = get_nn(batch)
            # NN_loss = MODEL.NN_loss(pc1, pc2, pred)
            Chamfer_loss = MODEL.chamfer_loss(pc1, pc2, pred)
            CC_loss = MODEL.cycle_consistency_loss(pred, backward_sf)
            # Smooth_loss = MODEL.smooth_loss(pc1, pred, color1)
            Curvature_loss = MODEL.curvature_loss(pc1, pc2, pred)
            smooth_loss = MODEL.computesmooth(pc1, pred)
            loss = WEIGHT_CC*CC_loss + WEIGHT_SMOOTH*smooth_loss + WEIGHT_CHAMFER*Chamfer_loss + WEIGHT_CUR*Curvature_loss

            #grads = optimizer.compute_gradients(loss)

            
            g_loss = MODEL.GLoss(d_model_real, d_logits_real, d_model_fake, d_logits_fake, is_training_pl_g)
            g_loss_total = WEIGHT_G*g_loss + WEIGHT_LOSS*loss
            d_loss = MODEL.DLoss(d_model_real, d_logits_real, d_model_fake, d_logits_fake, is_training_pl_d)

            tf.summary.scalar('d_loss', d_loss)
            tf.summary.scalar('g_loss', g_loss)
            # tf.summary.scalar('NN_loss', NN_loss)
            tf.summary.scalar('CC_loss', CC_loss)
            tf.summary.scalar('cf_loss', Chamfer_loss)
            tf.summary.scalar('curvature_loss', Curvature_loss)
            tf.summary.scalar('smooth_loss', smooth_loss)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('g_loss_total', g_loss_total)

            #with tf.variable_scope('op', reuse=tf.AUTO_REUSE) as scope:
            print("--- Get training operator")
                # Get training operator
            with tf.variable_scope('op', reuse=tf.AUTO_REUSE) as scope:
                learning_rate_g = get_learning_rate_g(batch)
                learning_rate_d = get_learning_rate_d(batch)
                tf.summary.scalar('learning_rate_g', learning_rate_g)
                tf.summary.scalar('learning_rate_d', learning_rate_d)
                if OPTIMIZER == 'momentum':
                    optimizer_g = tf.train.MomentumOptimizer(learning_rate_g, momentum=MOMENTUM)
                    optimizer_d = tf.train.MomentumOptimizer(learning_rate_d, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer_g = tf.train.AdamOptimizer(learning_rate_g)
                    optimizer_d = tf.train.AdamOptimizer(learning_rate_d)
                g_train_op = optimizer_g.minimize(g_loss_total, global_step=batch,
                                                var_list=tf.get_collection(
                                                    tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope='sa1|flow_embedding|layer3|layer4|up_sa_layer1|up_sa_layer2|up_sa_layer3|fa_layer4|fc1|fc2'))
                d_train_op = optimizer_d.minimize(d_loss, global_step=batch, 
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis'))

                # Add ops to save and restore all the variables.
            
            
            
            
            

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        
        sess = tf.Session(config=config)
        
        
        saver = tf.train.Saver(var_list=tf.get_collection(
                                            tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='sa1|flow_embedding|layer3|layer4|up_sa_layer1|up_sa_layer2|up_sa_layer3|fa_layer4|fc1|fc2|dis'))
        saver.restore(sess, PRETRAINED_DIR)
        saver_global = tf.train.Saver()
        log_string("Model restored.")

        '''
        init = tf.global_variables_initializer()
        sess.run(init)
        '''
        

        init = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='const|op'))
        sess.run(init)
        
        '''
        model_variables = slim.get_variables()
        restore_variables = [var for var in model_variables]
        for var in restore_variables:
            print(var.name)
        '''

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               #'masks_pl': masks_pl,
               'is_training_pl_g': is_training_pl_g,
               'is_training_pl_d': is_training_pl_d,
               'pred': pred,
               'backward_sf': backward_sf,
               #'NN_loss': NN_loss,
               'chamfer_loss': Chamfer_loss, 
               'curvature_loss': Curvature_loss,
               'smooth_loss': smooth_loss,
               'CC_loss': CC_loss,
               'loss': loss,
               'g_loss_total': g_loss_total,
               'd_logits_real': d_logits_real,
               'd_logits_fake': d_logits_fake,
               'g_loss': g_loss,
               'd_loss': d_loss,
               'g_train_op': g_train_op,
               'd_train_op': d_train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        epe3d_min = 10000.0
        outlier_min = 10000.0
        acc3d_1_max = 0.0
        acc3d_2_max = 0.0


        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            # epe3d, acc3d_1, acc3d_2, outlier, gsum, dsum = eval_one_epoch(sess, ops, test_writer)
            epe3d, acc3d_1, acc3d_2, outlier, gsum, dsum = eval_one_epoch_kitti(sess, ops, test_writer)


            if epe3d < epe3d_min:
                epe3d_min = epe3d
                save_path_epe3d = saver.save(sess, os.path.join(LOG_DIR, "model_epe3d_%03d.ckpt" % (epoch)))
                log_string("Model epe3d saved in file: %s" % save_path_epe3d)
            '''
            if acc3d_1 > acc3d_1_max:
                acc3d_1_max = acc3d_1
                save_path_acc3d_1 = saver.save(sess, os.path.join(LOG_DIR, "model_acc3d_1_%03d.ckpt" % (epoch)))
                log_string("Model acc3d_1 saved in file: %s" % save_path_acc3d_1)
            if acc3d_2 > acc3d_2_max:
                acc3d_2_max = acc3d_2
                save_path_acc3d_2 = saver.save(sess, os.path.join(LOG_DIR, "model_acc3d_2_%03d.ckpt" % (epoch)))
                log_string("Model acc3d_2 saved in file: %s" % save_path_acc3d_2)
            if outlier < outlier_min:
                outlier_min = outlier
                save_path_outlier = saver.save(sess, os.path.join(LOG_DIR, "model_outlier_%03d.ckpt" % (epoch)))
                log_string("Model outlier saved in file: %s" % save_path_outlier)
            '''

            save_path_latest = saver_global.save(sess, os.path.join(LOG_DIR, "model_latest.ckpt" ))
            log_string("Model latest saved in file: %s" % save_path_latest)


            if GD_F >= 1:
                if epoch % GD_F == 0:
                    train_one_epoch_d(sess, ops, train_writer)
                train_one_epoch_g(sess, ops, train_writer)
                
            
            else:
                gd_f = 1 / GD_F
                train_one_epoch_d(sess, ops, train_writer)
                if epoch % gd_f == 0:
                    train_one_epoch_g(sess, ops, train_writer)

            

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver_global.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def get_batch(dataset, idxs, start_idx, end_idx):

    bsize = end_idx-start_idx
    
    batch_data = np.zeros((bsize, NUM_POINT*2, 6))
    batch_label = np.zeros((bsize, NUM_POINT, 3))

    # shuffle idx to change point order (change FPS behavior)

    shuffle_idx = np.arange(NUM_POINT)
    np.random.shuffle(shuffle_idx)

    for i in range(bsize):

        pc1, pc2, flow= dataset[idxs[i+start_idx]]

        # move pc1 to center
        pc1_center = np.mean(pc1, 0)
        pc1 -= pc1_center
        pc2 -= pc1_center

        batch_data[i,:NUM_POINT,:3] = pc1[shuffle_idx]
        #batch_data[i,:NUM_POINT,3:] = color1[shuffle_idx]
        batch_data[i,NUM_POINT:,:3] = pc2[shuffle_idx]
        #batch_data[i,NUM_POINT:,3:] = color2[shuffle_idx]
        batch_label[i] = flow[shuffle_idx]

    return batch_data, batch_label


def get_eval_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT * 2, 6))
    batch_label = np.zeros((bsize, NUM_POINT, 3))
    # batch_mask = np.zeros((bsize, NUM_POINT))
    # shuffle idx to change point order (change FPS behavior)
    shuffle_idx = np.arange(NUM_POINT)
    np.random.shuffle(shuffle_idx)
    for i in range(bsize):
        pc1, pc2, color1, color2, flow = dataset[idxs[i + start_idx]]
        # move pc1 to center
        pc1_center = np.mean(pc1, 0)
        pc1 -= pc1_center
        pc2 -= pc1_center
        batch_data[i, :NUM_POINT, :3] = pc1[shuffle_idx]
        batch_data[i, :NUM_POINT, 3:] = color1[shuffle_idx]
        batch_data[i, NUM_POINT:, :3] = pc2[shuffle_idx]
        batch_data[i, NUM_POINT:, 3:] = color2[shuffle_idx]
        batch_label[i] = flow[shuffle_idx]
        # batch_mask[i] = mask1[shuffle_idx]
    return batch_data, batch_label  # , batch_mask

def get_cycle_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    # change here,  numpoint *(5, 3)
    batch_data = np.zeros((bsize, NUM_POINT * 2, 6))

    shuffle_idx = np.arange(NUM_POINT)

    for i in range(bsize):
        # ipdb.set_trace()
        # if dataset[0] == None:
        #     print (i, bsize)
        pos, color = dataset[idxs[i + start_idx]]

        pos1_center = np.mean(pos[0], 0) # 1 * 3

        for frame_idx in range(2):
            np.random.shuffle(shuffle_idx)
            batch_data[i, NUM_POINT*frame_idx:NUM_POINT*(frame_idx+1), :3] = \
                pos[frame_idx, shuffle_idx, :] - pos1_center
            batch_data[i, NUM_POINT*frame_idx:NUM_POINT*(frame_idx+1), 3:] = \
                color[frame_idx, shuffle_idx, :]

    return batch_data


def train_one_epoch_d(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training_d = True
    is_training_g = False

    # Shuffle train samples
    # train_idxs = np.arange(0, len(TRAIN_DATASET_KITTI))
    train_idxs = np.arange(0, 100)
    np.random.shuffle(train_idxs)
    # num_batches = len(TRAIN_DATASET_KITTI) // BATCH_SIZE
    num_batches = 100 // BATCH_SIZE

    log_string(str(datetime.now()))

    g_loss_sum = 0
    d_loss_sum = 0
    smooth_loss_sum = 0
    CC_loss_sum = 0
    cf_loss_sum = 0
    cv_loss_sum = 0
    loss_sum = 0
    g_loss_total_sum = 0
    
    for batch_idx in range(num_batches):

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(100, (batch_idx+1) * BATCH_SIZE)
        batch_data = get_cycle_batch(TRAIN_DATASET_KITTI, train_idxs, start_idx, end_idx)

        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                    # ops['labels_pl']: batch_label,
                    #ops['masks_pl']: batch_mask,
                    ops['is_training_pl_g']: is_training_g,
                    ops['is_training_pl_d']: is_training_d}
        summary, step, _, g_loss_val, d_loss_val, smooth_loss, cf_loss, CC_loss, cv_loss, loss, g_loss_total, pred_val, d_logits_real_val, d_logits_fake_val = sess.run(
            [ops['merged'], ops['step'],
            ops['d_train_op'],
            ops['g_loss'], ops['d_loss'], 
            ops['smooth_loss'], ops['chamfer_loss'], ops['CC_loss'], ops['curvature_loss'], ops['loss'], ops['g_loss_total'], 
            ops['pred'], 
            ops['d_logits_real'], ops['d_logits_fake']], feed_dict=feed_dict)
            
        train_writer.add_summary(summary, step)
        g_loss_sum += g_loss_val
        d_loss_sum += d_loss_val
        smooth_loss_sum += smooth_loss
        cf_loss_sum += cf_loss
        CC_loss_sum += CC_loss
        cv_loss_sum += cv_loss
        loss_sum += loss
        g_loss_total_sum += g_loss_total

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('g mean loss: %f' % (g_loss_sum / 10))
            log_string('d mean loss: %f' % (d_loss_sum / 10))
            
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('mean smooth loss: %f' % (smooth_loss_sum / 10))
            log_string('mean cf loss: %f' % (cf_loss_sum / 10))
            log_string('mean CC loss: %f' % (CC_loss_sum / 10))
            log_string('mean cv loss: %f' % (cv_loss_sum / 10))
            log_string('g total mean loss: %f' % (g_loss_total_sum / 10))
            
            g_loss_sum = 0
            d_loss_sum = 0
            
            smooth_loss_sum = 0
            cf_loss_sum = 0
            CC_loss_sum = 0
            cv_loss_sum = 0
            loss_sum = 0
            g_loss_total_sum = 0


def train_one_epoch_g(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training_d = False
    is_training_g = True

    # Shuffle train samples
    # train_idxs = np.arange(0, len(TRAIN_DATASET_KITTI))
    train_idxs = np.arange(0, 100)
    np.random.shuffle(train_idxs)
    # num_batches = len(TRAIN_DATASET_KITTI) // BATCH_SIZE
    num_batches = 100 // BATCH_SIZE

    log_string(str(datetime.now()))

    g_loss_sum = 0
    d_loss_sum = 0
    smooth_loss_sum = 0
    cf_loss_sum = 0
    CC_loss_sum = 0
    cv_loss_sum = 0
    loss_sum = 0
    g_loss_total_sum = 0
    
    for batch_idx in range(num_batches):

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(100, (batch_idx+1) * BATCH_SIZE)
        batch_data = get_cycle_batch(TRAIN_DATASET_KITTI, train_idxs, start_idx, end_idx)

        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                    # ops['labels_pl']: batch_label,
                    #ops['masks_pl']: batch_mask,
                    ops['is_training_pl_g']: is_training_g,
                    ops['is_training_pl_d']: is_training_d}
        summary, step, _, g_loss_val, d_loss_val, smooth_loss, cf_loss, CC_loss, cv_loss, loss, g_loss_total, pred_val, d_logits_real_val, d_logits_fake_val = sess.run(
            [ops['merged'], ops['step'],
            ops['g_train_op'],
            ops['g_loss'], ops['d_loss'], 
            ops['smooth_loss'], ops['chamfer_loss'], ops['CC_loss'], ops['curvature_loss'], ops['loss'], ops['g_loss_total'], 
            ops['pred'], 
            ops['d_logits_real'], ops['d_logits_fake']], feed_dict=feed_dict)
            
        train_writer.add_summary(summary, step)
        g_loss_sum += g_loss_val
        d_loss_sum += d_loss_val
        smooth_loss_sum += smooth_loss
        cf_loss_sum += cf_loss
        CC_loss_sum += CC_loss
        cv_loss_sum += cv_loss
        loss_sum += loss
        g_loss_total_sum += g_loss_total

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('g mean loss: %f' % (g_loss_sum / 10))
            log_string('d mean loss: %f' % (d_loss_sum / 10))
            
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('mean smooth loss: %f' % (smooth_loss_sum / 10))
            log_string('mean cf loss: %f' % (cf_loss_sum / 10))
            log_string('mean CC loss: %f' % (CC_loss_sum / 10))
            log_string('mean cv loss: %f' % (cv_loss_sum / 10))
            log_string('g total mean loss: %f' % (g_loss_total_sum / 10))
            
            g_loss_sum = 0
            d_loss_sum = 0
            
            smooth_loss_sum = 0
            cf_loss_sum = 0
            CC_loss_sum = 0
            cv_loss_sum = 0
            loss_sum = 0
            g_loss_total_sum = 0
            

def eval_one_epoch(sess, ops, test_writer):
    # ops: dict mapping from string to tf ops 
    global EPOCH_CNT
    is_training_g = False
    is_training_d = False

    test_idxs = np.arange(0, 3824)

    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (3824+BATCH_SIZE-1) // BATCH_SIZE

    g_loss_sum = 0
    loss_sum_l2 = 0
    d_loss_sum = 0

    sum_epe3d = 0
    sum_acc3d_1 = 0
    sum_acc3d_2 = 0
    sum_outlier = 0
    loss_sum = 0

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    #batch_mask = np.zeros((BATCH_SIZE, NUM_POINT))
    for batch_idx in range(num_batches):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(3824, (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
            #batch_mask = cur_batch_mask
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
            #batch_mask[0:cur_batch_size] = cur_batch_mask

        # ---------------------------------------------------------------------
        # ---- INFERENCE BELOW ----
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     #ops['masks_pl']: batch_mask,
                     ops['is_training_pl_g']: is_training_g,
                     ops['is_training_pl_d']: is_training_d}
        summary, step, g_loss_val, d_loss_val, pred_val, d_logits_real_val, d_logits_fake_val = sess.run(
            [ops['merged'], ops['step'],
            #ops['loss'],
            ops['g_loss'], ops['d_loss'], ops['pred'], 
            ops['d_logits_real'], ops['d_logits_fake']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        # ---- INFERENCE ABOVE ----
        # ---------------------------------------------------------------------

        pc1 = batch_data[:, :NUM_POINT, :3]
        color1 = batch_data[:, :NUM_POINT, 3:]
        pc2 = batch_data[:, NUM_POINT:, :3]
        color2 = batch_data[:, NUM_POINT:, 3:]
        pc1 = change_axis(pc1)
        batch_label = change_axis(batch_label)
        pred_val = change_axis(pred_val)

        error = np.linalg.norm(pred_val - batch_label, axis=-1)
        num = pred_val.shape[1]

        sf_gt_len = np.linalg.norm(batch_label, axis=-1) + 1e-20
        acc3d_1 = np.sum(np.logical_or((error <= 0.05), (error / sf_gt_len <= 0.05)), axis=1)  ###note the range
        acc3d_2 = np.sum(np.logical_or((error <= 0.1), (error / sf_gt_len <= 0.1)), axis=1)
        outlier = np.sum(np.logical_or((error > 0.3), (error / sf_gt_len > 0.1)), axis=1)
        # mask_sum = np.sum(mask, 1)
        acc3d_1 = acc3d_1 / num
        acc3d_1 = np.mean(acc3d_1)
        acc3d_2 = acc3d_2 / num
        acc3d_2 = np.mean(acc3d_2)
        outlier = outlier / num
        outlier = np.mean(outlier)
        EPE3D = np.sum(error, axis=-1) / num
        EPE3D = np.mean(EPE3D)


        tmp = np.sum((pred_val - batch_label)**2, 2) / 2.0
        #loss_val_np = np.mean(batch_mask * tmp)
        loss_val_np = np.mean(tmp)
        if cur_batch_size==BATCH_SIZE:
            g_loss_sum += g_loss_val
            loss_sum_l2 += loss_val_np
            d_loss_sum += d_loss_val
            sum_epe3d += EPE3D
            sum_acc3d_1 += acc3d_1
            sum_acc3d_2 += acc3d_2
            sum_outlier += outlier

        # Dump some results
        if batch_idx == 0:
            with open('test_results.pkl', 'wb') as fp:
                pickle.dump([batch_data, batch_label, pred_val], fp)

    log_string('g eval mean loss: %f' % (g_loss_sum / float(3824/BATCH_SIZE)))
    log_string('d eval mean loss: %f' % (d_loss_sum / float(3824/BATCH_SIZE)))
    log_string('eval mean loss: %f' % (loss_sum_l2 / float(3824/BATCH_SIZE)))


    epe3d = sum_epe3d / float(3824 // BATCH_SIZE)
    acc3d_1 = sum_acc3d_1 / float(3824 // BATCH_SIZE)
    acc3d_2 = sum_acc3d_2 / float(3824 // BATCH_SIZE)
    outlier = sum_outlier / float(3824 // BATCH_SIZE)
    '''log_string('eval mean rec loss: %f' % (rec_loss_sum / float(len(TEST_DATASET) / BATCH_SIZE)))
    log_string('eval mean depth rec loss: %f' % (depth_rec_loss_sum / float(len(TEST_DATASET) / BATCH_SIZE)))
    log_string('eval mean 3d rec loss: %f' % (rec_loss_3d_sum / float(len(TEST_DATASET) / BATCH_SIZE)))
    log_string('eval mean 3d depth rec loss: %f' % (depth_rec_loss_3d_sum / float(len(TEST_DATASET) / BATCH_SIZE)))'''
    log_string('eval mean EPE 3D: %f' % (epe3d))
    log_string('eval mean acc3d_1: %f' % (acc3d_1))
    log_string('eval mean acc3d_2 : %f' % (acc3d_2))
    log_string('eval mean outlier : %f' % (outlier))
    
    # outfile = open(os.path.join('train_eval_result', 'eval_result.txt'), 'w')
    # outfile.write('g eval mean loss: %f' % (g_loss_sum / float(50/BATCH_SIZE)) + '\n')
    # outfile.write('d eval mean loss: %f' % (d_loss_sum / float(50/BATCH_SIZE)) + '\n')
    # outfile.write('eval mean loss: %f' % (loss_sum_l2 / float(50/BATCH_SIZE)) + '\n')
    # outfile.close()

    EPOCH_CNT += 1
    return epe3d, acc3d_1, acc3d_2, outlier, g_loss_sum/float(len(TEST_DATASET)/BATCH_SIZE), d_loss_sum/float(len(TEST_DATASET)/BATCH_SIZE)



def eval_one_epoch_kitti(sess, ops, test_writer):
    # ops: dict mapping from string to tf ops 
    global EPOCH_CNT
    is_training_g = False
    is_training_d = False

    test_idxs = np.arange(0, 50)

    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (50+BATCH_SIZE-1) // BATCH_SIZE

    g_loss_sum = 0
    loss_sum_l2 = 0
    d_loss_sum = 0

    sum_epe3d = 0
    sum_acc3d_1 = 0
    sum_acc3d_2 = 0
    sum_outlier = 0
    loss_sum = 0

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION KITII ----'%(EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    #batch_mask = np.zeros((BATCH_SIZE, NUM_POINT))
    for batch_idx in range(num_batches):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(50, (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_label = get_eval_batch(TEST_DATASET_KITTI, test_idxs, start_idx, end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
            #batch_mask = cur_batch_mask
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
            #batch_mask[0:cur_batch_size] = cur_batch_mask

        # ---------------------------------------------------------------------
        # ---- INFERENCE BELOW ----
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     #ops['masks_pl']: batch_mask,
                     ops['is_training_pl_g']: is_training_g,
                     ops['is_training_pl_d']: is_training_d}
        summary, step, g_loss_val, d_loss_val, pred_val, d_logits_real_val, d_logits_fake_val = sess.run(
            [ops['merged'], ops['step'],
            #ops['loss'],
            ops['g_loss'], ops['d_loss'], ops['pred'], 
            ops['d_logits_real'], ops['d_logits_fake']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        # ---- INFERENCE ABOVE ----
        # ---------------------------------------------------------------------

        pc1 = batch_data[:, :NUM_POINT, :3]
        color1 = batch_data[:, :NUM_POINT, 3:]
        pc2 = batch_data[:, NUM_POINT:, :3]
        color2 = batch_data[:, NUM_POINT:, 3:]
        pc1 = change_axis(pc1)
        batch_label = change_axis(batch_label)
        pred_val = change_axis(pred_val)

        error = np.linalg.norm(pred_val - batch_label, axis=-1)
        num = pred_val.shape[1]

        sf_gt_len = np.linalg.norm(batch_label, axis=-1) + 1e-20
        acc3d_1 = np.sum(np.logical_or((error <= 0.05), (error / sf_gt_len <= 0.05)), axis=1)  ###note the range
        acc3d_2 = np.sum(np.logical_or((error <= 0.1), (error / sf_gt_len <= 0.1)), axis=1)
        outlier = np.sum(np.logical_or((error > 0.3), (error / sf_gt_len > 0.1)), axis=1)
        # mask_sum = np.sum(mask, 1)
        acc3d_1 = acc3d_1 / num
        acc3d_1 = np.mean(acc3d_1)
        acc3d_2 = acc3d_2 / num
        acc3d_2 = np.mean(acc3d_2)
        outlier = outlier / num
        outlier = np.mean(outlier)
        EPE3D = np.sum(error, axis=-1) / num
        EPE3D = np.mean(EPE3D)


        tmp = np.sum((pred_val - batch_label)**2, 2) / 2.0
        #loss_val_np = np.mean(batch_mask * tmp)
        loss_val_np = np.mean(tmp)
        if cur_batch_size==BATCH_SIZE:
            g_loss_sum += g_loss_val
            loss_sum_l2 += loss_val_np
            d_loss_sum += d_loss_val
            sum_epe3d += EPE3D
            sum_acc3d_1 += acc3d_1
            sum_acc3d_2 += acc3d_2
            sum_outlier += outlier

        # Dump some results
        if batch_idx == 0:
            with open('test_results.pkl', 'wb') as fp:
                pickle.dump([batch_data, batch_label, pred_val], fp)

    log_string('g eval mean loss: %f' % (g_loss_sum / float(50/BATCH_SIZE)))
    log_string('d eval mean loss: %f' % (d_loss_sum / float(50/BATCH_SIZE)))
    log_string('eval mean loss: %f' % (loss_sum_l2 / float(50/BATCH_SIZE)))


    epe3d = sum_epe3d / float(50 // BATCH_SIZE)
    acc3d_1 = sum_acc3d_1 / float(50 // BATCH_SIZE)
    acc3d_2 = sum_acc3d_2 / float(50 // BATCH_SIZE)
    outlier = sum_outlier / float(50 // BATCH_SIZE)
    '''log_string('eval mean rec loss: %f' % (rec_loss_sum / float(len(TEST_DATASET) / BATCH_SIZE)))
    log_string('eval mean depth rec loss: %f' % (depth_rec_loss_sum / float(len(TEST_DATASET) / BATCH_SIZE)))
    log_string('eval mean 3d rec loss: %f' % (rec_loss_3d_sum / float(len(TEST_DATASET) / BATCH_SIZE)))
    log_string('eval mean 3d depth rec loss: %f' % (depth_rec_loss_3d_sum / float(len(TEST_DATASET) / BATCH_SIZE)))'''
    log_string('eval mean EPE 3D: %f' % (epe3d))
    log_string('eval mean acc3d_1: %f' % (acc3d_1))
    log_string('eval mean acc3d_2 : %f' % (acc3d_2))
    log_string('eval mean outlier : %f' % (outlier))
    
    # outfile = open(os.path.join('train_eval_result', 'eval_result.txt'), 'w')
    # outfile.write('g eval mean loss: %f' % (g_loss_sum / float(50/BATCH_SIZE)) + '\n')
    # outfile.write('d eval mean loss: %f' % (d_loss_sum / float(50/BATCH_SIZE)) + '\n')
    # outfile.write('eval mean loss: %f' % (loss_sum_l2 / float(50/BATCH_SIZE)) + '\n')
    # outfile.close()

    EPOCH_CNT += 1
    return epe3d, acc3d_1, acc3d_2, outlier, g_loss_sum/float(len(TEST_DATASET_KITTI)/BATCH_SIZE), d_loss_sum/float(len(TEST_DATASET_KITTI)/BATCH_SIZE)




def change_axis(pos1):
    pos1_x = pos1[:,:, 2]
    pos1_y = -pos1[:,:, 1]
    pos1_z = pos1[:,:, 0]

    pos1 = np.stack([pos1_x, pos1_y, pos1_z], axis=-1)

    return pos1

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
