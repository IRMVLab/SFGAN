'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob


class SceneflowDataset():
    def __init__(self, root='../data', npoints=2048, depth_threshold=30,train=True):
        self.npoints = npoints
        self.train = train
        self.root = root
        self.depth_threshold = depth_threshold
        if self.train:
          self.datapath = []
          for sequence in range(11):
            sequence = str(sequence).zfill(2)
            #self.datapath += glob.glob(os.path.join(self.root,'depth_img_new_noresize/dataset/sequences',sequence,'image2','*.npz'))
            #self.datapath += glob.glob(os.path.join(self.root,'depth_img_new_noresize/dataset/sequences',sequence,'image3','*.npz'))
            #self.datapath += glob.glob(os.path.join(self.root, 'velo2cam_pc/dataset/sequences', sequence,'*.npz'))
            self.datapath += glob.glob(os.path.join(self.root, 'data_odometry_velodyne/sequences/', sequence, 'velodyne/*.bin'))

        else:
            self.datapath = glob.glob(os.path.join(self.root,'depth_img_new_noresize/dataset/sequences', '01/image2','*.npz'))
        self.cache = {}
        self.cache_size = 30000


    def __getitem__(self, index):
        velo1_path = self.datapath[index]
        path1_split = velo1_path.split('/')
        idx1 = int(path1_split[-1][:6])
        idx2 = str(idx1 + 1).zfill(6)
        velo2_path = os.path.join(self.root, 'data_odometry_velodyne/sequences/', path1_split[-3], 'velodyne/',idx2+'.bin')

        if os.path.exists(velo2_path):
            pos1 = np.fromfile(velo1_path, dtype=np.float32).reshape(-1, 4)
            pos1 = pos1[:, :3]
            pos2 = np.fromfile(velo2_path, dtype=np.float32).reshape(-1, 4)
            pos2 = pos2[:, :3]

            within_depth1 = np.logical_and((pos1[:, 0] < self.depth_threshold), (pos1[:, 0] > -self.depth_threshold))
            within_depth2 = np.logical_and((pos2[:, 0] < self.depth_threshold), (pos2[:, 0] > -self.depth_threshold))
            within_lr1 = np.logical_and((pos1[:, 1] < self.depth_threshold), (pos1[:, 1] > -self.depth_threshold))
            within_lr2 = np.logical_and((pos2[:, 1] < self.depth_threshold), (pos2[:, 1] > -self.depth_threshold))
            within_surround1 = np.logical_and(within_depth1, within_lr1)
            within_surround2 = np.logical_and(within_depth2, within_lr2)
            is_ground1 = (pos1[:, 2] < -1.1)
            not_ground1 = np.logical_not(is_ground1)
            is_ground2 = (pos2[:, 2] < -1.1)
            not_ground2 = np.logical_not(is_ground2)
            sampled_indices1 = np.logical_and(within_surround1, not_ground1)
            sampled_indices2 = np.logical_and(within_surround2, not_ground2)

            pos1 = pos1[sampled_indices1]
            pos2 = pos2[sampled_indices2]
            #print(velo1_path,pos1.shape,velo2_path,pos2.shape)

            if self.train:
                n1 = pos1.shape[0]
                if n1 >= self.npoints:
                    sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
                else:
                    sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
                n2 = pos2.shape[0]
                if n2 >= self.npoints:
                    sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
                else:
                    sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

                pos1_ = np.copy(pos1[sample_idx1, :])
                pos2_ = np.copy(pos2[sample_idx2, :])
                color1_ = np.zeros((pos1_.shape))
                color2_ = np.zeros((pos2_.shape))
            else:
                pos1_ = np.copy(pos1[:self.npoints, :])
                pos2_ = np.copy(pos2[:self.npoints, :])
                color1_ = np.zeros((pos1_.shape))
                color2_ = np.zeros((pos2_.shape))
        else:
            pos1_ = np.zeros((self.npoints,3))
            pos2_ = np.zeros((self.npoints,3))
            color1_ = np.zeros((self.npoints,3))
            color2_ = np.zeros((self.npoints,3))

        return pos1_, pos2_, color1_, color2_

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    # import mayavi.mlab as mlab
    d = SceneflowDataset(npoints=2048)
    print(len(d))
    import time
    tic = time.time()
    for i in range(100):
        pc1, pc2, c1, c2, flow, m1, m2 = d[i]

        print(pc1.shape)
        print(pc2.shape)
        print(flow.shape)
        '''print(np.sum(m1))
        print(np.sum(m2))
        pc1_m1 = pc1[m1==1,:]
        pc1_m1_n = pc1[m1==0,:]
        print(pc1_m1.shape)
        print(pc1_m1_n.shape)
        mlab.points3d(pc1_m1[:,0], pc1_m1[:,1], pc1_m1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc1_m1_n[:,0], pc1_m1_n[:,1], pc1_m1_n[:,2], scale_factor=0.05, color=(0,1,0))
        raw_input()

        mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=0.05, color=(0,1,0))
        raw_input()
        mlab.quiver3d(pc1[:,0], pc1[:,1], pc1[:,2], flow[:,0], flow[:,1], flow[:,2], scale_factor=1)
        raw_input()'''

    print(time.time() - tic)
    print(pc1.shape, type(pc1))


