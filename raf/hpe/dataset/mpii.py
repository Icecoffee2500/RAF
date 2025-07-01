# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import json_tricks as json

import numpy as np
from scipy.io import loadmat, savemat
from hpe.dataset.joints_dataset import JointsDataset


logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, image_size, heatmap_size, is_train, transform=None):
        super().__init__(cfg, root, image_set, image_size, heatmap_size, is_train, transform)

        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info(f'=> load {len(self.db)} samples')

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.root,
                                 'annot',
                                 self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=float)
            s = np.array([a['scale'], a['scale']], dtype=float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_2d = np.zeros((self.num_joints, 2), dtype=float)
            joints_2d_vis = np.zeros((self.num_joints, 2), dtype=float)
            if self.image_set != 'test':
                joints = np.array(a['joints']) # 16개의 joints (16,2)의 shape을 가짐.
                joints[:, 0:2] = joints[:, 0:2] - 1 # 1은 왜 빼줄까?
                joints_vis = np.array(a['joints_vis']) # (16,1)의 shape을 가짐.
                assert len(joints) == self.num_joints, \
                    f'joint num diff: {len(joints)} vs {self.num_joints}'

                joints_2d[:, 0:2] = joints[:, 0:2] # (16,3)이긴 한데 마지막은 다 0
                joints_2d_vis[:, 0] = joints_vis[:]
                joints_2d_vis[:, 1] = joints_vis[:]

            # image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            image_dir = 'images'
            gt_db.append({
                'image': os.path.join(self.root, image_dir, image_name),
                'center': c,
                'scale': s,
                'rotation': 0, # new!
                'joints_2d': joints_2d,
                'joints_2d_vis': joints_2d_vis,
                'dataset': 'mpii', # new!
                'filename': '',
                'imgnum': 0,
                })

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               f'gt_{cfg.DATASET.TEST_SET}.mat') # gt_valid.mat파일을 불러옴.
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0]) # (16, 2, 2958) => (2, 2958, 16)

        # print(f"jnt_missing => {jnt_missing.shape}")
        # print(f"pos_gt_src => {pos_gt_src.shape}")
        # print(f"headboxes_src => {headboxes_src.shape}")
        # print(f"pos_pred_src => {pos_pred_src.shape}")

        # 각 joint의 index를 저장하는 것.
        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        # PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
        PCKh = np.divide(1.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01) # [0.00, 0.01, ..., 0.50] # 50개
        pckAll = np.zeros((len(rng), 16)) # (50, 16)

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold, jnt_visible)
            # pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
            pckAll[r, :] = np.divide(1.*np.sum(less_than_threshold, axis=1), jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(float)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
