import os
import json
import shutil
import random
from copy import deepcopy
from collections import OrderedDict
from scipy.io import loadmat, savemat
import numpy as np

class MPIISplitter:
    
    def __init__(self, src_root, dst_name_root, num_seed, num_groups):
        self.src_root = src_root
        self.dst_name_root = dst_name_root
        self.num_seed = num_seed
        self.num_groups = num_groups

        print(self.src_root)
        print(self.dst_name_root)
    
    def create(self):

        # Source path
        src_path = dict(
            ann_dir = os.path.join(self.src_root, "annot"),
            image_dir = os.path.join(self.src_root, "images"),
            gt_file = "gt_valid.mat",
            train_json_file = "train.json",
            val_json_file = "valid.json"
        )

        # Destination path
        dst_path_lst = [
            dict(
                dst_root = f"{self.dst_name_root}_{i}",
                ann_dir = "annot",
                train_json_file = f"train_split_{i}.json",
                val_json_file = "valid_half.json",
            ) for i in range(self.num_groups)
        ]

        # Add proxy path
        dst_path_lst.append(dict(
                dst_root = f"{self.dst_name_root}_proxy",
                ann_dir = "annot",
                train_json_file = f"train_proxy.json",
                val_json_file = "valid_half.json",
            ))
        
        with open(os.path.join(src_path["ann_dir"], src_path["train_json_file"]), "r") as f:
            train_json = json.load(f, object_pairs_hook=OrderedDict) # train annotation
        
        # print(f"train json type: {type(train_json)}")
        
        src_train_json_idx = list(range(len(train_json))) # 0 ~ 22245
        dst_train_json_idxs = self.random_split_index(src_train_json_idx, self.num_groups)
        # print(f"src train json : \n{src_train_json_idx}")
        # print(f"length of dst train json[0]: {len(dst_train_json_idxs[0])}")
        # print(f"dst train json[0] : \n{dst_train_json_idxs[0]}")
        # print(f"length of dst train json[1]: {len(dst_train_json_idxs[1])}")
        # print(f"dst train json[1] : \n{dst_train_json_idxs[1]}")
        # print(f"length of dst train json[2]: {len(dst_train_json_idxs[2])}")
        # print(f"dst train json[2] : \n{dst_train_json_idxs[2]}")

        with open(os.path.join(src_path["ann_dir"], src_path["val_json_file"]), "r") as f:
            val_json = json.load(f, object_pairs_hook=OrderedDict)
        
        src_val_json_idx = list(range(len(val_json))) # 0 ~ 2958
        dst_val_json_idxs = self.random_split_index(src_val_json_idx, 2)
        dst_proxy_train_json_idx = dst_val_json_idxs[0]
        dst_val_json_idx = dst_val_json_idxs[1]

        # print(f"src val json : \n{src_val_json_idx}")
        # print(f"length of dst proxy train json: {len(dst_proxy_train_json_idx)}")
        # print(f"dst proxy train json[0] : \n{dst_proxy_train_json_idx}")
        # print(f"length of dst val json: {len(dst_val_json_idx)}")
        # print(f"dst val json[0] : \n{dst_val_json_idx}")

        # Split Train Client/Proxy datasets
        for i, dst_path in enumerate(dst_path_lst):
            # train
            self.copy_annotations_to_dest(
                indexes=dst_proxy_train_json_idx if "proxy" in dst_path["dst_root"] else dst_train_json_idxs[i],
                src_json_file=val_json if "proxy" in dst_path["dst_root"] else train_json,
                dest_root=dst_path["dst_root"],
                new_json_dir_name=dst_path["ann_dir"],
                new_json_file_name=dst_path["train_json_file"],
            )
            # valid
            self.copy_annotations_to_dest(
                indexes=dst_val_json_idx,
                src_json_file=val_json,
                dest_root=dst_path["dst_root"],
                new_json_dir_name=dst_path["ann_dir"],
                new_json_file_name=dst_path["val_json_file"],
            )
            # images 폴더 생성
            # new_image_dir_path = os.path.join(dst_path["dst_root"], "images") # /data/mpii_split_0/images
            src_image_dir_path = src_path["image_dir"]
            target_image_dir_path = os.path.join(dst_path["dst_root"], "images") # /data/mpii_split_0/images
            # os.makedirs(new_image_dir_path, exist_ok=True)
            
            # 심볼릭 링크 생성
            os.symlink(src_image_dir_path, target_image_dir_path)
            print(f"Symbolic link created from {src_image_dir_path} to {target_image_dir_path}")

        
        # split gt_valid
        src_gt_file = os.path.join(self.src_root, 'annot', 'gt_valid.mat') # gt_valid.mat파일을 불러옴.
        gt_dict = loadmat(src_gt_file)
        jnt_missing = gt_dict['jnt_missing'] # (16, 2958)
        pos_pred_src = gt_dict['pos_pred_src'] # (16, 2, 2958)
        pos_gt_src = gt_dict['pos_gt_src'] # (16, 2, 2958)
        headboxes_src = gt_dict['headboxes_src'] # (2, 2, 2958)

        val_jnt_missing = jnt_missing[:, dst_val_json_idx]
        val_pos_pred_src = pos_pred_src[:, :, dst_val_json_idx]
        val_pos_gt_src = pos_gt_src[:, :, dst_val_json_idx]
        val_headboxes_src = headboxes_src[:, :, dst_val_json_idx]

        print("val_jnt_missing shape:", val_jnt_missing.shape)  # 예상 결과: (16, 1479)
        print("val_pos_pred_src shape:", val_pos_pred_src.shape)  # 예상 결과: (16, 2, 1479)
        print("val_pos_gt_src shape:", val_pos_gt_src.shape)  # 예상 결과: (16, 2, 1479)
        print("val_headboxes_src shape:", val_headboxes_src.shape)  # 예상 결과: (2, 2, 1479)

        new_gt_dict = deepcopy(gt_dict)
        new_gt_dict['jnt_missing'] = val_jnt_missing
        new_gt_dict['pos_pred_src'] = val_pos_pred_src
        new_gt_dict['pos_gt_src'] = val_pos_gt_src
        new_gt_dict['headboxes_src'] = val_headboxes_src

        print("new_jnt_missing shape:", new_gt_dict['jnt_missing'].shape)  # 예상 결과: (16, 1479)

        for dst_path in dst_path_lst:
            output_dir = os.path.join(dst_path["dst_root"], dst_path["ann_dir"])

            new_gt_dict_file = os.path.join(output_dir, 'gt_valid_half.mat')
            savemat(new_gt_dict_file, mdict=new_gt_dict)


    # src_lst를 num_groups 개수만큼으로 쪼갠다.
    # 예시: [[img1], [img2], [img3]]
    def random_split_index(self, src_lst, num_groups):
        dst_lst = deepcopy(src_lst)
        random.shuffle(dst_lst)
        
        random_train_img_idx_sets = []
        count = 0
        length = len(src_lst) // num_groups
        for i in range(num_groups):
            img_set = []
            if i == num_groups - 1:
                length = len(dst_lst) - count
            img_set[: length] = dst_lst[count : count + length]
            random_train_img_idx_sets.append(img_set)
            count += length
        
        return random_train_img_idx_sets
    
    def copy_annotations_to_dest(
        self, indexes, src_json_file, dest_root, new_json_dir_name, new_json_file_name, is_remove=False
    ):
        new_dir_path = os.path.join(dest_root, new_json_dir_name) # /data/mpii_split_0/annot
        
        if is_remove:
            # 만약 만들려는 폴더가 이미 존재하면 그 폴더와 하위 항목까지 전부 삭제
            if os.path.exists(new_dir_path):
                shutil.rmtree(new_dir_path)
        os.makedirs(new_dir_path, exist_ok=True)
        
        new_json = [src_json_file[idx] for idx in indexes]
        
        with open(os.path.join(new_dir_path, new_json_file_name), "w") as ff:
            json.dump(new_json, ff)
        
        print(f"Success copy to {new_dir_path}/{new_json_file_name}!")
        print(f"Length of new json: {len(new_json)}")