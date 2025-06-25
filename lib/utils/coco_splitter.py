# %%
import os
import json
import shutil
import random
from copy import deepcopy
from collections import OrderedDict

# %%

class COCOSplitter:
    
    def __init__(self, src_root, dst_name_root, num_seed, num_groups, val_proxy_rate=None):
        self.src_root = src_root
        self.dst_name_root = dst_name_root
        self.num_seed = num_seed
        self.num_groups = num_groups
        self.val_proxy_rate = val_proxy_rate
    
    def create(self):
        os.makedirs(self.src_root, exist_ok=True) # coco를 가져올 폴더 만들기 ??
        
        src_path = dict(
            ann_dir = os.path.join(self.src_root, "annotations"),
            train_json_file = "person_keypoints_train2017.json",
            val_json_file = "person_keypoints_val2017.json"
        )
        dst_path_lst = [dict(
            dst_root = f"{self.dst_name_root}_{i}",
            ann_dir = "annotations",
            train_json_file = f"person_keypoints_train2017_split_{i}.json",
            train_imgs = f"images/train2017_split_{i}",
            val_json_file = "person_keypoints_val2017_split.json",
            val_imgs = "images/val2017_split"
        ) for i in range(self.num_groups)]

        # Add proxy folder
        if self.val_proxy_rate is not None:
            dst_path_lst.append(dict(
                dst_root = os.path.join(os.path.dirname(self.dst_name_root), "coco_proxy"),
                ann_dir = "annotations",
                train_json_file = f"person_keypoints_train2017_proxy.json",
                train_imgs = f"images/train2017_proxy",
                val_json_file = "person_keypoints_val2017_proxy.json",
                val_imgs = "images/val2017_proxy"
            ))
        
        
        
        # source/train의 annotation 파일을 읽기모드로 열어서 dictionary로 train_coco에 저장
        with open(os.path.join(src_path["ann_dir"], src_path["train_json_file"]), "r") as f:
            train_coco = json.load(f, object_pairs_hook=OrderedDict) # train annotation
        
        # train_imgs_idx_ordered: train set의 이미지 index를 저장 (ordered)
        src_train_imgs_idx = [img["id"] for img in train_coco["images"]]
        print(f"The number of train images is {len(src_train_imgs_idx)} in current directory.")
        
        # [coco_img_1, coco_img_2, coco_img_3]
        # coco_img_1 = [img1, img2, ...]
        dst_train_img_idx_sets = self.random_split_images(src_train_imgs_idx, self.num_groups)
        
        with open(os.path.join(src_path["ann_dir"],
                               src_path["val_json_file"]), "r") as f:
            val_coco = json.load(f, object_pairs_hook=OrderedDict)
        
        # src_val_imgs_idx: val set의 이미지 index를 저장 (ordered)
        src_val_imgs_idx = [img["id"] for img in val_coco["images"]]
        print(f"The number of val images is {len(src_val_imgs_idx)} in current dir.")

        # dst_val_img_idx_sets = self.random_split_images(src_val_imgs_idx, self.num_groups)

        dst_proxy_train_img_idx, dst_val_img_idx = self.random_split_val_proxy_images_with_ratio(src_val_imgs_idx, self.val_proxy_rate)
        print(f"proxy train idx => {sorted(dst_proxy_train_img_idx)}")
        print(f"val idx => {sorted(dst_val_img_idx)}")

        print("=" * 60)
        
        # Split Train ----------------------------------------------------------------
        
        # coco_split
        # 각 split set에 image와 annotation 저장
        # for i, img_idx_set in enumerate(dst_train_img_idx_sets):
        #     self.copy_images_to_dest(
        #         images=img_idx_set,
        #         source_root=self.src_root,
        #         src_img_dir="images/train2017",
        #         dest_root=dst_path_lst[i]["dst_root"],
        #         new_dir_name=dst_path_lst[i]["train_imgs"]
        #     )
        #     self.copy_annotations_to_dest(
        #         images=img_idx_set,
        #         src_json_file=train_coco,
        #         dest_root=dst_path_lst[i]["dst_root"],
        #         new_json_dir_name=dst_path_lst[i]["ann_dir"],
        #         new_json_file_name=dst_path_lst[i]["train_json_file"],
        #     )
        
        # Split Train Client/Proxy datasets
        for i, dst_path in enumerate(dst_path_lst):
            self.copy_images_to_dest(
                images=dst_proxy_train_img_idx if "proxy" in dst_path["dst_root"] else dst_train_img_idx_sets[i],
                source_root=self.src_root,
                src_img_dir="images/val2017" if "proxy" in dst_path["dst_root"] else "images/train2017",
                dest_root=dst_path["dst_root"],
                new_dir_name=dst_path["train_imgs"]
            )
            self.copy_annotations_to_dest(
                images=dst_proxy_train_img_idx if "proxy" in dst_path["dst_root"] else dst_train_img_idx_sets[i],
                src_json_file=val_coco if "proxy" in dst_path["dst_root"] else train_coco,
                dest_root=dst_path["dst_root"],
                new_json_dir_name=dst_path["ann_dir"],
                new_json_file_name=dst_path["train_json_file"],
            )
        
        # if self.val_proxy_rate is not None:
        #     # coco_proxy/images/train2017_proxy에 proxy imgs 복사
        #     self.copy_images_to_dest(
        #         images=dst_proxy_train_img_idx,
        #         source_root=self.src_root,
        #         src_img_dir="images/val2017",
        #         dest_root=dst_path_lst[-1]["dst_root"],
        #         new_dir_name=dst_path_lst[-1]["train_imgs"]
        #     )
        #     # coco_proxy/annotations/person_keypoints_train2017_proxy.json에 proxy anns 복사
        #     self.copy_annotations_to_dest(
        #         images=dst_proxy_train_img_idx,
        #         src_json_file=val_coco,
        #         dest_root=dst_path_lst[-1]["dst_root"],
        #         new_json_dir_name=dst_path_lst[-1]["ann_dir"],
        #         new_json_file_name=dst_path_lst[-1]["train_json_file"],
        #     )


        # Split Test ----------------------------------------------------------------
    
        # Split Test Client/Proxy datasets
        for dst_path in dst_path_lst:
            self.copy_images_to_dest(
                images=dst_val_img_idx,
                source_root=self.src_root,
                src_img_dir="images/val2017",
                dest_root=dst_path["dst_root"],
                new_dir_name=dst_path["val_imgs"]
            )
            self.copy_annotations_to_dest(
                images=dst_val_img_idx,
                src_json_file=val_coco,
                dest_root=dst_path["dst_root"],
                new_json_dir_name=dst_path["ann_dir"],
                new_json_file_name=dst_path["val_json_file"],
                is_remove=False,
        )
        
        # Split person_detection_results
        print("=" * 60)
        detection_dir = "person_detection_results"
        current_detection_dir = os.path.join(self.src_root, detection_dir)
        test_detection_json_name = "COCO_test-dev2017_detections_AP_H_609_person.json"
        val_detection_json_name = "COCO_val2017_detections_AP_H_56_person.json"

        # copy COCO_test-dev2017_detections_AP_H_609_person.json
        with open(os.path.join(current_detection_dir, test_detection_json_name), "r") as f:
            test_detection_json = json.load(f, object_pairs_hook=OrderedDict)

        for dst_path in dst_path_lst:
            self.copy_person_detection_results(
                test_detection_json,
                dst_path["dst_root"],
                detection_dir,
                test_detection_json_name,
            )

        # copy COCO_val2017_detections_AP_H_56_person.json
        with open(os.path.join(current_detection_dir, val_detection_json_name), "r") as f:
            val_detection_json = json.load(f, object_pairs_hook=OrderedDict)

        for dst_path in dst_path_lst:
            self.copy_person_detection_results(
                val_detection_json,
                dst_path["dst_root"],
                detection_dir,
                val_detection_json_name,
            )
        
    
    # src_lst를 num_groups 개수만큼으로 쪼갠다.
    # 예시: [[img1], [img2], [img3]]
    def random_split_images(self, src_lst, num_groups):
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
    
    # val image의 index를 val/proxy index로 쪼개주는 함수
    def random_split_val_proxy_images_with_ratio(self, src_val_lst, proxy_ratio):
        dst_lst = deepcopy(src_val_lst)
        random.shuffle(dst_lst)
        
        random_proxy_img_idx = []
        random_val_img_idx = []
        proxy_length = int(len(src_val_lst) * proxy_ratio)
        val_length = len(src_val_lst) - proxy_length

        random_proxy_img_idx[: proxy_length] = dst_lst[0 : proxy_length]
        random_val_img_idx[: val_length] = dst_lst[proxy_length :]
        
        return random_proxy_img_idx, random_val_img_idx
    
    def copy_images_to_dest(self, images, source_root, src_img_dir, dest_root, new_dir_name):
        
        new_dir_path = os.path.join(dest_root, new_dir_name)
        
        # 만약 만들려는 폴더가 이미 존재하면 그 폴더와 하위 항목까지 전부 삭제
        if os.path.exists(new_dir_path):
            shutil.rmtree(new_dir_path)
        
        os.makedirs(new_dir_path)
        
        for img in images:
            img_name = format(img, "012") + ".jpg"
            src_file = os.path.join(source_root, src_img_dir, img_name)
            dest_file = os.path.join(new_dir_path, img_name)
            shutil.copyfile(src_file, dest_file)
        
        print(f"The number of new images is {len(images)} in new directory.")
        print(f"Success copy to images in {new_dir_path}")
    
    def copy_annotations_to_dest(
        self, images, src_json_file, dest_root, new_json_dir_name, new_json_file_name, is_remove=True
    ):
        new_dir_path = os.path.join(dest_root, new_json_dir_name) # /data/coco_tiny/annotations
        
        if is_remove:
            # 만약 만들려는 폴더가 이미 존재하면 그 폴더와 하위 항목까지 전부 삭제
            if os.path.exists(new_dir_path):
                shutil.rmtree(new_dir_path)
        os.makedirs(new_dir_path, exist_ok=True)
        
        # look for annotations
        new_images = [img for img in src_json_file["images"] if img["id"] in images]
        new_annotations = [ann for ann in src_json_file["annotations"] if ann["image_id"] in images]

        # update and save
        new_json_file = deepcopy(src_json_file)
        new_json_file["images"] = new_images
        new_json_file["annotations"] = new_annotations
        
        with open(os.path.join(new_dir_path, new_json_file_name), "w") as ff:
            json.dump(new_json_file, ff)
        
        print(f"Success copy to {new_dir_path}/{new_json_file_name}!")

    def copy_person_detection_results(self, json_file, dest_root,
                                      new_json_dir_name, new_json_file_name):
        new_dir_path = os.path.join(dest_root, new_json_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
        
        new_json_file = deepcopy(json_file)
        
        with open(os.path.join(new_dir_path, new_json_file_name), "w") as ff:
            json.dump(new_json_file, ff)
        
        print(f"Success copy to {new_dir_path}/{new_json_file_name}!")


# %%
if __name__ == "__main__":
    COCOSplitter.create(
        src_root="/home/djjin/Mywork/Lets-do-HPE/data/coco",
        num_train_imgs=1000,
        num_test_imgs=100,
        num_seed=0,
    )
