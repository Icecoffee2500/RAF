# %%
import json, os
import shutil
import random
from copy import deepcopy
from collections import OrderedDict


# %% 
def get_random_images(images, nums=10, seed=40):
    random.seed(seed)
    get_random_images = random.sample(images, k=nums)
    return get_random_images


# %%
def copy_images(images, source_root, img_dir_name, dest_root, new_dir_name):
    new_dir_path = os.path.join(dest_root, new_dir_name)
    
    # 만약 만들려는 폴더가 이미 존재하면 그 폴더와 하위 항목까지 전부 삭제
    if os.path.exists(new_dir_path):
        shutil.rmtree(new_dir_path)
    
    os.makedirs(new_dir_path)
    
    for img in images:
        img_name = format(img, "012") + ".jpg"
        src_file = os.path.join(source_root, img_dir_name, img_name)
        dest_file = os.path.join(new_dir_path, img_name)
        shutil.copyfile(src_file, dest_file)
    
    print(f"The number of new images is {len(images)} in new directory.")
    print(f"Success copy to images in {new_dir_path}")


def copy_anntations(
    images, json_file, dest_root, new_json_dir_name, new_json_file_name, is_remove=True
):
    new_images = []
    new_annotations = []
    new_dir_path = os.path.join(dest_root, new_json_dir_name)
    if is_remove:
        if os.path.exists(new_dir_path):
            shutil.rmtree(new_dir_path)
        os.makedirs(new_dir_path)
    # look for annotations
    for img in json_file["images"]:
        if img["id"] in images:
            new_images.append(img)

    for ann in json_file["annotations"]:
        if ann["image_id"] in images:
            new_annotations.append(ann)

    # update and save
    new_json_file = deepcopy(json_file)
    new_json_file["images"] = new_images
    new_json_file["annotations"] = new_annotations
    with open(os.path.join(new_dir_path, new_json_file_name), "w") as ff:
        json.dump(new_json_file, ff)
    print(f"Success copy to {new_dir_path}/{new_json_file_name}!")


def copy_person_detection_results(
    json_file, dest_root, new_json_dir_name, new_json_file_name
):
    new_dir_path = os.path.join(dest_root, new_json_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)
    new_json_file = deepcopy(json_file)
    with open(os.path.join(new_dir_path, new_json_file_name), "w") as ff:
        json.dump(new_json_file, ff)
    print(f"Success copy to {new_dir_path}/{new_json_file_name}!")


def create(
    src_root="/home/djjin/Mywork/Lets-do-HPE/data/coco_tiny",
    dest_root="/data/coco_tiny",
    num_train_imgs=1000,
    num_test_imgs=100,
    num_seed=0,
):
    os.makedirs(src_root, exist_ok=True)
    source_root = src_root
    annot_data_dir = os.path.join(source_root, "annotations")

    train_json_file_name = "person_keypoints_train2017.json"
    val_json_file_name = "person_keypoints_val2017.json"

    dest_root = dest_root
    new_json_dir_name = "annotations"

    new_train_json_file_name = "person_keypoints_train2017_tiny.json"
    new_train_imgs_name = "images/train2017_tiny"

    new_val_json_file_name = "person_keypoints_val2017_tiny.json"
    new_val_imgs_name = "images/val2017_tiny"

    # %%
    # Train
    with open(os.path.join(annot_data_dir, train_json_file_name), "r") as f:
        train_coco = json.load(f, object_pairs_hook=OrderedDict)

    train_cur_imgs = []
    for img in train_coco["images"]:
        train_cur_imgs.append(img["id"])
    print(f"The number of train images is {len(train_cur_imgs)} in current directory.")

    train_images = get_random_images(train_cur_imgs, num_train_imgs, seed=num_seed)

    copy_images(
        train_images, source_root, "images/train2017", dest_root, new_train_imgs_name
    )
    copy_anntations(
        train_images, train_coco, dest_root, new_json_dir_name, new_train_json_file_name
    )

    # %%
    # Validation
    print("=" * 60)
    with open(os.path.join(annot_data_dir, val_json_file_name), "r") as f:
        val_coco = json.load(f, object_pairs_hook=OrderedDict)

    val_cur_imgs = []
    for img in val_coco["images"]:
        val_cur_imgs.append(img["id"])
    print(f"The number of val images is {len(val_cur_imgs)} in current dir.")

    real_val_images = []
    val_images = get_random_images(val_cur_imgs, num_test_imgs)
    val_one_image = 49759
    for image in val_images:
        real_val_images.append(image)
    real_val_images.append(val_one_image)
    print(f"real_val_images => {real_val_images}")
    copy_images(real_val_images, source_root, "images/val2017", dest_root, new_val_imgs_name)
    copy_anntations(
        val_images,
        val_coco,
        dest_root,
        new_json_dir_name,
        new_val_json_file_name,
        is_remove=False,
    )

    print("=" * 60)
    detection_dir = "person_detection_results"
    current_detection_dir = os.path.join(source_root, detection_dir)
    test_detection_json_name = "COCO_test-dev2017_detections_AP_H_609_person.json"
    val_detection_json_name = "COCO_val2017_detections_AP_H_56_person.json"

    # copy COCO_test-dev2017_detections_AP_H_609_person.json
    with open(os.path.join(current_detection_dir, test_detection_json_name), "r") as f:
        test_detection_json = json.load(f, object_pairs_hook=OrderedDict)

    copy_person_detection_results(
        test_detection_json,
        dest_root,
        detection_dir,
        test_detection_json_name,
    )

    # copy COCO_val2017_detections_AP_H_56_person.json
    with open(os.path.join(current_detection_dir, val_detection_json_name), "r") as f:
        val_detection_json = json.load(f, object_pairs_hook=OrderedDict)

    copy_person_detection_results(
        val_detection_json,
        dest_root,
        detection_dir,
        val_detection_json_name,
    )


# %%
if __name__ == "__main__":
    create(
        src_root="/home/djjin/Mywork/Lets-do-HPE/data/coco",
        num_train_imgs=1000,
        num_test_imgs=100,
        num_seed=0,
    )
