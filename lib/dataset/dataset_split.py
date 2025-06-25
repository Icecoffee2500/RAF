from torch.utils.data import Dataset
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        input_img, target_joints, target_joints_vis, heatmaps, heatmaps_weights, meta = self.dataset[self.idxs[item]]
        return input_img, target_joints, target_joints_vis, heatmaps, heatmaps_weights, meta

def build_split_union_dataset(ori_dataset, num_of_splits, split_size=1000):
    dict_split_base = split_indexes_from_full_dataset(ori_dataset, split_size=split_size)
    print(f"client: {[index for idx, index in enumerate(dict_split_base[0]) if idx < 5]}")
    dict_splits = {}
    current_union = set()
    for i in range(len(dict_split_base)):
        current_union = current_union.union(dict_split_base[i])
        dict_splits[i] = current_union

    return DatasetSplit(ori_dataset, dict_splits[num_of_splits])

def split_indexes_from_full_dataset(dataset, split_size):
    num_samples = len(dataset)  # 전체 데이터 개수
    split_base_num = int(num_samples / split_size)
    splitted_indexes_dict = {}  # 각 split dataset을 저장할 딕셔너리
    all_idxs = np.arange(num_samples)  # 전체 데이터 인덱스 리스트
    
    for i in range(split_base_num):
        splitted_indexes_dict[i] = set(np.random.choice(all_idxs, split_size, replace=False))
        all_idxs = list(set(all_idxs) - splitted_indexes_dict[i])

    return splitted_indexes_dict

def build_splitted_dataset(ori_dataset, dataset_idx, split_size=1000):
    splitted_indexes_dict = split_indexes_from_full_dataset(ori_dataset, split_size=split_size)
    print(f"client {dataset_idx}: {[index for idx, index in enumerate(splitted_indexes_dict[dataset_idx]) if idx < 5]}")
    return DatasetSplit(ori_dataset, splitted_indexes_dict[dataset_idx])