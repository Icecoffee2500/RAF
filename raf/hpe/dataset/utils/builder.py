import torchvision.transforms as transforms
from typing import Type, Any, Union
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SplitJointsDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        input_img, heatmaps, heatmaps_weights, meta = self.dataset[self.idxs[item]]
        return input_img, heatmaps, heatmaps_weights, meta

def build_split_dataset(ori_dataset: Dataset, dataset_idx: int, samples_per_split: int=1000) -> SplitJointsDataset:

    total_num_samples = len(ori_dataset)  # type: ignore  # 전체 데이터 개수
    num_clients = int(total_num_samples / samples_per_split)
    client_indices = {}  # 각 split dataset을 저장할 딕셔너리
    all_idxs = np.arange(total_num_samples)  # 전체 데이터 인덱스 리스트
    
    for client_id in range(num_clients):
        client_indices[client_id] = set(np.random.choice(all_idxs, samples_per_split, replace=False))
        all_idxs = list(set(all_idxs) - client_indices[client_id])
    
    print(f"client {dataset_idx}: {[index for idx, index in enumerate(client_indices[dataset_idx]) if idx < 5]}")
    return SplitJointsDataset(ori_dataset, client_indices[dataset_idx])

def build_train_val_dataloader(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: list[int], worker: int
    ) -> tuple[DataLoader, DataLoader]:
    print(f"batch_size[0]: {batch_size[0]}")
    print(f"batch_size[1]: {batch_size[1]}")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=worker,
        pin_memory=True,
        sampler=None,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=worker,
        pin_memory=True,
    )
    
    return train_data_loader, val_data_loader