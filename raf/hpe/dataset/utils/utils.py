from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torch.utils.data import Sampler
import numpy as np
import random

class SingleResFromMultiRes(Dataset):
    """
    Wrap a MultiResDataset that returns ([img_r1, img_r2, ...], label)
    into a dataset that returns one resolution per sample.
    If base_len = N and R resolutions, new length = N * R.
    Order: idx -> base_idx = idx // R, res_idx = idx % R
    """
    def __init__(self, base_multi_res_dataset):
        self.base = base_multi_res_dataset  # Subset or original
        # underlying dataset (MRFairFaceDataset)
        underlying = base_multi_res_dataset.dataset if isinstance(base_multi_res_dataset, Subset) else base_multi_res_dataset
        self.R = len(underlying.resolutions)
        self.N = len(base_multi_res_dataset)   # Subset.__len__ already reflects sampled size

        print(f"resolution is {self.R}")

    def __len__(self):
        return self.N * self.R

    def __getitem__(self, idx):
        base_idx = idx // self.R
        res_idx = idx % self.R
        imgs_list, label = self.base[base_idx]    # imgs_list: list of tensors (C,H,W)
        single_img = imgs_list[res_idx]
        return single_img, label


def sample_dataset_subset(dataset, n_samples, seed=42):
    N = len(dataset)
    if n_samples >= N:
        return dataset  # 그냥 원본 반환
    rng = np.random.RandomState(seed)
    indices = rng.choice(N, size=n_samples, replace=False)
    return Subset(dataset, indices)

class ResolutionBatchSampler(Sampler):
    """
    dataset_len: len(single_res_dataset) == N * R
    R: number of resolutions
    batch_size: batch size
    shuffle: whether to shuffle within each resolution
    drop_last: whether to drop last smaller batch per resolution
    """
    def __init__(self, dataset_len, R, batch_size, shuffle=True, drop_last=False, seed=None):
        assert dataset_len % R == 0, "dataset_len should be N * R"
        self.dataset_len = dataset_len
        self.R = R
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.N = dataset_len // R
        self.seed = seed

        # precompute indices for each resolution:
        # for res r, indices are: r, r+R, r+2R, ..., r+(N-1)*R
        self.group_indices = []
        for r in range(R):
            self.group_indices.append([r + k * R for k in range(self.N)])

    def __iter__(self):
        rng = random.Random(self.seed)
        # For each group create a local copy and optionally shuffle
        groups = [g.copy() for g in self.group_indices]
        if self.shuffle:
            for g in groups:
                rng.shuffle(g)

        # Generate batches: we will iterate groups in round-robin or random order
        # Option A: random order of groups each epoch to mix which resolution appears earlier
        order = list(range(self.R))
        if self.shuffle:
            rng.shuffle(order)

        batches = []
        for r in order:
            g = groups[r]
            # split g into batches
            for i in range(0, len(g), self.batch_size):
                batch = g[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        # finally shuffle the order of batches if desired (optional)
        if self.shuffle:
            rng.shuffle(batches)

        # yield flat index sequence of batches
        for batch in batches:
            yield batch

    def __len__(self):
        # total number of batches across all resolutions
        if self.drop_last:
            per_group = self.N // self.batch_size
        else:
            per_group = (self.N + self.batch_size - 1) // self.batch_size
        return per_group * self.R


class MixedResolutionDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.total_len = len(original_dataset)
        
        # 3등분 구간 설정 (총 12000개라면 4000개씩)
        self.split_1 = self.total_len // 3
        self.split_2 = (self.total_len // 3) * 2
        
    def __getitem__(self, index):
        # 1. 원본 데이터셋에서 모든 해상도 버전을 가져옴
        # 가정: original_dataset[index]는 (imgs_list, hms_list, weights_list, meta) 형태라고 가정
        # imgs_list = [img_high, img_mid, img_low] (각각 3D Tensor)
        imgs, hms, weights, meta = self.dataset[index]
        
        # 2. 인덱스 구간에 따라 해상도 선택
        if index < self.split_1:
            # 0 ~ 1/3 구간: High Resolution (Index 0)
            selected_img = imgs[0]
            selected_hm = hms[0]
            selected_weight = weights[0]
            resolution_id = 0 # 배칭을 위해 식별자 필요시 사용
            
        elif index < self.split_2:
            # 1/3 ~ 2/3 구간: Mid Resolution (Index 1)
            selected_img = imgs[1]
            selected_hm = hms[1]
            selected_weight = weights[1]
            resolution_id = 1
            
        else:
            # 2/3 ~ 끝 구간: Low Resolution (Index 2)
            selected_img = imgs[2]
            selected_hm = hms[2]
            selected_weight = weights[2]
            resolution_id = 2
            
        return selected_img, selected_hm, selected_weight, meta

    def __len__(self):
        return self.total_len

class HeteroBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        
        # 구간 정의
        self.split_1 = dataset_len // 3
        self.split_2 = (dataset_len // 3) * 2
        
        # 인덱스 그룹화
        self.high_indices = list(range(0, self.split_1))
        self.mid_indices = list(range(self.split_1, self.split_2))
        self.low_indices = list(range(self.split_2, self.dataset_len))
        
    def __iter__(self):
        # 1. 각 그룹 내에서 인덱스 셔플 (선택 사항, 데이터 순서도 섞고 싶다면)
        random.shuffle(self.high_indices)
        random.shuffle(self.mid_indices)
        random.shuffle(self.low_indices)
        
        # 2. 각 그룹별로 배치 생성
        batches = []
        
        # High Batches
        for i in range(0, len(self.high_indices), self.batch_size):
            batches.append(self.high_indices[i : i + self.batch_size])
            
        # Mid Batches
        for i in range(0, len(self.mid_indices), self.batch_size):
            batches.append(self.mid_indices[i : i + self.batch_size])
            
        # Low Batches
        for i in range(0, len(self.low_indices), self.batch_size):
            batches.append(self.low_indices[i : i + self.batch_size])
            
        # 3. 배치 단위로 셔플 (High 배치, Mid 배치, Low 배치가 섞임)
        random.shuffle(batches)
        
        # 4. yield
        for batch in batches:
            yield batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size