# import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from timm.data import create_transform
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Subset
from torch.utils.data import Sampler
import random

# -----------------------------
# 1. 설정
# -----------------------------
ROOT_DIR = Path("./data/fairface_dataset")
# IMG_ROOT  = ROOT_DIR / "val"
IMG_ROOT  = ROOT_DIR
VAL_LABEL_CSV = ROOT_DIR / "fairface_label_val.csv"
TRAIN_LABEL_CSV = ROOT_DIR / "fairface_label_train.csv"

# FairFace의 race, gender, age 라벨 매핑 (공식 순서)
RACE_LABELS = ['White', 'Black', 'Indian', 'East Asian', 'Southeast Asian', 
               'Middle Eastern', 'Latino_Hispanic']
GENDER_LABELS = ['Male', 'Female']
AGE_LABELS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']

# -----------------------------
# 2. Custom Dataset 클래스
# -----------------------------
class MRFairFaceDataset(Dataset):
    def __init__(self, csv_path, img_root, cls_res: list, transform=None, post_transform=None):
        self.meta = pd.read_csv(csv_path)
        self.img_root = img_root
        self.resolutions = cls_res
        self.transform = transform
        self.post_transform = post_transform

        # 라벨을 인덱스로 변환
        self.meta['race_idx'] = self.meta['race'].map({v : i for i, v in enumerate(RACE_LABELS)})
        # self.meta['gender_idx'] = self.meta['gender'].map({v : i for i, v in enumerate(GENDER_LABELS)})
        # self.meta['age_idx'] = self.meta['age'].map({v : i for i, v in enumerate(AGE_LABELS)})

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.img_root / row['file']  # file 컬럼: val/xxx.jpg
        img = Image.open(img_path).convert("RGB")    # 항상 3채널로
        # if self.transform:
        #     img = self.transform(img)                # -> torch.FloatTensor (C,H,W)

        # 1. 기본 transform 적용 (ResFormer 원본처럼)
        if self.transform is not None:
            img = self.transform(img)   # 여전히 PIL.Image 유지하거나 tensor로 해도 됨

        # 2. post_transform이 있으면 여러 resolution 이미지 생성
        if self.post_transform is not None:
            trans_imgs = [tf(img) for tf in self.post_transform]  # 각 tf는 ToTensor + Normalize 포함
        else:
            # post_transform 없을 때는 기존처럼 하나만
            # 하지만 ViT 입력을 위해 최소 ToTensor/Normalize는 필요
            if self.transform is None:
                # fallback: 기본 normalization만이라도
                default_tf = transforms.Compose([
                    transforms.Resize(224, antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]
                    ),
                ])
                trans_imgs = [default_tf(img)]
            else:
                trans_imgs = [img] if isinstance(img, torch.Tensor) else [transforms.ToTensor()(img)]

        label = torch.tensor(row['race_idx'], dtype=torch.long)

        return trans_imgs, label  # trans_imgs: list[Tensor] (e.g., 3개 resolution → len=3)

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


def build_transform(is_train, input_size):
    assert torch.all(torch.sort(torch.Tensor(input_size), descending = True).values == torch.Tensor(input_size))
    resize_im = input_size[0] > 32
    input_size_high = input_size[0]
    
    # ---------------- train transform
    if is_train:
        transform = create_transform(
            input_size=input_size_high,
            is_training=True,
            # color_jitter=0.4,
            color_jitter=None,
            # auto_augment='rand-m9-mstd0.5-inc1',
            auto_augment=None,
            interpolation='bicubic',
            # re_prob=0.25,
            re_prob=0.0,
            re_mode=None,
            # re_mode='pixel',
            # re_mode='const',
            re_count=1,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(input_size_high, padding=4)
        
        if isinstance(input_size, list):
            post_t_list = []
            t1 = transform.transforms.pop(-2) # 아마도 transforms.ToTensor()
            t2 = transform.transforms.pop(-1) # 아마도 transforms.Normlize()
            
            for i, res_sz in enumerate(input_size):
                t = []
                if i > 0:
                    t.append(transforms.Resize(res_sz, interpolation=InterpolationMode.BICUBIC),)
                t.append(t1)
                if t2 is not None:
                    t.append(t2)
                post_t_list.append(transforms.Compose(t))
            return transform, post_t_list
        
        else:
            return transform, None
    
    # ---------------- val transform
    
    t_list = []

    if isinstance(input_size, list):
        for res_sz in input_size:
            t = []
            if resize_im:
                size = int((256 / 224) * res_sz)
                t.append(
                    transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(res_sz))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),)
            t_list.append(transforms.Compose(t))
        return None, t_list
    
    else:
        t = []
        if resize_im:
            size = int((256 / 224) * input_size_high)
            t.append(
                transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size_high))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        ))
        return transforms.Compose(t), None


# -----------------------------
# Visualize 함수 (랜덤으로 몇 장 보여주기)
# -----------------------------
def show_samples(dataset, num_samples=12):
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    axs = axs.ravel()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img, target = dataset[idx]
        img = img[0].permute(1, 2, 0).numpy()                     # (C,H,W) → (H,W,C)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axs[i].imshow(img)
        title = f"Race: {RACE_LABELS[target]}"
        axs[i].set_title(title, fontsize=10)
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.show()

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