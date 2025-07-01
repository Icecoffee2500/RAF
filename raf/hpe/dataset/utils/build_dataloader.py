import torchvision.transforms as transforms
from typing import Type, Any
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

from hpe.dataset.utils.dataset_split import build_split_union_dataset, build_splitted_dataset

def build_split_union_dataloader(
    config: dict,
    dataset_class: Type[Any],
    root: str,
    image_set: str,
    image_size,
    heatmap_size,
    is_train: bool,
    split_data: bool
    ) -> DataLoader:
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = dataset_class(
        cfg=config,
        root=root,
        image_set=image_set,
        image_size=image_size,
        heatmap_size=heatmap_size,
        is_train=is_train,
        transform=  transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    if split_data:
        num_of_splits = config.DATASET.NUMBER_OF_SPLITS - 1
        dataset = build_split_union_dataset(dataset, num_of_splits)
        # dataset = build_split_union_dataset(dataset, num_of_splits, split_size=500)
        print(f"The length of splitted dataset is {len(dataset)}")
    
    sampler = None
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=(sampler is None),
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=sampler,
    )
    
    return data_loader

def build_split_dataloader(
    config: dict,
    dataset_class: Type[Any],
    dataset_idx: int,
    root: str,
    image_set: str,
    image_size,
    heatmap_size,
    is_train: bool,
    split_size: int,
    split_data: bool,
    batch_size: int,
    ) -> DataLoader:
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = dataset_class(
        cfg=config,
        root=root,
        image_set=image_set,
        image_size=image_size,
        heatmap_size=heatmap_size,
        is_train=is_train,
        transform=  transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    if split_data:
        split_size_ = split_size * 1000
        # split_size = config.DATASET.NUMBER_OF_SPLITS * 1000
        # split_size = config.DATASET.NUMBER_OF_SPLITS * 500
        dataset = build_splitted_dataset(dataset, dataset_idx, split_size=split_size_)
        print(f"The length of splitted dataset is [{len(dataset)}] and dataset index is [{dataset_idx}]")
        print(f"Image Size is [{image_size}] and Heatmap Size is [{heatmap_size}]")
    
    sampler = None
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=sampler,
    )
    
    return data_loader