from raf.segmentation.dataset.cityscapes import CityscapesDataset
from raf.segmentation.dataset.transforms import (
    Compose, RandomCrop, Resize, RandomScale,
    ShortSideResize, RandomHorizontalFlip,
    ToTensor, Normalize
)
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# dataset_config = {
#     'dataset_root': 'cityscapes',
#     'train_split': 'train',
#     'valid_split': 'val',
#     'mode': 'gtFine',
#     'target_type': 'semantic',
#     'train_batch_size': 32,
#     'valid_batch_size': 64,
#     'num_workers': 4,
#     'pin_memory': True,
#     'transform_config': {
#         'resolution': 512,
#         'crop_size': 512,
#         'brightness': 0.5,
#         'contrast': 0.5,
#         'saturation': 0.5,
#     }
# }

class DatasetBuilder:
    def __init__(self, dataset_config):
        self.transform_config = dataset_config.transform_config
        self.train_transform = _build_transform(self.transform_config, is_train=True)
        self.valid_transform = _build_transform(self.transform_config, is_train=False)
        if 'cityscapes' in dataset_config.dataset_root:
            self.train_dataset, self.valid_dataset = _build_cityscapes_dataset_train_valid(
                dataset_root=dataset_config.dataset_root,
                train_split=dataset_config.train_split,
                valid_split=dataset_config.valid_split,
                mode=dataset_config.mode,
                target_type=dataset_config.target_type,
                transform=self.train_transform
            )
        else:
            raise ValueError(f"Dataset {dataset_config.dataset_root} not supported")
        self.train_dataloader, self.valid_dataloader = _build_train_valid_dataloader(
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            train_batch_size=dataset_config.train_batch_size,
            valid_batch_size=dataset_config.valid_batch_size,
            num_workers=dataset_config.num_workers,
            pin_memory=dataset_config.pin_memory
        )

    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_valid_dataloader(self):
        return self.valid_dataloader
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_valid_dataset(self):
        return self.valid_dataset

def _build_transform(transform_config: dict, is_train: bool = True):
    # The config is an easydict, so we can access keys with dot notation

    transforms = []
    if is_train:
        # For training, use SegFormer-style augmentations
        # 1. Random Resize with a ratio from a given range
        if transform_config.get('random_scale'):
            transforms.append(RandomScale(scale_range=transform_config.get('random_scale')))
        
        # 2. Random Horizontal Flip
        if transform_config.get('random_flip'):
            transforms.append(RandomHorizontalFlip())
        
        # 3. Random Crop to a fixed size
        # if transform_config.get('train_crop_size'):
        #     transforms.append(
        #         RandomCrop(
        #             size=transform_config.get('train_crop_size'),
        #             pad_if_needed=True
        #         )
        #     )
        if transform_config.get('train_crop_size'):
            transforms.append(
                RandomCrop(  # 위에서 정의한 custom RandomCrop
                    size=transform_config.get('train_crop_size'),
                    pad_if_needed=True,  # 활성화
                    # pad_mode='constant',  # zero-pad
                    image_pad_value=0,    # image black pad
                    label_ignore_index=255  # label ignore (Cityscapes)
                )
            )
        
    else:
        # # For validation, use a fixed resize
        # transforms.append(Resize(transform_config.resolution))
        if transform_config.get('train_crop_size'):  # short side size = training crop size (e.g., 1024 for Cityscapes)
            transforms.append(ShortSideResize(
                short_side_size=transform_config.get('train_crop_size')[0]
                ))  # Assume square, or adjust

    transforms.append(ToTensor())
    transforms.append(
        Normalize(
            mean=transform_config.normalize.mean,
            std=transform_config.normalize.std
            )
        )
    transform = Compose(transforms)
    return transform

def _build_cityscapes_dataset_train_valid(dataset_root, train_split, valid_split, mode, target_type, transform):
    train_dataset = CityscapesDataset(dataset_root, train_split, mode, target_type, transform)
    valid_dataset = CityscapesDataset(dataset_root, valid_split, mode, target_type, transform)
    return train_dataset, valid_dataset

def _build_train_valid_dataloader(train_dataset, valid_dataset, train_batch_size, valid_batch_size, num_workers, pin_memory):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    return train_dataloader, valid_dataloader