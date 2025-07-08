from raf.segmentation.dataset.cityscapes import CityscapesDataset
from raf.segmentation.dataset.transforms import (
    Compose, RandomCrop, Resize,
    ColorJitter, RandomHorizontalFlip,
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
    transforms = []
    transforms.append(Resize(transform_config.resolution))
    if is_train:
        transforms.append(RandomCrop(size=(transform_config.crop_size, transform_config.crop_size)))
        transforms.append(ColorJitter(
            brightness=transform_config.brightness,
            contrast=transform_config.contrast,
            saturation=transform_config.saturation,
        ))
        transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
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