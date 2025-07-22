import json
from pathlib import Path
import re

from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from PIL import Image
import numpy as np
import torch

class CityscapesDataset(Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    train_id_to_color = [c.color for c in Cityscapes.classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)

    # Create a full mapping table from original ID to train_id.
    # This handles all possible pixel values from 0-255.
    # Any ID not in Cityscapes.classes will remain mapped to 255 (ignore_index).
    _id_to_train_id_map = np.full(256, 255, dtype=np.uint8)
    for c in Cityscapes.classes:
        _id_to_train_id_map[c.id] = c.train_id
    id_to_train_id = _id_to_train_id_map

    def __init__(self, root, split='train', mode='gtFine', target_type='semantic', transform=None):
        self.root = Path(root).expanduser()
        self.mode = mode
        self.target_type = target_type
        self.images_dir = self.root / 'leftImg8bit' / split

        self.targets_dir = self.root / self.mode / split
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not self.images_dir.is_dir() or not self.targets_dir.is_dir():
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city_path in self.images_dir.iterdir():
            city = city_path.name
            img_dir = self.images_dir / city
            target_dir = self.targets_dir / city

            for file_name in img_dir.iterdir():
                self.images.append(img_dir / file_name)
                pure_img_name = re.sub(r'_leftImg8bit$', '', file_name.stem)
                target_name = f"{pure_img_name}_{self._get_target_suffix(self.mode, self.target_type)}"
                self.targets.append(target_dir / target_name)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        target = torch.from_numpy(target).long()
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return f'{mode}_instanceIds.png'
        elif target_type == 'semantic':
            return f'{mode}_labelIds.png'
        elif target_type == 'color':
            return f'{mode}_color.png'
        elif target_type == 'polygon':
            return f'{mode}_polygons.json'
        elif target_type == 'depth':
            return f'{mode}_disparity.png'