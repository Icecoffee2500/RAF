from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from omegaconf import DictConfig, ListConfig
from typing import Tuple
import numpy as np

from hpe.dataset.utils.builder import build_train_val_dataloader, build_split_dataset
from hpe.dataset.coco import COCODataset as coco
from hpe.dataset.mpii import MPIIDataset as mpii


class DataManager:
    """데이터 로딩 및 관리 전담 클래스"""
    
    def __init__(self, 
                 dataset_cfg: DictConfig,
                 client_id: int,
                 image_size: list,
                 heatmap_size: list,
                 train_batch_size: int,
                 valid_batch_size: int,
                 samples_per_split: int = 0,
                 workers: int = 4):
        """
        Args:
            dataset_cfg: 데이터셋 설정
            client_id: 클라이언트 ID (데이터 분할용)
            image_size: 이미지 크기
            heatmap_size: 히트맵 크기  
            batch_size: 배치 크기
            samples_per_split: 클라이언트별 샘플 수 (0이면 전체 사용)
            workers: 데이터로더 워커 수
        """
        self.dataset_cfg = dataset_cfg
        self.client_id = client_id
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.samples_per_split = samples_per_split
        self.workers = workers
        
        # ListConfig ➔ list / numpy array 변환 (Hydra 호환성 문제 해결)
        def _standardize(size_obj, *, dtype=np.int32):
            """ListConfig → list → np.ndarray (int) 변환"""
            if isinstance(size_obj, ListConfig):
                size_obj = list(size_obj)
            # 다중 해상도 여부 판단
            if isinstance(size_obj, (list, tuple)) and len(size_obj) > 0 and isinstance(size_obj[0], (list, tuple, ListConfig)):
                # 리스트-오브-리스트: 각 요소를 numpy array 로 변환
                _out = []
                for elem in size_obj:
                    if isinstance(elem, ListConfig):
                        elem = list(elem)
                    _out.append(np.asarray(elem, dtype=dtype))
                return _out
            else:
                # 단일 해상도
                return np.asarray(size_obj, dtype=dtype)

        self.image_size = _standardize(self.image_size, dtype=np.int32)
        self.heatmap_size = _standardize(self.heatmap_size, dtype=np.int32)
        
        # 데이터셋과 데이터로더 생성
        self.train_dataset, self.valid_dataset = self._create_datasets()
        self.train_loader, self.valid_loader = self._create_dataloaders()
    
    def _get_dataset_class(self):
        """데이터셋 클래스 반환"""
        if self.dataset_cfg.name == 'coco':
            return coco
        elif self.dataset_cfg.name == 'mpii':
            return mpii
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {self.dataset_cfg.name}")
    
    def _get_transforms(self):
        """데이터 변환 파이프라인 반환"""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        return transforms.Compose([transforms.ToTensor(), normalize])
    
    def _create_datasets(self) -> Tuple:
        """훈련 및 검증 데이터셋 생성"""
        dataset_class = self._get_dataset_class()
        transform = self._get_transforms()
        
        # 훈련 데이터셋 생성
        train_dataset = dataset_class(
            cfg=self._create_legacy_config(),  # 임시: legacy config 생성
            root=self.dataset_cfg.root,
            image_set=self.dataset_cfg.train_set,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            is_train=True,
            transform=transform,
        )
        
        # 클라이언트별 데이터 분할
        if self.samples_per_split > 0:
            train_dataset = build_split_dataset(
                train_dataset, 
                dataset_idx=self.client_id, 
                samples_per_split=self.samples_per_split
            )
        
        # 검증 데이터셋 생성 (단일 해상도 사용)
        valid_image_size = self.image_size[0] if isinstance(self.image_size[0], (list, tuple)) else self.image_size
        valid_heatmap_size = self.heatmap_size[0] if isinstance(self.heatmap_size[0], (list, tuple)) else self.heatmap_size
        
        valid_dataset = dataset_class(
            cfg=self._create_legacy_config(),
            root=self.dataset_cfg.root,
            image_set=self.dataset_cfg.test_set,
            image_size=valid_image_size,
            heatmap_size=valid_heatmap_size,
            is_train=False,
            transform=transform,
        )
        
        return train_dataset, valid_dataset
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """데이터로더 생성"""
        train_loader, valid_loader = build_train_val_dataloader(
            self.train_dataset, 
            self.valid_dataset,
            [self.train_batch_size, self.valid_batch_size],  # [train_bs, valid_bs]
            self.workers
        )
        return train_loader, valid_loader
    
    def _create_legacy_config(self):
        """임시: 기존 데이터셋 클래스가 요구하는 legacy config 생성"""
        from easydict import EasyDict as edict
        
        config = edict()
        config.DATASET = edict()
        config.DATASET.DATASET = self.dataset_cfg.name
        config.DATASET.FLIP = self.dataset_cfg.flip
        config.DATASET.ROT_FACTOR = self.dataset_cfg.rot_factor
        config.DATASET.SCALE_FACTOR = self.dataset_cfg.scale_factor
        config.DATASET.TARGET_HEATMAP = self.dataset_cfg.target_heatmap
        config.DATASET.NUM_JOINTS_HALF_BODY = self.dataset_cfg.num_joints_half_body
        config.DATASET.PROB_HALF_BODY = self.dataset_cfg.prob_half_body
        config.DATASET.SHIFT_FACTOR = self.dataset_cfg.shift_factor
        config.DATASET.SHIFT_PROB = self.dataset_cfg.shift_prob
        config.DATASET.SELECT_DATA = self.dataset_cfg.select_data
        
        config.TEST = edict()
        config.TEST.USE_UDP = self.dataset_cfg.use_udp
        
        config.MODEL = edict()
        config.MODEL.NUM_JOINTS = self.dataset_cfg.num_joints

        config.MODEL.EXTRA = edict()
        config.MODEL.EXTRA.SIGMA = self.dataset_cfg.sigma
        config.MODEL.EXTRA.HEATMAP_TYPE = self.dataset_cfg.heatmap_type

        
        return config
    
    def get_train_loader(self) -> DataLoader:
        """훈련 데이터로더 반환"""
        return self.train_loader
    
    def get_valid_loader(self) -> DataLoader:
        """검증 데이터로더 반환"""
        return self.valid_loader
    
    def get_dataset_info(self) -> dict:
        """데이터셋 정보 반환"""
        return {
            "train_size": len(self.train_dataset),
            "valid_size": len(self.valid_dataset),
            "num_joints": self.dataset_cfg.num_joints,
            "dataset_name": self.dataset_cfg.name,
            "client_id": self.client_id,
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size
        } 