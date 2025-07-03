"""
Resolution configuration utilities for multi-resolution training.
"""

import numpy as np
from typing import Tuple, List

# Resolution 설정 정의
RESOLUTION_CONFIGS = {
    'max_high': {'image': np.array([432, 576]), 'heatmap': np.array([108, 144])},
    'sup_high': {'image': np.array([288, 384]), 'heatmap': np.array([72, 96])},
    'high': {'image': np.array([192, 256]), 'heatmap': np.array([48, 64])},
    'mid': {'image': np.array([144, 192]), 'heatmap': np.array([36, 48])},
    'low': {'image': np.array([96, 128]), 'heatmap': np.array([24, 32])},
    # 'sup_low': {'image': np.array([48, 64]), 'heatmap': np.array([12, 16])}
}

# Resolution 계층 정의 (상위에서 하위 순서)
RESOLUTION_HIERARCHY = ['max_high', 'sup_high', 'high', 'mid', 'low', 'sup_low']

# KD에서 사용할 multi-resolution 매핑 (각 resolution에서 어떤 하위 resolution들을 사용할지)
KD_RESOLUTION_MAPPING = {
    'max_high': ['max_high', 'sup_high', 'high', 'mid', 'low',],
    'sup_high': ['sup_high', 'high', 'mid', 'low'],
    'high': ['high', 'mid', 'low'],
    'mid': ['mid', 'low'], 
    'low': ['low'],
    # 'sup_low': ['sup_low'],
}

def setup_client_resolutions(client_res_list: List[str], use_kd: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    클라이언트별 resolution 설정을 생성합니다.
    
    Args:
        client_res_list (list): 각 클라이언트의 resolution 리스트
        use_kd (bool): Knowledge Distillation 사용 여부
        
    Returns:
        tuple: (image_sizes, heatmap_sizes) - 각 클라이언트별 설정 배열
        
    Examples:
        >>> client_resolutions = ['high', 'mid', 'low']
        >>> img_sizes, hm_sizes = setup_client_resolutions(client_resolutions, use_kd=True)
        >>> print(len(img_sizes))  # 3 (3개 클라이언트)
    """
    image_sizes = np.empty(len(client_res_list), dtype=object)
    heatmap_sizes = np.empty(len(client_res_list), dtype=object)
    
    for i, res in enumerate(client_res_list):
        img_size, hm_size = _get_multi_resolution_configs(res, use_kd)
        image_sizes[i] = img_size
        heatmap_sizes[i] = hm_size
    
    return image_sizes, heatmap_sizes

def _get_multi_resolution_configs(resolution: str, use_kd: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    주어진 resolution에 대해 multi-resolution 설정을 생성합니다.
    
    Args:
        resolution (str): 기준 resolution ('high', 'mid', 'low' 등)
        use_kd (bool): Knowledge Distillation 사용 여부
        
    Returns:
        tuple: (image_sizes, heatmap_sizes) numpy arrays
        
    Examples:
        >>> img_sizes, hm_sizes = _get_multi_resolution_configs('high', use_kd=True)
        >>> print(img_sizes.shape)  # (3, 2) - high, mid, low 각각의 [width, height]
        
        >>> img_sizes, hm_sizes = _get_multi_resolution_configs('low', use_kd=False)
        >>> print(img_sizes.shape)  # (2,) - [width, height]
    """
    if resolution not in RESOLUTION_CONFIGS:
        raise ValueError(f"Unknown resolution: {resolution}. Available: {list(RESOLUTION_CONFIGS.keys())}")
    
    if not use_kd:
        # KD를 사용하지 않으면 단일 resolution만 반환
        config = RESOLUTION_CONFIGS[resolution]
        return config['image'], config['heatmap']
    
    # KD 사용 시 매핑된 multi-resolution 사용
    target_resolutions = KD_RESOLUTION_MAPPING.get(resolution, [resolution])
    
    # 단일 resolution인 경우 numpy array로 직접 반환
    if len(target_resolutions) == 1:
        config = RESOLUTION_CONFIGS[target_resolutions[0]]
        return config['image'], config['heatmap']
    
    # Multi-resolution인 경우 리스트로 구성
    image_sizes = []
    heatmap_sizes = []
    
    for res in target_resolutions:
        config = RESOLUTION_CONFIGS[res]
        image_sizes.append(config['image'])
        heatmap_sizes.append(config['heatmap'])
    
    return np.array(image_sizes), np.array(heatmap_sizes)

def setup_single_client_resolution(resolution: str, use_kd: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 클라이언트(Centralized Learning)용 resolution 설정을 생성합니다.
    
    Args:
        resolution (str): 클라이언트의 resolution
        use_kd (bool): Knowledge Distillation 사용 여부
        
    Returns:
        tuple: (image_size, heatmap_size) numpy arrays
        
    Examples:
        >>> img_size, hm_size = setup_single_client_resolution('high', use_kd=True)
    """
    return _get_multi_resolution_configs(resolution, use_kd)

def get_available_resolutions() -> List[str]:
    """
    사용 가능한 모든 resolution 리스트를 반환합니다.
    
    Returns:
        list: 사용 가능한 resolution 문자열 리스트
    """
    return list(RESOLUTION_CONFIGS.keys())

def is_multi_resolution(image_size) -> bool:
    """
    주어진 image_size가 multi-resolution인지 확인합니다.
    
    Args:
        image_size: config.MODEL.IMAGE_SIZE 값
        
    Returns:
        bool: multi-resolution이면 True, 단일 resolution이면 False
    """
    if isinstance(image_size, np.ndarray) and len(image_size.shape) > 1:
        return True
    if isinstance(image_size, (list, tuple)) and len(image_size) > 0:
        first_element = image_size[0]
        return isinstance(first_element, (np.ndarray, list, tuple))
    return False 