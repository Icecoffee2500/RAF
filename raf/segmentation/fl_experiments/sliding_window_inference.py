import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple

# 가정: SegFormer 모델 로드 (논문 GitHub의 init_segmentor처럼)
# model = YourSegFormerModel()  # pretrained weights 로드
# model.eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

def resize_image(image: np.ndarray, short_side: int = 1024) -> np.ndarray:
    """Short side를 1024로 rescale, aspect ratio 유지 (논문 eval 방식)."""
    h, w = image.shape[:2]
    if h < w:
        new_h = short_side
        new_w = int(w * (short_side / h))
    else:
        new_w = short_side
        new_h = int(h * (short_side / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def sliding_window_inference(model, image: np.ndarray, window_size: Tuple[int, int] = (1024, 1024),
                            stride: int = 512, num_classes: int = 19, device: torch.device = torch.device('cuda')) -> np.ndarray:
    """
    Sliding window inference 구현.
    - image: 입력 이미지 (H, W, 3), BGR or RGB.
    - window_size: (1024, 1024) for Cityscapes.
    - stride: overlap을 위한 stride (e.g., 512 for 50% overlap).
    - num_classes: Cityscapes=19.
    반환: 전체 이미지의 세그멘테이션 마스크 (H, W).
    """
    # 1. 이미지 rescale (short side=1024)
    image = resize_image(image)
    orig_h, orig_w = image.shape[:2]
    
    # 2. Padding: 윈도우가 딱 맞게 (multiple of window_size)
    pad_h = (window_size[0] - orig_h % stride) % stride
    pad_w = (window_size[1] - orig_w % stride) % stride
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    pad_h_full, pad_w_full = padded_image.shape[:2]  # padded 크기
    
    # 3. Normalize (ImageNet mean/std 예시, 모델에 맞게 조정)
    image_tensor = torch.from_numpy(padded_image.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.to(device)
    
    # 4. Sliding windows 생성 및 inference
    count_map = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)  # overlap count
    pred_map = np.zeros((num_classes, pad_h_full, pad_w_full), dtype=np.float32)  # 누적 예측
    
    for y in range(0, pad_h_full - window_size[0] + 1, stride):
        for x in range(0, pad_w_full - window_size[1] + 1, stride):
            # 윈도우 추출
            window = image_tensor[:, y:y+window_size[0], x:x+window_size[1]].unsqueeze(0)  # [1, 3, 1024, 1024]
            
            # Inference
            with torch.no_grad():
                output = model(window)  # 가정: output = [1, num_classes, 1024, 1024]
                output = F.softmax(output, dim=1).squeeze(0).cpu().numpy()  # [num_classes, 1024, 1024]
            
            # 누적
            pred_map[:, y:y+window_size[0], x:x+window_size[1]] += output
            count_map[y:y+window_size[0], x:x+window_size[1]] += 1
    
    # 5. 병합: 평균
    pred_map /= count_map[None, :, :] + 1e-6  # avoid div by zero
    pred_map = pred_map[:, :orig_h, :orig_w]  # 원본 크기로 crop
    
    # 6. Argmax로 클래스 맵 생성
    final_mask = np.argmax(pred_map, axis=0)  # [H, W]
    return final_mask

# 사용 예시
image = cv2.imread('path/to/cityscapes/image.jpg')  # [H, W, 3]
mask = sliding_window_inference(model, image)

# 평가: mask로 mIoU 계산 (별도 구현 필요, e.g., Cityscapes GT와 비교)