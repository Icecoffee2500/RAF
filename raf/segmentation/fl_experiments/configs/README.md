# SegFormer Configuration Files

이 폴더에는 SegFormer 논문의 정확한 설정값을 적용한 config 파일들이 있습니다.

## 파일 설명

### 기본 설정
- **base.yaml**: 모든 설정의 기본이 되는 파일로, SegFormer 논문의 핵심 하이퍼파라미터를 포함

### 데이터셋별 설정
- **ade20k_config.yaml**: ADE20K 데이터셋용 설정 (512×512 해상도)
- **federated_config.yaml**: Federated Learning 실험용 설정

### 모델 변형
- **segformer_variants.yaml**: SegFormer-B0부터 B5까지의 모든 모델 변형과 성능 벤치마크

## SegFormer 논문 설정값 적용 내용

### 핵심 하이퍼파라미터
```yaml
# Optimizer (SegFormer 논문)
optimizer: AdamW
learning_rate: 6e-5
weight_decay: 0.01
betas: [0.9, 0.999]

# Scheduler
scheduler: polynomial_decay  
power: 1.0
warmup_iterations: 1500 (ADE20K) / 10 epochs (Cityscapes)

# Training
batch_size: 16 (일반), 8 (federated), 4 (B5 모델)
crop_size: [512, 512] (ADE20K), [512, 1024] (Cityscapes)
```

### 데이터 증강
```yaml
# SegFormer 논문 스타일 증강
random_scale: [0.5, 2.0]  # ADE20K
random_scale: [0.75, 1.25]  # Federated (안정성 위해 감소)
random_crop: true
random_flip: true
color_jitter: true
```

### 모델 구성
```yaml
# SegFormer 아키텍처
backbone: mit_b0  # 또는 mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
decode_head:
  channels: 256
  dropout: 0.1
  num_conv: 2
```

## 모델 크기별 특성

| 모델 | 파라미터 | FLOPs | Batch Size | ADE20K mIoU | Cityscapes mIoU |
|------|----------|-------|------------|-------------|-----------------|
| B0   | 3.8M     | 7.9G  | 16         | 37.4        | 76.2           |
| B1   | 13.7M    | 16.0G | 16         | 40.7        | 78.5           |
| B2   | 27.4M    | 62.4G | 16         | 45.3        | 81.0           |
| B3   | 47.3M    | 79.0G | 8          | 47.1        | 81.4           |
| B4   | 64.1M    | 95.7G | 8          | 48.0        | 81.4           |
| B5   | 84.7M    | 131.2G| 4          | 49.1        | 82.3           |

## Federated Learning 설정

### 핵심 설정
- **클라이언트 수**: 3개
- **라운드 수**: 50라운드  
- **로컬 에포크**: 5 에포크/라운드
- **집계 방법**: FedAvg (가중 평균)

### Knowledge Distillation
- **온도**: 4.0
- **알파**: 0.7 (distillation loss 가중치)
- **다해상도**: 지원 [1.0, 0.5, 0.25] 가중치

## 사용 방법

### 준비사항
wandb 로깅을 사용하려면 설치 필요:
```bash
pip install wandb
wandb login  # wandb 계정 로그인
```

### 중앙화 학습
```bash
# 기본 실험명 사용
python -m raf.segmentation.fl_experiments.main

# 커스텀 실험명
python -m raf.segmentation.fl_experiments.main exp_name=segformer_baseline
python -m raf.segmentation.fl_experiments.main exp_name=my_experiment device_id=1
```

### ADE20K 학습  
```bash
python -m raf.segmentation.fl_experiments.main exp_name=ade20k_test --config configs/ade20k_config.yaml
```

### Federated Learning
```bash
python -m raf.segmentation.fl_experiments.main exp_name=federated_experiment mode=federated
```

### 모델 변형 선택
base.yaml 또는 다른 config에서 model.backbone을 변경:
```yaml
model:
  backbone: mit_b2  # B0, B1, B2, B3, B4, B5 중 선택
```

### GPU 설정
config에서 사용할 GPU를 지정할 수 있습니다:
```yaml
# 특정 GPU 사용
device_id: 0        # GPU 0 사용
device_id: 1        # GPU 1 사용

# 디바이스 문자열 사용
device_id: "cuda:1" # GPU 1 사용
device_id: "cpu"    # CPU 강제 사용

# 자동 선택 (기본값)
# device_id 설정 안함 또는 null
```

### 터미널에서 GPU 직접 지정
```bash
# GPU 지정 (기본 실험명 사용)
python -m raf.segmentation.fl_experiments.main device_id=1
python -m raf.segmentation.fl_experiments.main device_id=2

# 실험 이름과 함께 GPU 지정
python -m raf.segmentation.fl_experiments.main exp_name=gpu1_test device_id=1
python -m raf.segmentation.fl_experiments.main exp_name=gpu2_test device_id=2

# 디바이스 문자열로 지정
python -m raf.segmentation.fl_experiments.main exp_name=cuda_test device_id=cuda:1
python -m raf.segmentation.fl_experiments.main device_id=cpu

# 다른 설정과 함께 사용
python -m raf.segmentation.fl_experiments.main exp_name=central_test device_id=1 mode=central

# Hydra 방식으로도 가능
python -m raf.segmentation.fl_experiments.main +device_id=1  # + 기호 사용
```

### 환경변수로 GPU 설정
```bash
# 특정 GPU만 보이게 설정
export CUDA_VISIBLE_DEVICES=1
python -m raf.segmentation.fl_experiments.main

# 여러 GPU 중 선택
export CUDA_VISIBLE_DEVICES=0,2
python -m raf.segmentation.fl_experiments.main  # GPU 0 사용됨
```

### GPU Fallback 동작
1. **지정한 GPU가 없는 경우**: GPU 0으로 자동 전환
2. **CUDA가 없는 경우**: CPU로 자동 전환
3. **상세한 정보 출력**: 어떤 GPU를 사용하는지 명확히 표시

### 모델 테스트
학습 완료 후 저장된 checkpoint로 성능 평가:
```bash
# Best checkpoint 테스트
python -m raf.segmentation.fl_experiments.test --checkpoint checkpoints/my_experiment/best.pth

# 특정 GPU에서 테스트
python -m raf.segmentation.fl_experiments.test --checkpoint checkpoints/my_experiment/best.pth --device cuda:1

# 다른 데이터셋으로 테스트
python -m raf.segmentation.fl_experiments.test --checkpoint checkpoints/my_experiment/best.pth --data_root /path/to/test_data
```

## 논문 대비 개선사항

1. **정확한 하이퍼파라미터**: SegFormer 논문의 정확한 설정값 적용
2. **모델 크기별 최적화**: 메모리에 따른 배치 크기 자동 조정
3. **Federated 최적화**: FL에 맞는 학습률 및 증강 강도 조정
4. **Knowledge Distillation**: 다해상도 증류 지원
5. **유연한 구성**: 데이터셋별, 실험별 구성 파일 분리
6. **Checkpoint 관리**: Best mIoU 자동 저장 및 복원
7. **Reproducibility**: 고정된 시드로 동일한 결과 보장
8. **상세한 로깅**: Wandb 연동 및 진행상황 추적

## 참고 논문
- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (NeurIPS 2021)
- Original paper settings을 기반으로 구성됨 