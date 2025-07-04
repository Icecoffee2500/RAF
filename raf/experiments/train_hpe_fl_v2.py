#!/usr/bin/env python3
"""
리팩토링된 연합학습 HPE 훈련 스크립트 (Hydra + Component Architecture)

사용법:
    python train_hpe_fl_v2.py experiment=mpii_fedavg_multi
    python train_hpe_fl_v2.py experiment=coco_centralized_single
    python train_hpe_fl_v2.py experiment=mpii_fedavg_multi federated.num_clients=5
"""

import sys
import os
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# 프로젝트 루트 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from hpe.models import ViT, TopdownHeatmapSimpleHead, ViTPose
from hpe.utils.random_utils import init_random_seed, set_random_seed
from hpe.utils.checkpoint_utils import load_checkpoint
from hpe.utils.resolution_utils import setup_client_resolutions
from hpe.federated_v2.orchestrator.fl_orchestrator import FLOrchestrator


def create_global_model(cfg: DictConfig) -> torch.nn.Module:
    """글로벌 모델 생성"""
    print("=== 글로벌 모델 생성 ===")
    
    # Backbone 생성
    backbone = ViT(
        img_size=cfg.model.backbone.img_size,
        patch_size=cfg.model.backbone.patch_size,
        embed_dim=cfg.model.backbone.embed_dim,
        # in_channels=3,
        in_channels=cfg.model.backbone.in_channels,
        num_heads=cfg.model.backbone.num_heads,
        depth=cfg.model.backbone.depth,
        # qkv_bias=True,
        qkv_bias=cfg.model.backbone.qkv_bias,
        drop_path_rate=cfg.model.backbone.drop_path_rate,
        use_gpe=cfg.model.use_gpe,
        use_lpe=cfg.model.use_lpe,
    )
    
    # OmegaConf DictConfig ➔ plain dict 변환 (타입 체크 통과)
    extra_cfg = cfg.model.keypoint_head.extra
    if extra_cfg is not None:
        try:
            from omegaconf import OmegaConf
            extra_cfg = OmegaConf.to_container(extra_cfg, resolve=True)
        except Exception:
            # fallback: DictConfig 는 Mapping protocol 지원
            extra_cfg = {k: extra_cfg[k] for k in extra_cfg.keys()}

    # Keypoint Head 생성
    keypoint_head = TopdownHeatmapSimpleHead(
        in_channels=cfg.model.keypoint_head.in_channels,
        num_deconv_layers=cfg.model.keypoint_head.num_deconv_layers,
        num_deconv_filters=cfg.model.keypoint_head.num_deconv_filters,
        num_deconv_kernels=cfg.model.keypoint_head.num_deconv_kernels,
        # extra=dict(final_conv_kernel=1),
        extra=extra_cfg,
        out_channels=cfg.model.keypoint_head.out_channels,
    )
    
    # 전체 모델 조합
    global_model = ViTPose(backbone, keypoint_head)
    
    # 사전 훈련된 가중치 로드
    if cfg.get('pretrained_path'):
        pretrained_path = project_root / cfg.pretrained_path
        if pretrained_path.exists() and "mae" in str(pretrained_path):
            print(f"사전 훈련된 가중치 로드: {pretrained_path}")
            load_checkpoint(global_model, pretrained_path)
        else:
            print(f"사전 훈련된 가중치 파일을 찾을 수 없음: {pretrained_path}")
    
    return global_model


def setup_resolution_config(cfg: DictConfig) -> DictConfig:
    """해상도 설정 구성"""
    print("=== 해상도 설정 구성 ===")
    
    # 클라이언트별 해상도 설정
    if cfg.federated.method != "centralized":
        # 연합학습인 경우 클라이언트별 해상도 할당
        client_resolutions = []
        if cfg.resolution.type == "multi":
            # 다중 해상도인 경우 각 클라이언트에 서로 다른 해상도 할당
            base_resolutions = cfg.resolution.resolutions
            for i in range(cfg.federated.num_clients):
                client_resolutions.append(base_resolutions[i % len(base_resolutions)])
        else:
            # 단일 해상도인 경우 모든 클라이언트가 같은 해상도 사용
            single_res = cfg.resolution.resolutions[0]
            client_resolutions = [single_res] * cfg.federated.num_clients
        
        # Knowledge Distillation 사용 여부 결정
        kd_enabled = cfg.resolution.type == "multi"
        
        # numpy.ndarray → python list 로 변환하여 Hydra config 호환성 확보
        image_sizes_np, heatmap_sizes_np = setup_client_resolutions(client_resolutions, kd_enabled)
        
        # object 배열일 수 있으므로 각각 안전하게 tolist 수행
        image_sizes = [(
            img.tolist() if hasattr(img, "tolist") else img
        ) for img in image_sizes_np]
        
        heatmap_sizes = [(
            hm.tolist() if hasattr(hm, "tolist") else hm
        ) for hm in heatmap_sizes_np]
        
    else:
        # 중앙 집중식인 경우 단일 해상도 사용
        single_res = cfg.resolution.resolutions[0]
        image_sizes = [single_res]
        heatmap_sizes = [[int(single_res[0] // 4), int(single_res[1] // 4)]]
        kd_enabled = False
    
    # 설정 업데이트
    cfg.resolution.image_size = image_sizes
    cfg.resolution.heatmap_size = heatmap_sizes
    cfg.loss.kd.enabled = kd_enabled
    
    print(f"클라이언트 수: {cfg.federated.num_clients}")
    print(f"이미지 크기: {cfg.resolution.image_size}")
    print(f"히트맵 크기: {cfg.resolution.heatmap_size}")
    print(f"KD 활성화: {cfg.loss.kd.enabled}")
    
    return cfg


@hydra.main(version_base=None, config_path="../configs_hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 함수"""
    print("="*80)
    print("연합학습 HPE 훈련 시작 (리팩토링된 아키텍처)")
    print("="*80)
    
    # 설정 출력
    print("실행 설정:")
    print(OmegaConf.to_yaml(cfg))
    
    # 랜덤 시드 설정
    seed = init_random_seed(cfg.get('seed', 42))
    print(f"랜덤 시드 설정: {seed}")
    set_random_seed(seed)
    
    # 디바이스 설정
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"디바이스: {device}")
    
    # 해상도 설정 구성
    cfg = setup_resolution_config(cfg)
    
    # 글로벌 모델 생성
    global_model = create_global_model(cfg)
    global_model.to(device)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"글로벌 모델 파라미터 수: {total_params:,}")
    
    # 연합학습 오케스트레이터 생성
    print("\n=== 연합학습 오케스트레이터 초기화 ===")
    orchestrator = FLOrchestrator(
        cfg=cfg,
        global_model=global_model,
        device=device
    )
    
    try:
        # 연합학습 실행
        print("\n=== 연합학습 실행 ===")
        results = orchestrator.run_federated_learning()
        
        # 최종 결과 출력
        print("\n" + "="*80)
        print("연합학습 완료!")
        print("="*80)
        print(f"베스트 성능: {results['best_performance']:.4f}")
        print(f"최종 성능: {results['final_performance']:.4f}")
        print(f"총 소요시간: {results['total_time']}")
        print(f"성능 히스토리: {orchestrator.get_performance_history()}")
        
        return results['best_performance']
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
        return 0.0
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


if __name__ == "__main__":
    main() 