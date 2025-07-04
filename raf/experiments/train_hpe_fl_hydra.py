import sys
import os
from pathlib import Path
import torch
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict as edict
from hydra.core.config_store import ConfigStore

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Schema 등록
from configs_hydra.schema import register_configs
register_configs()

from hpe.utils.logging import ShellColors as sc
from hpe.utils.logging import create_logger_sfl
from hpe.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from hpe.utils.random_utils import init_random_seed, set_random_seed
from hpe.utils.misc_utils import show_info
from hpe.utils.resolution_utils import setup_client_resolutions, is_multi_resolution
from federated.server import FedServer
from hpe.federated.client import FLClient
from hpe.models import ViT
from hpe.models import TopdownHeatmapSimpleHead
from hpe.models import ViTPose

def convert_hydra_to_legacy_config(cfg: DictConfig):
    """Hydra config를 기존 FLClient가 이해할 수 있는 legacy config로 변환"""
    
    # EasyDict 기반 legacy config 생성
    legacy_config = edict()
    
    # 기본 설정
    legacy_config.OUTPUT_DIR = cfg.output_dir
    legacy_config.LOG_DIR = cfg.log_dir
    legacy_config.WORKERS = cfg.workers
    legacy_config.PRINT_FREQ = cfg.print_freq
    
    # DATASET 설정
    legacy_config.DATASET = edict()
    legacy_config.DATASET.DATASET = cfg.dataset.name
    legacy_config.DATASET.ROOT = cfg.dataset.root
    legacy_config.DATASET.TRAIN_SET = cfg.dataset.train_set
    legacy_config.DATASET.TEST_SET = cfg.dataset.test_set
    legacy_config.DATASET.FLIP = cfg.dataset.flip
    legacy_config.DATASET.ROT_FACTOR = cfg.dataset.rot_factor
    legacy_config.DATASET.SCALE_FACTOR = cfg.dataset.scale_factor
    legacy_config.DATASET.TARGET_HEATMAP = cfg.dataset.target_heatmap
    legacy_config.DATASET.NUM_JOINTS_HALF_BODY = cfg.dataset.num_joints_half_body
    legacy_config.DATASET.PROB_HALF_BODY = cfg.dataset.prob_half_body
    
    # MODEL 설정
    legacy_config.MODEL = edict()
    legacy_config.MODEL.NAME = cfg.model.name
    legacy_config.MODEL.TYPE = cfg.model.type
    legacy_config.MODEL.NUM_JOINTS = cfg.model.num_joints if hasattr(cfg.model, 'num_joints') else cfg.dataset.num_joints
    legacy_config.MODEL.IMAGE_SIZE = cfg.model.image_size
    legacy_config.MODEL.HEATMAP_SIZE = cfg.model.heatmap_size
    legacy_config.MODEL.USE_GPE = cfg.model.use_gpe
    legacy_config.MODEL.USE_LPE = cfg.model.use_lpe
    legacy_config.MODEL.USE_GAP = cfg.model.use_gap
    
    # TRAIN 설정
    legacy_config.TRAIN = edict()
    legacy_config.TRAIN.BATCH_SIZE = cfg.training.batch_size
    legacy_config.TRAIN.LR = cfg.training.lr
    legacy_config.TRAIN.BEGIN_EPOCH = cfg.training.begin_epoch
    legacy_config.TRAIN.END_EPOCH = cfg.training.end_epoch
    legacy_config.TRAIN.LR_STEP = cfg.training.lr_step
    legacy_config.TRAIN.LR_FACTOR = cfg.training.lr_factor
    legacy_config.TRAIN.OPTIMIZER = cfg.training.optimizer
    legacy_config.TRAIN.SHUFFLE = cfg.training.shuffle
    
    # TEST 설정
    legacy_config.TEST = edict()
    legacy_config.TEST.BATCH_SIZE = cfg.test.batch_size if hasattr(cfg, 'test') else 32
    legacy_config.TEST.FLIP_TEST = cfg.test.flip_test if hasattr(cfg, 'test') else True
    legacy_config.TEST.USE_GT_BBOX = cfg.test.use_gt_bbox if hasattr(cfg, 'test') else False
    legacy_config.TEST.USE_UDP = cfg.test.use_udp if hasattr(cfg, 'test') else True
    
    # LOSS 설정
    legacy_config.LOSS = edict()
    legacy_config.LOSS.HM_LOSS = cfg.loss.hm_loss
    legacy_config.LOSS.HM_LOSS_WEIGHT = cfg.loss.hm_loss_weight
    legacy_config.LOSS.USE_TARGET_WEIGHT = cfg.loss.use_target_weight
    
    # FED 설정
    legacy_config.FED = edict()
    legacy_config.FED.FEDAVG = cfg.federated.method == "fedavg"
    legacy_config.FED.FEDPROX = cfg.federated.method == "fedprox"
    legacy_config.FED.MU = cfg.federated.mu
    
    # KD 설정
    if hasattr(cfg.federated, 'kd'):
        legacy_config.KD_USE = cfg.federated.kd.enabled
        legacy_config.KD_ALPHA = cfg.federated.kd.alpha
        legacy_config.LOSS_SCALE = cfg.federated.kd.loss_scale
    else:
        legacy_config.KD_USE = False
        legacy_config.KD_ALPHA = 0.0
        legacy_config.LOSS_SCALE = 1.0
    
    # EVALUATION 설정
    legacy_config.EVALUATION = edict()
    legacy_config.EVALUATION.INTERVAL = cfg.evaluation.interval if hasattr(cfg, 'evaluation') else 10
    
    return legacy_config

def setup_wandb(cfg: DictConfig):
    """WandB 초기화"""
    if not cfg.wandb:
        return None
        
    try:
        import wandb
        now = datetime.now()
        today = now.strftime("%m%d_%H:%M")
        
        name = f"hydra-{cfg.federated.method}-{cfg.model.type}-{cfg.dataset.name}-{today}"
        
        wandb_config = {
            "exp_name": cfg.exp_name,
            "model_name": cfg.model.name,
            "model_type": cfg.model.type,
            "dataset": cfg.dataset.name,
            "fed_method": cfg.federated.method,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
        }
        
        wandb.init(
            project="RAF-hydra-refactoring",
            name=name,
            config=wandb_config,
            mode="online",
        )
        print(f"{sc.COLOR_GREEN}WandB 초기화 성공{sc.ENDC}")
        return wandb
        
    except Exception as e:
        print(f"{sc.COLOR_RED}WandB 초기화 실패: {e}{sc.ENDC}")
        return None

def create_model(cfg: DictConfig):
    """모델 생성"""
    # Backbone
    backbone = ViT(
        img_size=cfg.model.backbone.img_size,
        patch_size=cfg.model.backbone.patch_size,
        embed_dim=cfg.model.backbone.embed_dim,
        in_channels=3,
        num_heads=cfg.model.backbone.num_heads,
        depth=cfg.model.backbone.depth,
        qkv_bias=True,
        drop_path_rate=cfg.model.backbone.drop_path_rate,
        use_gpe=cfg.model.use_gpe,
        use_lpe=cfg.model.use_lpe,
        use_gap=cfg.model.use_gap,
    )
    
    # Head
    deconv_head = TopdownHeatmapSimpleHead(
        in_channels=cfg.model.keypoint_head.in_channels,
        num_deconv_layers=cfg.model.keypoint_head.num_deconv_layers,
        num_deconv_filters=cfg.model.keypoint_head.num_deconv_filters,
        num_deconv_kernels=cfg.model.keypoint_head.num_deconv_kernels,
        extra=dict(final_conv_kernel=1),
        out_channels=cfg.model.keypoint_head.out_channels,
    )
    
    return ViTPose(backbone, deconv_head)

def create_clients(cfg: DictConfig, global_model, device, wdb, logger, legacy_config):
    """FL 클라이언트들 생성"""
    clients = []
    
    # Multi-resolution 처리
    if isinstance(cfg.model.image_size[0], (list, tuple)):
        image_sizes = cfg.model.image_size
        heatmap_sizes = cfg.model.heatmap_size
    else:
        image_sizes = [cfg.model.image_size] * cfg.federated.num_clients
        heatmap_sizes = [cfg.model.heatmap_size] * cfg.federated.num_clients
    
    # Extra config 생성 (FLClient가 요구하는 형식)
    extra = {
        "backbone": OmegaConf.to_container(cfg.model.backbone, resolve=True),
        "keypoint_head": OmegaConf.to_container(cfg.model.keypoint_head, resolve=True)
    }
    
    for idx in range(cfg.federated.num_clients):
        client = FLClient(
            client_id=idx,
            config=legacy_config,  # 변환된 legacy config 사용
            device=device,
            init_model=global_model,
            extra=extra,
            wdb=wdb,
            logger=logger,
            im_size=image_sizes[idx] if idx < len(image_sizes) else image_sizes[0],
            hm_size=heatmap_sizes[idx] if idx < len(heatmap_sizes) else heatmap_sizes[0],
            batch_size=cfg.training.batch_size,
            is_proxy=False,
            samples_per_split=1000,
        )
        clients.append(client)
    
    return clients

@hydra.main(version_base=None, config_path="../configs_hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra 기반 Federated Learning 메인 함수"""
    
    print(f"{sc.COLOR_CYAN}=== Hydra Federated Learning 시작 ==={sc.ENDC}")
    print(f"실험명: {cfg.exp_name}")
    print(f"데이터셋: {cfg.dataset.name} (joints: {cfg.dataset.num_joints})")
    print(f"모델: {cfg.model.name}-{cfg.model.type}")
    print(f"연합학습: {cfg.federated.method} (clients: {cfg.federated.num_clients})")
    print(f"해상도: {cfg.model.image_size}")
    
    # 시드 설정
    seed = init_random_seed(cfg.seed)
    set_random_seed(seed)
    
    # Legacy config 변환
    legacy_config = convert_hydra_to_legacy_config(cfg)
    
    # WandB 초기화
    wdb = setup_wandb(cfg)
    
    # 로거 생성
    logger, final_output_dir = create_logger_sfl(
        legacy_config, f"{cfg.exp_name}.yaml", f"train_{cfg.gpu}", arg=cfg.exp_name
    )
    
    # 디바이스 설정
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # 모델 생성
    global_fl_model = create_model(cfg)
    
    # 사전훈련 모델 로딩
    if cfg.model.pretrained and "mae" in cfg.model.pretrained:
        pretrained_path = project_root / cfg.model.pretrained
        if pretrained_path.exists():
            load_checkpoint(global_fl_model, pretrained_path)
            print(f"{sc.COLOR_GREEN}사전훈련 모델 로딩: {pretrained_path}{sc.ENDC}")
        else:
            print(f"{sc.COLOR_YELLOW}사전훈련 모델 파일이 없음: {pretrained_path}{sc.ENDC}")
    
    global_fl_model.to(device)
    
    # 연합학습 실행
    if cfg.federated.enabled:
        print(f"{sc.COLOR_BLUE}연합학습 모드 실행{sc.ENDC}")
        fl_clients = create_clients(cfg, global_fl_model, device, wdb, logger, legacy_config)
        fed_server = FedServer()
        run_federated_training(cfg, fl_clients, fed_server, global_fl_model, logger, wdb, final_output_dir)
    else:
        print(f"{sc.COLOR_BLUE}중앙집중식 훈련 모드 (구현 예정){sc.ENDC}")

def run_federated_training(cfg, fl_clients, fed_server, global_fl_model, logger, wdb, final_output_dir):
    """연합학습 훈련 실행"""
    avg_perf_buf = [0.0]
    
    for epoch in range(cfg.training.begin_epoch, cfg.training.end_epoch):
        print(f"\n{sc.COLOR_CYAN}=== Epoch {epoch}/{cfg.training.end_epoch} ==={sc.ENDC}")
        init_time = datetime.now()
        
        # Scheduler step
        for client in fl_clients:
            client.lr_scheduler.step()
        
        # 클라이언트 훈련
        client_weights = []
        for idx, client in enumerate(fl_clients):
            if is_multi_resolution(cfg.model.image_size if isinstance(cfg.model.image_size[0], (list, tuple)) else [cfg.model.image_size]):
                print(f">>> Client [{idx}] Multi-res (KD) Training")
                client.train_multi_resolution(epoch)
            else:
                print(f">>> Client [{idx}] Single-res Training") 
                client.train_single_resolution(epoch)
            
            client_weights.append(client.model.state_dict())
        
        # 가중치 집계 및 브로드캐스트
        w_glob_client = fed_server.aggregate(logger, client_weights)
        for client in fl_clients:
            client.model.load_state_dict(w_glob_client)
        global_fl_model.load_state_dict(w_glob_client)
        
        epoch_time = datetime.now() - init_time
        logger.info(f"Epoch {epoch} 완료, 소요시간: {epoch_time}")
        
        # 평가
        if epoch % cfg.evaluation.interval == 0:
            curr_avg_perf = 0
            for client_idx, client in enumerate(fl_clients):
                perf = client.evaluate(final_output_dir, wdb)
                curr_avg_perf += perf / len(fl_clients)
            
            if curr_avg_perf > max(avg_perf_buf):
                logger.info(f"새로운 최고 성능: {curr_avg_perf:.4f}")
                avg_perf_buf.append(curr_avg_perf)
                
                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": global_fl_model.state_dict(),
                    "perf": curr_avg_perf,
                    "config": cfg,
                }, final_output_dir)

if __name__ == "__main__":
    main() 