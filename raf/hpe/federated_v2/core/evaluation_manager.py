import torch
import time
import numpy as np
from datetime import datetime
from omegaconf import DictConfig
from typing import Dict, Any

from hpe.utils.training_utils import AverageMeter
from hpe.utils.evaluate import accuracy
from hpe.utils.post_processing import get_final_preds
from hpe.utils.logging import ShellColors as sc


class EvaluationManager:
    """평가 로직 전담 클래스"""
    
    def __init__(self, 
                 dataset_cfg: DictConfig,
                 test_cfg: DictConfig,
                 device: torch.device):
        """
        Args:
            dataset_cfg: 데이터셋 설정
            test_cfg: 테스트 설정  
            device: 디바이스
        """
        self.dataset_cfg = dataset_cfg
        self.test_cfg = test_cfg
        self.device = device
        
        # 메트릭 추적용
        self.losses = AverageMeter()
        self.acc = AverageMeter()
    
    def evaluate(self,
                model_manager,
                data_manager,
                final_output_dir: str,
                wdb=None,
                logger=None) -> float:
        """모델 평가 실행
        
        Returns:
            float: 성능 지표 (mAP 등)
        """
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        
        model = model_manager.get_model()
        criterion = model_manager.get_criterion()
        valid_loader = data_manager.get_valid_loader()
        valid_dataset = data_manager.valid_dataset
        
        model.eval()
        
        num_samples = len(valid_dataset)
        all_preds = np.zeros((num_samples, self.dataset_cfg.num_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        bbox_ids = []
        img_idx = 0
        
        valid_loader_len = len(valid_loader)
        
        with torch.no_grad():
            end = time.time()
            epoch_start_time = datetime.now()
            
            for batch_idx, (img, heatmap, heatmap_weight, meta) in enumerate(valid_loader):
                img = img.to(self.device)
                heatmap = heatmap.to(self.device) 
                heatmap_weight = heatmap_weight.to(self.device)
                
                # Forward pass
                output = model(img)
                
                # Flip Test (선택적)
                if hasattr(self.test_cfg, 'flip_test') and self.test_cfg.flip_test:
                    output_heatmap = self._apply_flip_test(model, img, output, meta)
                else:
                    output_heatmap = output
                
                # 손실 계산
                loss = self._calculate_loss(criterion, output_heatmap, heatmap, heatmap_weight)
                
                num_images = img.size(0)
                
                # 정확도 계산
                _, avg_acc, cnt, _ = accuracy(
                    output_heatmap.detach().cpu().numpy(),
                    heatmap.detach().cpu().numpy()
                )
                
                # 메트릭 업데이트
                self.acc.update(avg_acc, cnt)
                self.losses.update(loss.item(), num_images)
                
                # 배치 시간 측정
                batch_time.update(time.time() - end)
                end = time.time()
                
                # 최종 예측 계산
                c = meta["center"].numpy()
                s = meta["scale"].numpy()
                score = meta["score"].numpy()
                
                preds, maxvals = get_final_preds(
                    self._create_legacy_config(), 
                    output_heatmap.clone().cpu().numpy(), 
                    c, s, 11
                )
                
                # 결과 저장
                all_preds[img_idx : img_idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[img_idx : img_idx + num_images, :, 2:3] = maxvals
                
                all_boxes[img_idx : img_idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[img_idx : img_idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[img_idx : img_idx + num_images, 4] = np.prod(s * 200, 1)
                all_boxes[img_idx : img_idx + num_images, 5] = score
                
                image_path.extend(meta["image"])
                bbox_ids.extend(meta["bbox_id"])
                
                img_idx += num_images
                
                # 로깅
                if batch_idx % 100 == 0 and logger:  # print_freq 대신 고정값 사용
                    self._log_evaluation_progress(
                        batch_idx, valid_loader_len, batch_time,
                        num_images, epoch_start_time, logger
                    )
            
            # 최종 평가
            perf_indicator = 0
            
            try:
                name_values, perf_indicator = valid_dataset.evaluate(
                    self._create_legacy_config(),
                    all_preds, 
                    final_output_dir, 
                    all_boxes, 
                    image_path, 
                    bbox_ids
                )
                
                # WandB 로깅
                if wdb:
                    wdb.log({
                        "performance": perf_indicator,
                        "loss_valid": self.losses.avg,
                        "acc_valid": self.acc.avg
                    })
                
                # 결과 출력
                if logger:
                    self._print_evaluation_results(name_values, logger)
                    logger.info(f"평가 완료, 소요시간: {datetime.now() - epoch_start_time}")
                
            except Exception as e:
                if logger:
                    logger.error(f"평가 중 오류 발생: {e}")
                perf_indicator = 0.0
        
        return perf_indicator
    
    def _apply_flip_test(self, model, img, output, meta):
        """Flip Test 적용"""
        img_flipped = img.flip(3)
        features_flipped = model.backbone(img_flipped)
        
        output_flipped_heatmap = model.keypoint_head.inference_model(
            features_flipped, meta["flip_pairs"]
        )
        
        output_heatmap = (
            output + torch.from_numpy(output_flipped_heatmap.copy()).to(self.device)
        ) * 0.5
        
        return output_heatmap
    
    def _calculate_loss(self, criterion, pred_heatmap, gt_heatmap, hm_weight):
        """손실 계산"""
        loss = 0
        for k, v in criterion.items():
            if k == "heatmap":
                l = v(pred_heatmap, gt_heatmap, hm_weight)
                loss += l  # weight는 이미 criterion에 포함됨
        return loss
    
    def _create_legacy_config(self):
        """임시: 기존 평가 함수가 요구하는 legacy config 생성"""
        from easydict import EasyDict as edict
        
        config = edict()
        config.MODEL = edict()
        config.MODEL.NUM_JOINTS = self.dataset_cfg.num_joints
        
        config.TEST = edict()
        config.TEST.FLIP_TEST = getattr(self.test_cfg, 'flip_test', False)
        config.TEST.USE_GT_BBOX = getattr(self.test_cfg, 'use_gt_bbox', False)
        
        return config
    
    def _log_evaluation_progress(self, batch_idx, valid_loader_len, batch_time,
                               num_images, epoch_start_time, logger):
        """평가 진행상황 로깅"""
        msg = f"Test[{batch_idx}/{valid_loader_len}] "
        msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Time{sc.ENDC} {batch_time.avg:.3f}(s) "
        msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Speed{sc.ENDC} {num_images/batch_time.avg:.1f}(samples/s) "
        msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Loss{sc.ENDC} {self.losses.avg:.4f} "
        msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Accuracy{sc.ENDC} {self.acc.avg:.3f} "
        
        elapsed_time = str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (valid_loader_len - batch_idx - 1)).split('.')[0].split(':')
        msg += f"| {sc.COLOR_LIGHT_BLUE}ETA{sc.ENDC} {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s |"
        
        logger.info(msg)
    
    def _print_evaluation_results(self, name_values, logger):
        """평가 결과 출력"""
        model_name = f"{self.dataset_cfg.name}_evaluation"
        
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name, logger)
        else:
            self._print_name_value(name_values, model_name, logger)
    
    def _print_name_value(self, name_value, full_arch_name, logger):
        """이름-값 쌍 출력"""
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        
        logger.info("| Arch " + " ".join([f"| {name}" for name in names]) + " |")
        logger.info("|---" * (num_values + 1) + "|")
        logger.info(
            "| "
            + full_arch_name
            + "\n"
            + " "
            + " ".join(["| {:.3f}".format(value) for value in values])
            + " |"
        )
    
    def get_evaluation_metrics(self) -> Dict[str, float]:
        """평가 메트릭 반환"""
        return {
            "avg_loss": self.losses.avg,
            "avg_accuracy": self.acc.avg,
            "num_samples": self.losses.count
        } 