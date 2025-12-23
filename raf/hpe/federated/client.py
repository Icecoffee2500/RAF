# Works as a local trainer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad
from datetime import datetime
import time
import logging
import numpy as np
from copy import deepcopy

from configs.hpe.config import get_model_name
from hpe.utils.training_utils import AverageMeter, gpu_timer
from hpe.federated.scheduler import MultistepWarmUpRestargets
from hpe.dataset.utils.builder import build_train_val_dataloader, build_split_dataset
from hpe.utils.evaluate import accuracy
from hpe.dataset.coco import COCODataset as coco
from hpe.dataset.mpii import MPIIDataset as mpii
from hpe.utils.model_utils import get_vit_optimizer, get_loss
from hpe.utils.logging import ShellColors as sc
from hpe.utils.post_processing import get_final_preds
from hpe.federated.loss_fns import JointsKLDLoss

from hpe.dataset.utils.utils import MixedResolutionDataset, HeteroBatchSampler

from thop import profile

class FLClient:
    def __init__(
        self,
        client_id,
        config,
        device,
        init_model,
        extra,
        wdb,
        logger,
        im_size,
        hm_size,
        batch_size,
        cl_mr=False,
        is_proxy=False,
        samples_per_split=0,
    ):
        self.client_id = client_id
        self.config = config
        self.logger = logger
        self.device = device
        self.model = deepcopy(init_model)  # 이전과 동일하게 deepcopy 사용
        self.losses = AverageMeter()
        self.acc = AverageMeter()
        self.wdb = wdb
        self.is_proxy = is_proxy
        
        # global client model -> gpu
        self.model.to(device)
        
        self.optimizer = get_vit_optimizer(config, self.model, extra)
        self.lr_scheduler = MultistepWarmUpRestargets(
            self.optimizer, milestones=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR
        )
        self.criterion = get_loss(config)
        self.criterion_kd = JointsKLDLoss().to(device)
        
        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if config.DATASET.DATASET == 'coco':
            dataset_class = coco
        elif config.DATASET.DATASET == 'mpii':
            dataset_class = mpii
        else:
            raise ValueError(f"Invalid dataset: {config.DATASET.DATASET}")
        
        # ------------------------------------------------------------------------------
        # 나중에 hydra config의 instantiate 함수로 대체할 것.
        # ------------------------------------------------------------------------------
        train_dataset = dataset_class(
            cfg=config,
            root=config.DATASET.ROOT,
            image_set=config.DATASET.TRAIN_SET,
            image_size=im_size,
            heatmap_size=hm_size,
            is_train=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        if samples_per_split > 0:
            train_dataset = build_split_dataset(
                train_dataset, dataset_idx=self.client_id, samples_per_split=samples_per_split)
        
        self.valid_dataset = dataset_class(
            cfg=config,
            root=config.DATASET.ROOT,
            image_set=config.DATASET.TEST_SET,
            image_size=im_size[0] if isinstance(im_size[0], (np.ndarray, list)) else im_size,
            heatmap_size=hm_size[0] if isinstance(hm_size[0], (np.ndarray, list)) else hm_size,
            is_train=False,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        if cl_mr:
            train_dataset = MixedResolutionDataset(train_dataset)
            batch_sampler = HeteroBatchSampler(dataset_len=len(train_dataset), batch_size=32)
        
        self.train_loader, self.valid_loader = build_train_val_dataloader(
            train_dataset, self.valid_dataset,
            list([config.TRAIN.BATCH_SIZE, config.TEST.BATCH_SIZE]), config.WORKERS,
            cl_mr=cl_mr, batch_sampler=batch_sampler
        )
        
        self.dataset_length = len(train_dataset)
        
    
    def train_single_resolution(self, epoch):
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        self.model.train()
        
        # For FEDPROX
        if self.config.FED.FEDPROX:
            print(f"FEDPROX GOOOOOOOOOOO!!!!!!!!!!!!!")
            global_dict = self.model.state_dict()
            mu = 1.0
        else:
            global_dict = None
            mu = 0.0
        
        epoch_start_time = datetime.now()
        batch_num = len(self.train_loader)
        
        for batch_idx, (img, heatmap, heatmap_weight, meta) in enumerate(self.train_loader):
            # etime = gpu_timer(lambda: self._train_step_single(img, heatmap, heatmap_weight))
            # print(f"batch [{batch_idx}] img.shape: {img.shape}")
            # print(f"batch [{batch_idx}] heatmap.shape: {heatmap.shape}")
            # print(f"type(img): {type(img)}")
            # print(f"type(heatmap): {type(heatmap)}")
            # for i, (im, hm) in enumerate(zip(img, heatmap)):
            #     print(f"img[{i}].shape: {im.shape}")
            #     print(f"heatmap[{i}].shape: {hm.shape}")
            etime = gpu_timer(
                lambda: self._train_step_single(
                    img, heatmap, heatmap_weight,
                    global_dict=global_dict,
                    mu=mu
                )
            )
            batch_time.update(etime)
            
            # logging
            self._log_while_training(
                idx=self.client_id,
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=self.logger,
                train_batch_idx=batch_idx,
                train_batch_size=batch_num,
                total_batch_time=batch_time,
            )
    
    def _train_step_single(self, img, heatmap, heatmap_weight, **proximal):
        # FedProx
        global_dict = proximal.get("global_dict", None)
        mu = proximal.get("mu", None)
        
        # forward propagation
        img, heatmap, heatmap_weight = img.to(self.device), heatmap.to(self.device), heatmap_weight.to(self.device)
        # macs, params = profile(self.model, inputs=(img,), verbose=False)

        # flops = macs * 2
        # gflops = flops / 1e9

        # bytes_per_param = 4  # float32
        # model_size_bytes = params * bytes_per_param
        # model_size_mb = model_size_bytes / (1024 ** 2)

        # print(f"GFLOPs: {gflops:.2f} GFLOPs")
        # print(f"Model Size: {model_size_mb:.2f} MB")

        output = self.model(img)
        
        # calculate privacy loss
        loss = self.cal_loss(
            self.config,
            self.criterion,
            output,
            heatmap,
            heatmap_weight,
        )
        
        # backward propagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # calculate gradient norm
        grad_norm = self.clip_grads(self.model.parameters())
        
        # FedProx일 경우.
        # local 모델의 named_parameters와 순회하면서
        if global_dict is not None and mu is not None:
            for name, local_param in self.model.named_parameters():
                if name not in global_dict:
                    # 이 이름은 global에 없으므로 건너뛰거나 warning
                    print(f"[no global param] {name}")
                    continue

                global_param = global_dict[name]
                # 크기 체크
                if local_param.shape != global_param.shape:
                    print(f"[mismatch] {name}: local {tuple(local_param.shape)} vs global {tuple(global_param.shape)}")
                    continue
                
                # FedProx 적용
                if local_param.grad is None:
                    local_param.grad = torch.zeros_like(local_param)
                with torch.no_grad():
                    delta = mu * (global_param.detach() - local_param.detach())
                    local_param.grad.add_(delta)
        
        # step optimizer
        self.optimizer.step()
        
        # calculate accuracy
        _, avg_acc, cnt, pred = accuracy(
            output.detach().cpu().numpy(), heatmap.detach().cpu().numpy()
        ) # cnt는 acc가 0이상인 것의 개수
        
        # record accuracy
        self.acc.update(avg_acc, cnt)
        self.losses.update(loss.item(), img.size(0))

    def train_multi_resolution(self, epoch):
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        self.model.train()
        
        # For FEDPROX
        if self.config.FED.FEDPROX:
            print(f"FEDPROX GOOOOOOOOOOO!!!!!!!!!!!!!")
            global_dict = self.model.state_dict()
            mu = 1.0
        else:
            global_dict = None
            mu = 0.0
        
        epoch_start_time = datetime.now()
        batch_num = len(self.train_loader)
        
        for batch_idx, (imgs, heatmaps, heatmap_weights, meta) in enumerate(self.train_loader):
            etime = gpu_timer(
                lambda: self._train_step_multi(
                    imgs, heatmaps, heatmap_weights,
                    global_dict=global_dict,
                    mu=mu
                )
            )
            batch_time.update(etime)
            
            # logging
            self._log_while_training(
                idx=self.client_id,
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=self.logger,
                train_batch_idx=batch_idx,
                train_batch_size=batch_num,
                total_batch_time=batch_time,
            )
    
    def _train_step_multi(self, imgs, heatmaps, heatmap_weights, **proximal):
        # forward propagation
        teacher_output = None
        total_avg_acc = 0.0
        total_cnt = 0
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # FedProx
        global_dict = proximal.get("global_dict", None)
        mu = proximal.get("mu", None)
        
        imgs = [img.to(self.device) for img in imgs]
        heatmaps = [heatmap.to(self.device) for heatmap in heatmaps]
        heatmap_weights = [heatmap_weight.to(self.device) for heatmap_weight in heatmap_weights]
        
        for idx, img in enumerate(imgs):
            output = self.model(img)
            heatmap = heatmaps[idx]
            heatmap_weight = heatmap_weights[idx]
            
            # calculate gt loss
            # heatmap loss
            loss_gt = self.cal_loss(
                self.config,
                self.criterion, # MSE
                output,
                heatmap,
                heatmap_weight,
            )
            
            # knolwedge distillation loss
            if teacher_output is not None:
                output_interpolated = F.interpolate(output, size=(teacher_output.shape[2], teacher_output.shape[3]), mode='nearest')
                
                # knowledge distillation loss (MSE)
                loss_kd = self.cal_loss(
                    self.config,
                    self.criterion,
                    output_interpolated,
                    teacher_output.detach(),
                    heatmap_weight,
                )
                # Online Knowledge Distillation for Efficient Pose Estimation # kd loss is kl div loss!!
                # https://github.com/zhengli97/OKDHP/tree/master
                # loss_kd = self.criterion_kd(
                #     output_interpolated,
                #     teacher_output.detach(),
                #     heatmap_weights[idx]
                # )
                
                alpha = self.config.KD_ALPHA
                loss = loss_gt * alpha + loss_kd * (1 - alpha)
                # print(f"[{img.shape[2]}x{img.shape[3]}] KD Loss Used")
                # loss = loss_kd * (1 - alpha)
                # loss = loss_gt + loss_kd
            else:
                loss = loss_gt
                # print(f"[{img.shape[2]}x{img.shape[3]}] GT Loss Used")
                # loss = 0
            
            # if idx == 0:
            #     teacher_output = output
            teacher_output = output
            
            # calculate accuracy
            _, avg_acc, cnt, pred = accuracy(
                output.detach().cpu().numpy(), heatmap.detach().cpu().numpy()
            ) # cnt는 acc가 0이상인 것의 개수
        
            # sum loss, acc, cnt
            total_loss = total_loss + loss
            total_avg_acc = total_avg_acc + avg_acc
            total_cnt = total_cnt + cnt
        
        # averaging loss, acc
        # total_avg_loss = total_loss / len(imgs)
        # total_avg_loss = total_loss
        loss_scale = self.config.LOSS_SCALE
        
        total_avg_loss = total_loss * loss_scale
        total_avg_acc = total_avg_acc / len(imgs)
        
        # backward propagation
        self.optimizer.zero_grad()
        total_avg_loss.backward()
        
        # calculate gradient norm
        grad_norm = self.clip_grads(self.model.parameters())
        
        # FedProx일 경우.
        # local 모델의 named_parameters와 순회하면서
        if global_dict is not None and mu is not None:
            for name, local_param in self.model.named_parameters():
                if name not in global_dict:
                    # 이 이름은 global에 없으므로 건너뛰거나 warning
                    print(f"[no global param] {name}")
                    continue

                global_param = global_dict[name]
                # 크기 체크
                if local_param.shape != global_param.shape:
                    print(f"[mismatch] {name}: local {tuple(local_param.shape)} vs global {tuple(global_param.shape)}")
                    continue
                
                # FedProx 적용
                if local_param.grad is None:
                    local_param.grad = torch.zeros_like(local_param)
                with torch.no_grad():
                    delta = mu * (global_param.detach() - local_param.detach())
                    local_param.grad.add_(delta)
        
        # step optimizer
        self.optimizer.step()
        
        # record accuracy
        self.acc.update(total_avg_acc, total_cnt)
        self.losses.update(total_avg_loss.item(), imgs[0].size(0))
    
    def evaluate(self, final_output_dir, wdb):
    # def evaluate(self, final_output_dir, backbone, keypoint_head, wdb):
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        valid_loader_len = len(self.valid_loader)
        
        self.model.eval()
        
        num_samples = len(self.valid_dataset) # validation dataset의 전체 object 개수
        # 모든 object의 keypoint를 담을 변수
        all_preds = np.zeros((num_samples, self.config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
        # 모든 object의 bounding box를 담을 변수
        all_boxes = np.zeros((num_samples, 6))
        image_path = [] # image의 파일 경로
        bbox_ids = [] # bounding box의 id (object마다 bounding box의 번호가 부여됨.)
        img_idx = 0 # batch마다 idx를 업데이트 하면서 이미지의 순서를 알려줌.
        
        with torch.no_grad():
            end = time.time()
            epoch_start_time = datetime.now()
            for batch_idx, (img, heatmap, heatmap_weight, meta) in enumerate(self.valid_loader):
                img, heatmap, heatmap_weight = img.to(self.device), heatmap.to(self.device), heatmap_weight.to(self.device)                
                #---------forward prop-------------
                output = self.model(img)
                
                # Flip Test
                if self.config.TEST.FLIP_TEST:
                    img_flipped = img.flip(3).cuda()
                    features_flipped = self.model.backbone(img_flipped)
                    # features_flipped = backbone(img_flipped)
                    
                    output_flipped_heatmap = self.model.keypoint_head.inference_model(
                        features_flipped, meta["flip_pairs"]
                    )
                    # output_flipped_heatmap = keypoint_head.inference_model(
                    #     features_flipped, meta["flip_pairs"]
                    # )
                    output_heatmap = (
                        output + torch.from_numpy(output_flipped_heatmap.copy()).to(self.device)
                    ) * 0.5
                else:
                    output_heatmap = output
                
                loss = self.cal_loss(
                    self.config,
                    self.criterion,
                    output_heatmap,
                    heatmap,
                    heatmap_weight
                )
                
                num_images = img.size(0)
                
                # calculate accuracy
                _, avg_acc, cnt, pred = accuracy(
                    output_heatmap.detach().cpu().numpy(), heatmap.detach().cpu().numpy()
                ) # cnt는 acc가 0이상인 것의 개수
                
                # record accuracy
                self.acc.update(avg_acc, cnt)
                self.losses.update(loss.item(), num_images)
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                c = meta["center"].numpy()
                s = meta["scale"].numpy()
                score = meta["score"].numpy()
                preds, maxvals = get_final_preds(
                    self.config, output_heatmap.clone().cpu().numpy(), c, s, 11)
                
                all_preds[img_idx : img_idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[img_idx : img_idx + num_images, :, 2:3] = maxvals

                # double check this all_boxes parts
                all_boxes[img_idx : img_idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[img_idx : img_idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[img_idx : img_idx + num_images, 4] = np.prod(s * 200, 1) # area
                all_boxes[img_idx : img_idx + num_images, 5] = score
                image_path.extend(meta["image"])
                bbox_ids.extend(meta["bbox_id"])

                img_idx += num_images

                if batch_idx % self.config.PRINT_FREQ == 0:
                    msg = f"Test[{batch_idx}/{valid_loader_len}] "
                    msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Time{sc.ENDC} {batch_time.avg:.3f}(s) "
                    msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Speed{sc.ENDC} {num_images/batch_time.avg:.1f}(samples/s) "
                    msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Loss{sc.ENDC} {self.losses.avg:.4f} "
                    msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Accuracy{sc.ENDC} {self.acc.avg:.3f} "
                    
                    elapsed_time = str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (valid_loader_len - batch_idx - 1)).split('.')[0].split(':')
                    msg += f"| {sc.COLOR_LIGHT_BLUE}ETA{sc.ENDC} {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s |"
                    
                    self.logger.info(msg)
            
            perf_indicator = 0

            name_values, perf_indicator = self.valid_dataset.evaluate(
                self.config, all_preds, final_output_dir, all_boxes, image_path, bbox_ids,
            )

            if wdb:
                wdb.log({f"[Client {self.client_id}] performance": perf_indicator})
                wdb.log({f"[Client {self.client_id}] loss_valid": self.losses.avg})
                wdb.log({f"[Client {self.client_id}] acc_valid": self.acc.avg})

            
            _, full_arch_name = get_model_name(self.config)
            if isinstance(name_values, list):
                for name_value in name_values:
                    self._print_name_value(name_value, full_arch_name)
            else:
                self._print_name_value(name_values, full_arch_name)
            self.logger.info(f"This epoch takes {datetime.now() - epoch_start_time}")
            
        return perf_indicator
    
    def _print_name_value(self, name_value, full_arch_name):
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        self.logger.info("| Arch " + " ".join([f"| {name}" for name in names]) + " |")
        self.logger.info("|---" * (num_values + 1) + "|")
        self.logger.info(
            "| "
            + full_arch_name
            + "\n"
            + " "
            + " ".join(["| {:.3f}".format(value) for value in values])
            + " |"
        )
    
    def cal_loss(
        self,
        config,
        criterion: dict,
        pred_heatmap,
        gt_heatmap,
        hm_weight,
    ):
        loss = 0
        for k, v in criterion.items():
            if k == "heatmap":
                l = v(pred_heatmap, gt_heatmap, hm_weight)
                loss += l * config.LOSS.HM_LOSS_WEIGHT
        return loss
    
    def clip_grads(self, params):
        grad_clip = {"max_norm": 0.003, "norm_type": 2}
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **grad_clip)
    
    def _log_while_training(self, idx, epoch, epoch_start_time, logger, train_batch_idx, train_batch_size, total_batch_time):
        if train_batch_idx % self.config.PRINT_FREQ == 0:
            msg = f"\tEpoch[{epoch}][{train_batch_idx}/{train_batch_size}] "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Time{sc.ENDC} {total_batch_time.avg:.3f}(s) "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Loss{sc.ENDC} "
            msg += f"{self.losses.avg:.4f} "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Accuracy{sc.ENDC} "
            msg += f"{self.acc.avg:.3f} "
            
            elapsed_time = str((datetime.now()-epoch_start_time) / (train_batch_idx+1) * (train_batch_size-train_batch_idx-1)).split('.')[0].split(':')
            msg += f"| {sc.COLOR_LIGHT_BLUE}ETA{sc.ENDC} {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s |"
            logger.info(msg)
            
            # prefix = f"{os.path.join(output_dir, 'train')}_train_batch_idx"
            # save_debug_images(self.config, img, meta, pred * 4, outputs[0], prefix)
            if self.wdb:
                if self.is_proxy:
                    self.wdb.log({f"Proxy Avg loss": self.losses.avg})
                    self.wdb.log({f"Proxy Accuracy": self.acc.avg})
                else:
                    self.wdb.log({f"Client [{idx}] Avg loss": self.losses.avg})
                    self.wdb.log({f"Client [{idx}] Accuracy": self.acc.avg})
    
