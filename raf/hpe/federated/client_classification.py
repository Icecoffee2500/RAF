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
# from hpe.utils.evaluate import accuracy
from hpe.dataset.coco import COCODataset as coco
from hpe.dataset.mpii import MPIIDataset as mpii
from hpe.dataset.facefair import MRFairFaceDataset, build_transform, TRAIN_LABEL_CSV, IMG_ROOT, ROOT_DIR
from hpe.utils.model_utils import get_vit_optimizer, get_loss
from hpe.utils.logging import ShellColors as sc
from hpe.utils.post_processing import get_final_preds
from hpe.federated.loss_fns import JointsKLDLoss

from timm.utils import accuracy

from thop import profile

class FLClientClassification:
    def __init__(
        self,
        client_id,
        train_loader,
        valid_loader,
        config,
        device,
        init_model,
        extra,
        wdb,
        logger,
        batch_size,
        is_proxy=False,
        samples_per_split=0,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.valid_loader = valid_loader
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
        # self.criterion = get_loss(config)
        # self.criterion_kd = JointsKLDLoss().to(device)

        self.criterion = torch.nn.CrossEntropyLoss()
        
        
        # ------------------------------------------------------------------------------
        # 나중에 hydra config의 instantiate 함수로 대체할 것.
        # ------------------------------------------------------------------------------
        # train_dataset = dataset_class(
        #     cfg=config,
        #     root=config.DATASET.ROOT,
        #     image_set=config.DATASET.TRAIN_SET,
        #     image_size=im_size,
        #     heatmap_size=hm_size,
        #     is_train=True,
        #     transform=transforms.Compose([transforms.ToTensor(), normalize]),
        # )
        # if samples_per_split > 0:
        #     train_dataset = build_split_dataset(
        #         train_dataset, dataset_idx=self.client_id, samples_per_split=samples_per_split)
        
        # self.valid_dataset = dataset_class(
        #     cfg=config,
        #     root=config.DATASET.ROOT,
        #     image_set=config.DATASET.TEST_SET,
        #     image_size=im_size[0] if isinstance(im_size[0], (np.ndarray, list)) else im_size,
        #     heatmap_size=hm_size[0] if isinstance(hm_size[0], (np.ndarray, list)) else hm_size,
        #     is_train=False,
        #     transform=transforms.Compose([transforms.ToTensor(), normalize]),
        # )
        
        # self.train_loader, self.valid_loader = build_train_val_dataloader(
        #     train_dataset, self.valid_dataset,
        #     list([config.TRAIN.BATCH_SIZE, config.TEST.BATCH_SIZE]), config.WORKERS)
        
        
        # self.dataset_length = len(train_dataset)
        
    
    # def train_single_resolution(self, epoch):
    #     batch_time = AverageMeter()
    #     self.losses.reset()
    #     self.acc.reset()
    #     self.model.train()
        
    #     epoch_start_time = datetime.now()
    #     batch_num = len(self.train_loader)
        
    #     for batch_idx, (img, heatmap, heatmap_weight, meta) in enumerate(self.train_loader):
    #         # etime = gpu_timer(lambda: self._train_step_single(img, heatmap, heatmap_weight))
    #         etime = gpu_timer(
    #             lambda: self._train_step_single(img, heatmap, heatmap_weight)
    #         )
    #         batch_time.update(etime)
            
    #         # logging
    #         self._log_while_training(
    #             idx=self.client_id,
    #             epoch=epoch,
    #             epoch_start_time=epoch_start_time,
    #             logger=self.logger,
    #             train_batch_idx=batch_idx,
    #             train_batch_size=batch_num,
    #             total_batch_time=batch_time,
    #         )
    
    # def _train_step_single(self, img, heatmap, heatmap_weight):
        
    #     # forward propagation
    #     img, heatmap, heatmap_weight = img.to(self.device), heatmap.to(self.device), heatmap_weight.to(self.device)

    #     output = self.model(img)
        
    #     # calculate privacy loss
    #     loss = self.cal_loss(
    #         self.config,
    #         self.criterion,
    #         output,
    #         heatmap,
    #         heatmap_weight,
    #     )
        
    #     # backward propagation
    #     self.optimizer.zero_grad()
    #     loss.backward()
        
    #     # calculate gradient norm
    #     grad_norm = self.clip_grads(self.model.parameters())
        
    #     # step optimizer
    #     self.optimizer.step()
        
    #     # calculate accuracy
    #     _, avg_acc, cnt, pred = accuracy(
    #         output.detach().cpu().numpy(), heatmap.detach().cpu().numpy()
    #     ) # cnt는 acc가 0이상인 것의 개수
        
    #     # record accuracy
    #     self.acc.update(avg_acc, cnt)
    #     self.losses.update(loss.item(), img.size(0))

    def train_single_resolution(self, epoch):
        print_freq = 1000
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        self.model.train()
        
        epoch_start_time = datetime.now()
        batch_num = len(self.train_loader)
        
        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            # etime = gpu_timer(lambda: self._train_step_single(img, heatmap, heatmap_weight))
            etime = gpu_timer(
                lambda: self.train_one_epoch(imgs, labels)
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
                print_freq=print_freq
            )
    
    def train_one_epoch(self, imgs, labels):
        
        loss = 0
                
        imgs = imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # with torch.cuda.amp.autocast(enabled=True):
        outputs = self.model(imgs)
        loss = self.criterion(outputs, labels)
        
        # loss_scaler(
        #     loss, optimizer, clip_grad=max_norm,
        #     parameters=model.parameters(), create_graph=False,
        #     update_grad=(data_iter_step + 1) % args.accum_iter == 0
        # )
        self.optimizer.zero_grad()
        loss.backward

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        # step optimizer
        self.optimizer.step()
        self.losses.update(loss.item(), imgs.size(0))
        self.acc.update(acc1.item(), imgs.size(0))

    @torch.no_grad()
    def evaluate(self, wdb, amp=True):
        print_freq = 1000
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        valid_loader_len = len(self.valid_loader)

        criterion = torch.nn.CrossEntropyLoss()
        # switch to evaluation mode
        self.model.eval()
        
        end = time.time()
        epoch_start_time = datetime.now()
        for batch_idx, (imgs, labels) in enumerate(self.valid_loader):
            img = imgs[self.client_id].to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # compute output
            with torch.cuda.amp.autocast(enabled = amp):
                output = self.model(img)
                loss = criterion(output, labels)

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            batch_size = img.shape[0]

            self.acc.update(acc1.item(), batch_size)
            self.losses.update(loss.item(), img.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0:
                msg = f"Test[{batch_idx}/{valid_loader_len}] "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Time{sc.ENDC} {batch_time.avg:.3f}(s) "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Speed{sc.ENDC} {batch_size/batch_time.avg:.1f}(samples/s) "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Loss{sc.ENDC} {self.losses.avg:.4f} "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Accuracy{sc.ENDC} {self.acc.avg:.3f} "
                
                elapsed_time = str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (valid_loader_len - batch_idx - 1)).split('.')[0].split(':')
                msg += f"| {sc.COLOR_LIGHT_BLUE}ETA{sc.ENDC} {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s |"
                
                self.logger.info(msg)

        if wdb:
            # wdb.log({f"[Client {self.client_id}] performance": perf_indicator})
            wdb.log({f"[Client {self.client_id}] loss_valid": self.losses.avg})
            wdb.log({f"[Client {self.client_id}] acc_valid": self.acc.avg})

            
        #     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #     .format(top1=metric_logger.meters[f'res{res}-acc1'], top5=metric_logger.meters[f'res{res}-acc5'], losses=metric_logger.loss))

        # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #         .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return self.acc.avg
    
    def _evaluate(self, final_output_dir, wdb):
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
    
    def _log_while_training(self, idx, epoch, epoch_start_time, logger, train_batch_idx, train_batch_size, total_batch_time, print_freq):
        if train_batch_idx % print_freq == 0:
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
    

if __name__  == "__main__":
    print(f"ROOT_DIR: {ROOT_DIR.resolve()}")
    print(f"TRAIN_LABEL_CSV: {TRAIN_LABEL_CSV.resolve()}")
    client = FLClientClassification()
    print(f"length of dataset: {client.dataset_length}")
    imgs, label = next(iter(client.train_loader))
    for idx, img in enumerate(imgs):
        print(f"[{idx}] img shape: {img.shape}")
    print(f"label is {label}")