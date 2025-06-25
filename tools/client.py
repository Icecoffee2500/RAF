import sys, os
home_dir = os.path.dirname(os.path.abspath(f"{__file__}/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from lib.dataset.coco import COCODataset
from lib.dataset.mpii import MPIIDataset

from lib.core.config import get_model_name
import torch
from lib.utils.average_meter import AverageMeter
from datetime import datetime
import time
import logging
import numpy as np

from copy import deepcopy

from lib.core.scheduler import MultistepWarmUpRestargets
from lib.utils.utils import get_vit_client_optimizer
from lib.dataset.build_dataloader import build_split_dataloader

class Client:
    def __init__(self, idx, config, gpu, device, init_model, extra, client_backbone_block_num, is_proxy=False):
        self.idx = idx
        self.sent = False
        self.received = False
        self.main_server = "main"
        self.fed_server = "fed"
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gpu = gpu
        self.device = device
        self.model = deepcopy(init_model)
        self.losses = AverageMeter()
        self.acc = AverageMeter()
        
        # global client model -> gpu
        self.model.to(device)
        
        self.optimizer_client = get_vit_client_optimizer(config, self.model, extra, client_backbone_block_num=client_backbone_block_num)
        self.lr_scheduler_client = MultistepWarmUpRestargets(
            self.optimizer_client, milestones=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR
        )
        
        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Data loading code
        if is_proxy:
            self.train_loader = build_split_dataloader(
            config=config,
            dataset_class=MPIIDataset,
            dataset_idx=idx,
            root=config.DATASET.ROOT,
            image_set=config.DATASET.TRAIN_SET,
            image_size=config.MODEL.IMAGE_SIZE,
            # image_size=config.MODEL.PR_IMAGE_SIZE,
            heatmap_size=config.MODEL.HEATMAP_SIZE,
            # heatmap_size=config.MODEL.PR_HEATMAP_SIZE,
            is_train=True,
            # split_data=False,
            split_data=True
        )
        else:
            self.train_loader = build_split_dataloader(
                config=config,
                dataset_class=MPIIDataset,
                dataset_idx=idx,
                root=config.DATASET.ROOT,
                image_set=config.DATASET.TRAIN_SET,
                image_size=config.MODEL.IMAGE_SIZE[idx],
                heatmap_size=config.MODEL.HEATMAP_SIZE[idx],
                is_train=True,
                split_data=True
            )
        
        if is_proxy == False:
            self.valid_dataset = MPIIDataset(
                cfg=config,
                # root=config.DATASET_SETS[idx].ROOT,
                root=config.DATASET.ROOT,
                image_set=config.DATASET.TEST_SET,
                image_size=config.MODEL.IMAGE_SIZE[idx],
                heatmap_size=config.MODEL.HEATMAP_SIZE[idx],
                is_train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
            
            self.valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=config.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=config.WORKERS,
                pin_memory=True,
            )
    
    def Send(self, to, data):
        pass
    
    def Receive(self, _from):
        pass

    def UpdateModel(self, updated_model):
        self.model = updated_model
    
    def set_train_loader(self, loader):
        self.train_loader = loader
        self.train_iters = iter(self.train_loader)
    
    def set_train_iters(self):
        self.train_iters = iter(self.train_loader)
    
    # client model forwarding
    def forward_batch(self, img):
        self.activation  = self.model(img)
        return self.activation
    
    def backward_batch(self, gradient, **proximal):
        global_model_params_list = proximal.get("global_model_params_list", None)
        mu = proximal.get("mu", None)
        
        self.optimizer_client.zero_grad()
        self.activation.backward(gradient)
        
        # FedProx일 경우.
        if global_model_params_list is not None and mu is not None:
            for w, w_g in zip(self.model.parameters(), global_model_params_list):
                w.grad.data += mu * (w_g.data - w.data)

        self.optimizer_client.step()
    
    def forward_step(self):
        assert self.train_loader, "Train Loader is needed!!!"
        
        imgs, target_joint, target_joints_vis, heatmaps, heatmap_weights, meta = next(self.train_iters)
        imgs, heatmaps, heatmap_weights = imgs.to(self.device), heatmaps.to(self.device), heatmap_weights.to(self.device)
        return self.forward_batch(imgs), heatmaps, heatmap_weights
            
        
    def train(self, net, server, epoch, final_output_dir, wdb, args, extra, checkpoint_path):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        
        # server = Server(extra, self.config, args, checkpoint_path)
        
        end = time.time()
        epoch_start_time = datetime.now()
        
        for batch_idx, (imgs, target_joint, target_joints_vis, heatmap, heatmap_target, meta) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            torch.autograd.set_detect_anomaly(True)
            
            img = imgs[self.idx]
            heatmap = heatmap[self.idx]
            
            #---------forward prop-------------
            activation  = net(img)
            # print(f"activation shape => {activation.shape}")
            client_activation = activation.clone().detach().requires_grad_(True)
            
            # Sending activations to server and receiving gradients from server
            gradient, grad_norm = server.train(
                activation=client_activation,
                gt_heatmap=heatmap,
                heatmap_weight=heatmap_target,
                batch_idx=batch_idx,
                wdb=wdb,
                losses=losses,
                acc=acc
            )
            
            #--------backward prop -------------
            self.optimizer.zero_grad()
            activation.backward(gradient)
            self.optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            
            if batch_idx % self.config.PRINT_FREQ == 0 and self.gpu == 0:
                msg = f"Epoch[{epoch}][{batch_idx}/{len(self.train_loader)}] "
                msg += f"Time[{batch_time.val:.3f}/{batch_time.avg:.3f}] "
                msg += f"Speed[{img.size(0)/batch_time.val:.1f} samples/s] "
                msg += f"Data[{data_time.val:.3f}/{data_time.avg:.3f}] "
                msg += f"Loss[{losses.val:.4f}/{losses.avg:.4f}] "
                msg += f"Grad Norm[{grad_norm:.4f}] "
                msg += f"Accuracy[{acc.val:.3f}/{acc.avg:.3f}] "
                
                msg += f"ETA[{str((datetime.now()-epoch_start_time) / (batch_idx+1) * (len(self.train_loader)-batch_idx-1)).split('.')[0]}]"
                self.logger.info(msg)
                # prefix = "{}_{}".format(os.path.join(output_dir, "train"), batch_idx)
                # save_debug_images(self.config, img, meta, target_joint, pred * 4, outputs[0], prefix)
                if wdb:
                    wdb.log({"Avg loss": losses.avg})
                    wdb.log({"Accuracy": acc.avg})
        
        if self.gpu == 0:
            self.logger.info(f"This epoch takes {datetime.now() - epoch_start_time}")
            
        return net.state_dict()
    
    def evaluate(self, server, final_output_dir, wdb):
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        
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
            # for batch_idx, (imgs, targets, target_joints_vis, heatmaps, heatmap_weights, meta) in enumerate(self.valid_loader):
            for batch_idx, (img, target, target_joints_vis, heatmap, heatmap_weight, meta) in enumerate(self.valid_loader):
                # img = imgs[self.idx]
                # heatmap = heatmaps[self.idx]
                # heatmap_weight = heatmap_weights[self.idx]
                
                img, heatmap, heatmap_weight = img.to(self.device), heatmap.to(self.device), heatmap_weight.to(self.device)
                
                #---------forward prop-------------
                activation = self.model(img)
                
                # Flip Test
                if self.config.TEST.FLIP_TEST:
                    img_flipped = img.flip(3).cuda()
                    activation_flipped  = self.model(img_flipped)
                
                # Sending activation to server
                img_idx, self.losses, self.acc = server.evaluate(
                    valid_loader_len=len(self.valid_loader),
                    activation=activation,
                    activation_flipped=activation_flipped,
                    gt_heatmap=heatmap,
                    heatmap_weight=heatmap_weight,
                    meta=meta,
                    batch_idx=batch_idx,
                    epoch_start_time=epoch_start_time,
                    end=end,
                    batch_time=batch_time,
                    losses=self.losses,
                    acc=self.acc,
                    img_idx=img_idx,
                    all_preds=all_preds,
                    all_boxes=all_boxes,
                    image_path=image_path,
                    bbox_ids=bbox_ids
                )
            
            perf_indicator = 0

            name_values, perf_indicator = self.valid_dataset.evaluate(
                self.config, all_preds, final_output_dir, all_boxes, image_path, bbox_ids,
            )

            if wdb:
                wdb.log({f"[Client {self.idx}] performance": perf_indicator})
                wdb.log({f"[Client {self.idx}] loss_valid": self.losses.avg})
                wdb.log({f"[Client {self.idx}] acc_valid": self.acc.avg})

            
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