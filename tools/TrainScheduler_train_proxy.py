import os

from lib.utils.utils import ShellColors as sc
from copy import deepcopy
from lib.utils.average_meter import AverageMeter
import time
from datetime import datetime
import torch
from itertools import cycle
from collections import OrderedDict
from lib.dataset.mpii import MPIIDataset
from torch.utils.data import Dataset
import numpy as np

import dataset
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from typing import Union

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        input_img, target_joints, target_joints_vis, heatmaps, heatmaps_weights, meta = self.dataset[self.idxs[item]]
        return input_img, target_joints, target_joints_vis, heatmaps, heatmaps_weights, meta

class TrainScheduler:
    def __init__(self, config, wdb, clients: list, server, criterion, gpu) -> None:
        self.config = config
        self.wdb = wdb
        self.clients = clients
        self.server = server
        self.criterion = criterion
        self.gpu = gpu

        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.proxy_dataset = MPIIDataset(
            cfg=config,
            root=config.DATASET_PROXY.ROOT,
            image_set=config.DATASET_PROXY.TRAIN_SET,
            is_train=True,
            transform=  transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        self.train_dataset = MPIIDataset(
            cfg=config,
            root=config.DATASET_SETS[0].ROOT,
            image_set=config.DATASET_SETS[0].TRAIN_SET,
            is_train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        
        dict_split_base = self.dataset_split(self.train_dataset, split_size=1000)
        dict_splits = {}
        current_union = set()
        for i in range(len(dict_split_base)):
            current_union = current_union.union(dict_split_base[i])
            dict_splits[i] = current_union
        
        index = self.config.DATASET.NUMBER_OF_SPLITS - 1
        
        final_train_dataset = DatasetSplit(self.train_dataset, dict_splits[index])
        print(f"The length of splitted dataset is {len(final_train_dataset)}")
        
        self.train_sampler = None
        self.proxy_sampler = None
        # self.train_sampler = DistributedSampler(final_train_dataset, shuffle=True)
        # # self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        # self.proxy_sampler = DistributedSampler(self.proxy_dataset, shuffle=True)

        self.train_loader = DataLoader(
            # self.train_dataset,
            final_train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=(self.train_sampler is None),
            num_workers=config.WORKERS,
            pin_memory=True,
            sampler=self.train_sampler,
        )
        
        self.proxy_loader = DataLoader(
            self.proxy_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=(self.proxy_sampler is None),
            num_workers=config.WORKERS,
            pin_memory=True,
            sampler=self.proxy_sampler,
        )
        
        # self.global_client_model_params = None
        # _dummy_model = self.clients[0].model.state_dict(keep_vars=True)
        # self.global_client_model_params = OrderedDict(
        #     _dummy_model.state_dict(keep_vars=True)
        # )

    def dataset_split(self, dataset, split_size):
        num_samples = len(dataset)  # 전체 데이터 개수
        split_base_num = int(num_samples / split_size)
        dict_splits = {}  # 각 split dataset을 저장할 딕셔너리
        all_idxs = np.arange(num_samples)  # 전체 데이터 인덱스 리스트
        
        for i in range(split_base_num):
            dict_splits[i] = set(np.random.choice(all_idxs, split_size, replace=False))
            all_idxs = list(set(all_idxs) - dict_splits[i])

        return dict_splits
    
    def random_cutmix_same_pos(self, images, heatmaps, MASK_HOLES_NUM=2):
        """
        images: List of tensors, each with shape (N, C, W_i, H_i) for different resolutions
        heatmaps: List of tensors, each with shape (N, C, W_hm, H_hm) for different resolutions
        """
        images_out = deepcopy(images)
        heatmaps_out = deepcopy(heatmaps)
        
        N, _, W_ref, H_ref = images[0].shape  # 기준 해상도
        device = images[0].device

        # 기준 해상도에서 마스크 좌표 생성
        center_x = torch.randint(0, W_ref, (N, MASK_HOLES_NUM), device=device).int()
        center_y = torch.randint(0, H_ref, (N, MASK_HOLES_NUM), device=device).int()
        size = torch.randint(10, 20, (N, MASK_HOLES_NUM, 2), device=device).int()

        # 다른 해상도에 대해 좌표 스케일링
        updated_images = []
        updated_heatmaps = []
        
        rand_index = torch.randperm(N).to(device)

        for idx, (img, hm) in enumerate(zip(images_out, heatmaps_out)):
            _, _, W_i, H_i = img.shape
            _, _, W_hm, H_hm = hm.shape

            # 스케일링 비율 계산
            scale_x = W_i / W_ref
            scale_y = H_i / H_ref
            scale_x_hm = W_hm / W_ref
            scale_y_hm = H_hm / H_ref

            # 마스크 좌표 스케일링
            center_x_scaled = torch.round(center_x * scale_x).int()
            center_y_scaled = torch.round(center_y * scale_y).int()
            size_scaled = torch.round(size * torch.tensor([scale_x, scale_y], device=device)).int()

            center_x_hm_scaled = torch.round(center_x * scale_x_hm).int()
            center_y_hm_scaled = torch.round(center_y * scale_y_hm).int()
            size_hm_scaled = torch.round(size * torch.tensor([scale_x_hm, scale_y_hm], device=device)).int()

            # 클램핑
            x0 = torch.clamp(center_x_scaled - size_scaled[..., 0], 0, W_i)
            y0 = torch.clamp(center_y_scaled - size_scaled[..., 1], 0, H_i)
            x1 = torch.clamp(center_x_scaled + size_scaled[..., 0], 0, W_i)
            y1 = torch.clamp(center_y_scaled + size_scaled[..., 1], 0, H_i)

            x0_hm = torch.clamp(center_x_hm_scaled - size_hm_scaled[..., 0], 0, W_hm)
            y0_hm = torch.clamp(center_y_hm_scaled - size_hm_scaled[..., 1], 0, H_hm)
            x1_hm = torch.clamp(center_x_hm_scaled + size_hm_scaled[..., 0], 0, W_hm)
            y1_hm = torch.clamp(center_y_hm_scaled + size_hm_scaled[..., 1], 0, H_hm)

            # 랜덤 인덱스 생성
            img_rand = img[rand_index]
            hm_rand = hm[rand_index]

            # CutMix 적용
            for i in range(N):
                for j in range(MASK_HOLES_NUM):
                    img[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = img_rand[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]]
                    hm[i, :, y0_hm[i, j]:y1_hm[i, j], x0_hm[i, j]:x1_hm[i, j]] = hm_rand[i, :, y0_hm[i, j]:y1_hm[i, j], x0_hm[i, j]:x1_hm[i,j]]

            updated_images.append(img)
            updated_heatmaps.append(hm)

        return updated_images, updated_heatmaps
    
    def random_cutmix_same_pos_without_high_res_no_gt_aug(self, images, MASK_HOLES_NUM=2):
        """
        images: List of tensors, each with shape (N, C, W_i, H_i) for different resolutions
        """
        images_out = deepcopy(images)
        N, _, W_ref, H_ref = images[0].shape  # 기준 해상도
        device = images[0].device

        # 기준 해상도에서 마스크 좌표 생성
        center_x = torch.randint(0, W_ref, (N, MASK_HOLES_NUM), device=device).int()
        center_y = torch.randint(0, H_ref, (N, MASK_HOLES_NUM), device=device).int()
        size = torch.randint(10, 20, (N, MASK_HOLES_NUM, 2), device=device).int()

        # 다른 해상도에 대해 좌표 스케일링
        updated_images = []
        
        rand_index = torch.randperm(N).to(device)

        for idx, img in enumerate(images_out):
            if idx == 0:
                # print("here is client 0, Don't do anything!")
                updated_images.append(img)
                continue
            
            _, _, W_i, H_i = img.shape

            # 스케일링 비율 계산
            scale_x = W_i / W_ref
            scale_y = H_i / H_ref

            # 마스크 좌표 스케일링
            center_x_scaled = torch.round(center_x * scale_x).int()
            center_y_scaled = torch.round(center_y * scale_y).int()
            size_scaled = torch.round(size * torch.tensor([scale_x, scale_y], device=device)).int()

            # 클램핑
            x0 = torch.clamp(center_x_scaled - size_scaled[..., 0], 0, W_i)
            y0 = torch.clamp(center_y_scaled - size_scaled[..., 1], 0, H_i)
            x1 = torch.clamp(center_x_scaled + size_scaled[..., 0], 0, W_i)
            y1 = torch.clamp(center_y_scaled + size_scaled[..., 1], 0, H_i)

            # 랜덤 인덱스 생성
            # rand_index = torch.randperm(N).to(device)
            img_rand = img[rand_index]

            # CutMix 적용
            for i in range(N):
                for j in range(MASK_HOLES_NUM):
                    img[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = img_rand[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]]

            updated_images.append(img)

        return updated_images
    
    def random_cutout_same_pos_no_gt_aug(self, images, MASK_HOLES_NUM=2):
        """
        images: List of tensors, each with shape (N, C, W_i, H_i) for different resolutions
        heatmaps: List of tensors, each with shape (N, C, W_hm, H_hm) for different resolutions
        """
        
        images_out = deepcopy(images)
        
        N, _, W_ref, H_ref = images[0].shape  # 기준 해상도
        device = images[0].device

        # 기준 해상도에서 마스크 좌표 생성
        center_x = torch.randint(0, W_ref, (N, MASK_HOLES_NUM), device=device).int()
        center_y = torch.randint(0, H_ref, (N, MASK_HOLES_NUM), device=device).int()
        size = torch.randint(10, 20, (N, MASK_HOLES_NUM, 2), device=device).int()

        # 다른 해상도에 대해 좌표 스케일링
        updated_images = []

        for idx, img in enumerate(images_out):
            _, _, W_i, H_i = img.shape

            # 스케일링 비율 계산
            scale_x = W_i / W_ref
            scale_y = H_i / H_ref

            # 마스크 좌표 스케일링
            center_x_scaled = torch.round(center_x * scale_x).int()
            center_y_scaled = torch.round(center_y * scale_y).int()
            size_scaled = torch.round(size * torch.tensor([scale_x, scale_y], device=device)).int()

            # 클램핑
            x0 = torch.clamp(center_x_scaled - size_scaled[..., 0], 0, W_i)
            y0 = torch.clamp(center_y_scaled - size_scaled[..., 1], 0, H_i)
            x1 = torch.clamp(center_x_scaled + size_scaled[..., 0], 0, W_i)
            y1 = torch.clamp(center_y_scaled + size_scaled[..., 1], 0, H_i)

            # CutMix 적용
            for i in range(N):
                for j in range(MASK_HOLES_NUM):
                    img[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = 0

            updated_images.append(img)

        return updated_images
    
    def random_cutout_same_pos_without_high_res_no_gt_aug(self, images, MASK_HOLES_NUM=2):
        """
        images: List of tensors, each with shape (N, C, W_i, H_i) for different resolutions
        heatmaps: List of tensors, each with shape (N, C, W_hm, H_hm) for different resolutions
        """
        
        images_out = deepcopy(images)
        
        N, _, W_ref, H_ref = images[0].shape  # 기준 해상도
        device = images[0].device

        # 기준 해상도에서 마스크 좌표 생성
        center_x = torch.randint(0, W_ref, (N, MASK_HOLES_NUM), device=device).int()
        center_y = torch.randint(0, H_ref, (N, MASK_HOLES_NUM), device=device).int()
        size = torch.randint(10, 20, (N, MASK_HOLES_NUM, 2), device=device).int()

        # 다른 해상도에 대해 좌표 스케일링
        updated_images = []

        for idx, img in enumerate(images_out):
            if idx == 0:
                updated_images.append(img)
                continue
            _, _, W_i, H_i = img.shape

            # 스케일링 비율 계산
            scale_x = W_i / W_ref
            scale_y = H_i / H_ref

            # 마스크 좌표 스케일링
            center_x_scaled = torch.round(center_x * scale_x).int()
            center_y_scaled = torch.round(center_y * scale_y).int()
            size_scaled = torch.round(size * torch.tensor([scale_x, scale_y], device=device)).int()

            # 클램핑
            x0 = torch.clamp(center_x_scaled - size_scaled[..., 0], 0, W_i)
            y0 = torch.clamp(center_y_scaled - size_scaled[..., 1], 0, H_i)
            x1 = torch.clamp(center_x_scaled + size_scaled[..., 0], 0, W_i)
            y1 = torch.clamp(center_y_scaled + size_scaled[..., 1], 0, H_i)

            # CutMix 적용
            for i in range(N):
                for j in range(MASK_HOLES_NUM):
                    img[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = 0

            updated_images.append(img)

        return updated_images

    
    def random_cutmix(self, image, heatmap, MASK_HOLES_NUM=2):
        N, _, W_i, H_i = image.shape
        _, _, W_hm, H_hm = heatmap.shape
        
        device = image.device
        
        center_x = torch.randint(0, W_i, (N,MASK_HOLES_NUM), device=device).int()
        center_y = torch.randint(0, H_i, (N,MASK_HOLES_NUM), device=device).int()
        size = torch.randint(10, 20, (N,MASK_HOLES_NUM,2), device=device).int()
                
        center_x_hm = torch.round(center_x / 4).int()
        center_y_hm = torch.round(center_y / 4).int()
        size_hm = torch.round(size / 4).int()
        
        x0 = torch.clamp_(center_x-size[...,0],0,W_i)
        y0 = torch.clamp_(center_y-size[...,1],0,H_i)

        x1 = torch.clamp_(center_x+size[...,0],0,W_i)
        y1 = torch.clamp_(center_y+size[...,1],0,H_i)
        
        x0_hm = torch.clamp_(center_x_hm-size_hm[...,0],0,W_hm)
        y0_hm = torch.clamp_(center_y_hm-size_hm[...,1],0,H_hm)

        x1_hm = torch.clamp_(center_x_hm+size_hm[...,0],0,W_hm)
        y1_hm = torch.clamp_(center_y_hm+size_hm[...,1],0,H_hm)
        
        rand_index = torch.randperm(N).cuda()
        image_rand = image[rand_index]
        heatmap_rand = heatmap[rand_index]
        
        for i in range(N):
            for j in range(MASK_HOLES_NUM):
                image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = image_rand[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]]
                heatmap[i, :, y0_hm[i,j]:y1_hm[i,j], x0_hm[i,j]:x1_hm[i,j]] = heatmap_rand[i, :, y0_hm[i,j]:y1_hm[i,j], x0_hm[i,j]:x1_hm[i,j]]
        
        return image, heatmap
    
    def random_cutout(self, image, MASK_HOLES_NUM=2):
        img_aug = deepcopy(image)
        N, _, W_i, H_i = img_aug.shape
        
        device = image.device
        
        center_x = torch.randint(0, W_i, (N,MASK_HOLES_NUM), device=device).int()
        center_y = torch.randint(0, H_i, (N,MASK_HOLES_NUM), device=device).int()
        size = torch.randint(10, 20, (N,MASK_HOLES_NUM,2), device=device).int()
        
        x0 = torch.clamp_(center_x-size[...,0],0,W_i)
        y0 = torch.clamp_(center_y-size[...,1],0,H_i)

        x1 = torch.clamp_(center_x+size[...,0],0,W_i)
        y1 = torch.clamp_(center_y+size[...,1],0,H_i)

        for i in range(N):
            for j in range(MASK_HOLES_NUM):
                img_aug[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = 0
        
        return img_aug
    
    def clone_parameters(
        self,
        src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
        ) -> OrderedDict[str, torch.Tensor]:
        if isinstance(src, OrderedDict):
            return OrderedDict(
                {
                    name: param.clone().detach().requires_grad_(param.requires_grad)
                    for name, param in src.items()
                }
            )
        if isinstance(src, torch.nn.Module):
            return OrderedDict(
                {
                    name: param.clone().detach().requires_grad_(param.requires_grad)
                    for name, param in src.state_dict(keep_vars=True).items()
                }
            )
    
    def train(self, device, logger, epoch):
        # train_batch_size = len(self.clients[0].train_loader)
        train_batch_size = len(self.train_loader)
        proxy_batch_size = len(self.proxy_loader)
        # global_model_param = self.clients[0].model.parameters()
        # global_model = deepcopy(self.clients[0].model)
        
        num_samples = []

        # Proxy Cycle!!!!!!!
        # self.proxy_loader = cycle(iter(self.proxy_loader))
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # losses = AverageMeter()
        # acc = AverageMeter()
        losses_buf = []
        acc_buf = []
        
        for client in self.clients:
            client.losses.reset()
            losses_buf.append(client.losses)
            client.acc.reset()
            acc_buf.append(client.acc)
            client.model.train()
            # for name, param in client.model.named_parameters():
            #     print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        
        global_model_params = self.clone_parameters(OrderedDict(self.clients[0].model.state_dict()))
        global_model_params_list = list(global_model_params.values())
        # trainable_global_params = list(
        #     filter(lambda p: p.requires_grad, global_model_params.values())
        # )

        end = time.time()
        epoch_start_time = datetime.now()
        
        mu = 1.0
        
        # self.trainloaders_iter = []
        # for client in self.clients:
        #     client.train_sampler.set_epoch(epoch=epoch)
        #     self.trainloaders_iter.append(iter(client.train_loader))
        
        # -------- train_kd 용 -----------
        # self.clients[0].train_sampler.set_epoch(epoch=epoch)
        # self.train_sampler.set_epoch(epoch=epoch)
        # train_iters = iter(self.clients[0].train_loader)
        train_iters = iter(self.train_loader)
        
        # iterator로 만들어주기.
        # self.proxy_loader = iter(self.proxy_loader)
        # self.proxy_sampler.set_epoch(epoch=epoch)
        proxy_iter = iter(self.proxy_loader)

        # 기존대로 학습한번 가고
        for train_batch_idx in range(train_batch_size):

            Hps = []
            gt_heatmaps = []
            heatmap_weights_list = []
            activations_privacy = []
            clean_output = []

            # Step 1: 모든 client들의 activation 모으기.
            imgs, target_joint, target_joints_vis, heatmaps, heatmap_weights, meta = next(train_iters)
            # imgs, heatmaps는 list로 나옴 (각 client의 resolution 별로)
            
            # cutmixed input & gt_heatmap
            if self.config.DATASET.CUTMIX:
                if self.config.DATASET.SAME_POS:
                    if self.config.DATASET.CLEAN_HIGH:
                        imgs_aug = self.random_cutmix_same_pos_without_high_res_no_gt_aug(imgs)
                    else:
                        imgs_aug, _ = self.random_cutmix_same_pos(imgs, heatmaps) # no_gt_aug
            elif self.config.DATASET.CUTOUT:
                if self.config.DATASET.SAME_POS:
                    if self.config.DATASET.CLEAN_HIGH:
                        imgs_aug = self.random_cutout_same_pos_without_high_res_no_gt_aug(imgs)
                    else:
                        imgs_aug = self.random_cutout_same_pos_no_gt_aug(imgs)
            else:
                imgs_aug = imgs

            prox_terms = []
            
            #--------forward prop -------------
            for client_idx, client in enumerate(self.clients):
                if self.gpu == 0 and train_batch_idx == 0:
                    print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
                    print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_BLUE}Training{sc.ENDC} ------------")
                    print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
                
                # client.model.train()

                # train_loader의 mini batch 값들
                # imgs, target_joint, target_joints_vis, heatmaps, heatmap_weights, meta = next(self.trainloaders_iter[client_idx])
                
                if train_batch_idx == 0 and client_idx == 0:
                    print(f"train loader [{train_batch_idx}/{train_batch_size}] filename: {meta['image'][0]}")
                
                # measure data loading time
                torch.autograd.set_detect_anomaly(True)

                # img = imgs[client_idx]
                img = imgs_aug[client_idx]
                heatmap = heatmaps[client_idx]
                heatmap_weight = heatmap_weights[client_idx]
                
                img, heatmap, heatmap_weight = img.to(device), heatmap.to(device), heatmap_weight.to(device)
                # img, heatmap, heatmap_weight = img.to(device), heatmap.to(device), heatmap_weights.to(device)
                
                # cutmixed input & gt_heatmap
                # img, heatmap = self.random_cutmix(img, heatmap)
                
                # # cutout input
                # if client_idx == 0:
                #     img_clean = img.detach()
                # else:
                #     img = self.random_cutout(img)
                
                if client_idx != 0:
                    img = self.random_cutout(img)
                
                if train_batch_idx == 0:
                    num_samples.append(train_batch_size * img.shape[0])

                gt_heatmaps.append(heatmap)
                heatmap_weights_list.append(heatmap_weight)

                data_time.update(time.time() - end)
                
                # proximal_term = 0.0
                # if self.config.FED.FEDPROX:
                #     for w, w_t in zip(client.model.parameters(), global_model.parameters()):
                #         proximal_term += torch.norm(w - w_t, p=2)
                #         # proximal_term += torch.norm(w - w_t, p=2)**2
                
                # proximal term 추가.
                activation, Hp = client.model(img)
                # activation, Hp, proximal_term = client.model(img, global_model)
                    
                # print(f"[{client_idx}] proximal term: {proximal_term if proximal_term is not None else 'None'}")

                activations_privacy.append(activation)
                Hps.append(Hp)
                # prox_terms.append(proximal_term)
                
            
            gradients, grad_norms, losses_buf, acc_buf = self.server.train(
                activations=activations_privacy,
                Hps=Hps,
                gt_heatmaps=gt_heatmaps,
                heatmap_weights=heatmap_weights_list,
                batch_idx=train_batch_idx,
                wdb=self.wdb,
                criterion=self.criterion,
                losses_buf=losses_buf,
                acc_buf=acc_buf,
                device=device,
                # proximal_term=prox_terms if self.config.FED.FEDPROX else None,
                # mu=mu,
                kd_use=True if self.config.KD_USE else False,
            )

            #--------backward prop -------------
            for client_idx, client in enumerate(self.clients):
                client.optimizer_client.zero_grad()
                activations_privacy[client_idx].backward(gradients[client_idx])
                
                # print(f"trainable_global_params: {trainable_global_params}")
                if self.config.FED.FEDPROX:
                    # for w, w_g in zip(client.model.parameters(), trainable_global_params):
                    for w, w_g in zip(client.model.parameters(), global_model_params_list):
                        w.grad.data += mu * (w_g.data - w.data)

                client.optimizer_client.step()
            
            # ----------------- FedAvg per 1 batch ---------------------------------------------------------------
            # if train_batch_idx == 0:
            #     print(f"[{train_batch_idx}/{train_batch_size}] FedAvg per 1 batch ...")
            # # w_glob_client = FedAvg([client.model.state_dict() for client in self.clients]) # 각 client에서 update된 weight를 받아서 FedAvg로 합쳐줌.
            # # w_glob_client = fed_w_avg_sim([client.model.state_dict() for client in self.clients], loss_buffer=loss_buffer) # simple average weighted
            # w_glob_client = fed_w_avg_softmax([client.model.state_dict() for client in self.clients], loss_buffer=loss_buffer) # softmax weighted
            # # w_glob_client = fed_w_avg_softmax_scaled([client.model.state_dict() for client in self.clients], loss_buffer=loss_buffer) # softmax with scaling weighted
            
            # # FedAvg 후에 각 client의 모델을 global client model로 업데이트
            # for client in self.clients:
            #     client.model.load_state_dict(w_glob_client)
            # --------------------------------------------------------------------------------

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train_batch_idx % self.config.PRINT_FREQ == 0 and self.gpu == 0:
                msg = f"Epoch[{epoch}][{train_batch_idx}/{train_batch_size}] "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Time{sc.ENDC} {batch_time.avg:.3f}(s) "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Speed{sc.ENDC} {img.size(0)/batch_time.avg:.1f}(samples/s) "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Data Time{sc.ENDC} {data_time.avg:.3f}(s) "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Loss{sc.ENDC} "
                for losses in losses_buf:
                    msg += f"{losses.avg:.4f} "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Grad Norm{sc.ENDC} "
                for grad_norm in grad_norms:
                    msg += f"{grad_norm:.4f} "
                msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Accuracy{sc.ENDC} "
                for acc in acc_buf:
                    msg += f"{acc.avg:.3f} "
                
                elapsed_time = str((datetime.now()-epoch_start_time) / (train_batch_idx+1) * (train_batch_size-train_batch_idx-1)).split('.')[0].split(':')
                msg += f"| {sc.COLOR_LIGHT_BLUE}ETA{sc.ENDC} {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s |"
                logger.info(msg)
                
                # prefix = f"{os.path.join(output_dir, 'train')}_train_batch_idx"
                # save_debug_images(self.config, img, meta, target_joint, pred * 4, outputs[0], prefix)
                for idx in range(len(self.clients)):
                    if self.wdb:
                        self.wdb.log({f"Client [{idx}] Avg loss": losses_buf[idx].avg})
                        self.wdb.log({f"Client [{idx}] Accuracy": acc_buf[idx].avg})
        
        print(f"Epoch[{epoch}]: Train dataset Success!")
        
        # proxy dataset으로 한번 더 학습
        if self.config.TRAIN.USE_PROXY: # train:proxy = 1:1 비율로 뽑도록 하기.
            proxy_start_time = datetime.now()
            for proxy_batch_idx in range(proxy_batch_size):

                Hps_pr = []
                gt_heatmaps_pr = []
                heatmap_weights_pr = []
                activations_proxy = []
                
                imgs_pr, target_joint_pr, target_joints_vis_pr, heatmaps_pr, heatmap_weights_pr, meta_pr = next(proxy_iter)
                if proxy_batch_idx == 0:
                    print(f"proxy loader [{proxy_batch_idx}/{proxy_batch_size}] filename: {meta_pr['image'][0]}")
                
                # Step 1: 모든 client들의 activation 모으기.
                #--------forward prop -------------
                for client_idx, client in enumerate(self.clients):
                    if self.gpu == 0 and proxy_batch_idx == 0:
                        print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
                        print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_BLUE}Proxy Distillation{sc.ENDC} ------------")
                        print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")

                    img_pr = imgs_pr[client_idx]
                    heatmap_pr = heatmaps_pr[client_idx]

                    img_pr, heatmap_pr, heatmap_weights_pr = img_pr.to(device), heatmap_pr.to(device), heatmap_weights_pr.to(device)

                    # Server로 보낼 gt_heatmap과 heatmap_weights를 list에 저장
                    gt_heatmaps_pr.append(heatmap_pr)
                    heatmap_weights_pr.append(heatmap_weights_pr)

                    data_time.update(time.time() - end)
                    client.optimizer.zero_grad()
                    
                    # Forward Propagation
                    activation_pr, Hp_pr = client.model(img_pr)
                    
                    # Backpropagation을 위한 retain_grad 설정
                    client_proxy_activation_pr = activation_pr.clone().detach().requires_grad_(True)
                    
                    # Server로 보낼 activation과 Hp를 list에 저장
                    activations_proxy.append(client_proxy_activation_pr)
                    Hps_pr.append(Hp_pr)
                
                # Server로 보내서 Forward 후, Backprop한 gradient를 받음.
                # is_proxy=True이면 loss_kd + loss_hm으로 total loss가 됨.
                gradients, grad_norms = self.server.train(
                    activations=activations_proxy,
                    Hps=Hps_pr,
                    gt_heatmaps=gt_heatmaps_pr,
                    heatmap_weights=heatmap_weights_pr,
                    batch_idx=proxy_batch_idx,
                    wdb=self.wdb,
                    criterion=self.criterion,
                    losses_buf=losses_buf,
                    acc_buf=acc_buf,
                    device=device,
                    is_proxy=True,
                )
                
                #--------backward prop -------------
                for client_idx, client in enumerate(self.clients):
                    activations_proxy[client_idx].backward(gradients[client_idx])
                    # client.optimizer.step()
                
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if proxy_batch_idx % self.config.PRINT_FREQ == 0 and self.gpu == 0:
                    msg = f"Epoch[{epoch}][{proxy_batch_idx}/{proxy_batch_size}] "
                    msg += f"Time[{batch_time.val:.3f}/{batch_time.avg:.3f}] "
                    msg += f"Speed[{img_pr.size(0)*3/batch_time.val:.1f} samples/s] "
                    msg += f"Data[{data_time.val:.3f}/{data_time.avg:.3f}] "
                    # msg += f"Loss[{losses.val:.4f}/{losses.avg:.4f}] "
                    msg += f"Grad Norm[{grad_norms[0]:.4f}] "
                    # msg += f"Accuracy[{acc.val:.3f}/{acc.avg:.3f}] "
                    
                    msg += f"ETA[{str((datetime.now()-proxy_start_time) / (proxy_batch_idx+1) * (proxy_batch_size-proxy_batch_idx-1)).split('.')[0]}]"
                    logger.info(msg)
                    # prefix = "{}_{}".format(os.path.join(output_dir, "train"), batch_idx)
                    # save_debug_images(self.config, img, meta, target_joint, pred * 4, outputs[0], prefix)
                    for idx in range(len(self.clients)):
                        if self.wdb:
                            self.wdb.log({f"Client [{idx}] Avg loss": losses_buf[idx].avg})
                            self.wdb.log({f"Client [{idx}] Accuracy": acc_buf[idx].avg})
            
            print(f"Epoch[{epoch}]: Proxy dataset (Distillation) Success!")

        # After serving all clients for its local epochs------------
        # Federation process at Client-Side------------------------
        if self.gpu == 0:
            print(f"{sc.COLOR_RED}------------------------------------------------------------{sc.ENDC}")
            print(f"{sc.COLOR_RED}------ Fed Server: Federation process at Client-Side -------{sc.ENDC}")
            print(f"{sc.COLOR_RED}------------------------------------------------------------{sc.ENDC}")
        
        # self.global_client_model_params = FedAvg([client.model.state_dict() for client in self.clients], num_samples=num_samples) # 각 client에서 update된 weight를 받아서 FedAvg로 합쳐줌.
        w_glob_client = FedAvg([client.model.state_dict() for client in self.clients], num_samples=num_samples) # 각 client에서 update된 weight를 받아서 FedAvg로 합쳐줌.
        # w_glob_client = fed_w_avg_sim([client.model.state_dict() for client in self.clients], loss_buffer=loss_buffer) # simple average weighted
        # w_glob_client = fed_w_avg_softmax([client.model.state_dict() for client in self.clients], loss_buffer=loss_buffer) # softmax weighted
        # w_glob_client = fed_w_avg_softmax_scaled([client.model.state_dict() for client in self.clients], loss_buffer=loss_buffer) # softmax with scaling weighted
        if self.gpu == 0:
            logger.info("Federation Process Done!")
        
        # Update client-side global model
        if self.gpu == 0:
            logger.info("load Fed-Averaged weight to the global client model ...")
        
        # FedAvg 후에 각 client의 모델을 global client model로 업데이트
        for client in self.clients:
            client.model.load_state_dict(w_glob_client)
        
        logger.info(f"This epoch takes {datetime.now() - epoch_start_time}")

        return w_glob_client

# Federated averaging: FedAvg
def FedAvg(weights: list, num_samples: list = None):
    if num_samples is None:
        num_samples = [1 for i in range(len(weights))]
    print(f"num_samples = {num_samples}")
    # w_avg = deepcopy(weights[0]) # weight_averaged 초기화
    w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화
    
    for k in w_avg.keys():
        # for i in range(1, len(weights)): # client의 개수 - 1 만큼 반복
        for i in range(len(weights)):
            w_avg[k] += weights[i][k].detach() * num_samples[i]
            # w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], sum(num_samples))
    return w_avg

# Federated averaging: Federated Weighted Avearge by softmax
def fed_w_avg_softmax(weights, loss_buffer):
    w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()} # OrderedDict에서 각 텐서를 사용하여 초기화
    # w_avg = OrderedDict({k: torch.zeros_like(v) for k, v in weights[0].items()}) # OrderedDict에서 각 텐서를 사용하여 초기화
    
    loss_tensor = torch.tensor(loss_buffer)
            
    # ----------- fed weights by softmax ------------------
    # inverse loss value & graph detach
    loss_inverse = -loss_tensor.detach()
    fed_weights = torch.softmax(loss_inverse, dim=0)
    # -----------------------------------------------------
    # print(f"fed weights => {fed_weights}")
    
    for k in w_avg.keys():
        for i in range(len(weights)): # client의 개수 - 1 만큼 반복
            w_avg[k] += weights[i][k].detach() * fed_weights[i].item()
    return w_avg

# Federated averaging: Federated Weighted Avearge by softmax with scaling
def fed_w_avg_softmax_scaled(weights, loss_buffer):
    # w_avg = deepcopy(weights[0]) # weight_averaged 초기화
    # w_avg = torch.zeros_like(weights[0]) # weight_averaged 초기화
    w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화
    
    loss_tensor = torch.tensor(loss_buffer)
            
    # ----------- fed weights by softmax with scale ------------------
    scaling_factor = 10.0
    # inverse loss value & graph detach
    loss_inverse = -loss_tensor.detach() * scaling_factor
    # loss_inverse = torch.log(loss_tensor + 1e-5)
    fed_weights = torch.softmax(loss_inverse, dim=0)
    # -----------------------------------------------------
    # print(f"fed weights => {fed_weights}")
    
    for k in w_avg.keys():
        for i in range(len(weights)): # client의 개수 - 1 만큼 반복
            w_avg[k] += weights[i][k].detach() * fed_weights[i].item()
    return w_avg

# Federated averaging: Federated Weighted Avearge by simple average
def fed_w_avg_sim(weights, loss_buffer):
    # w_avg = deepcopy(weights[0]) # weight_averaged 초기화
    # w_avg = torch.zeros_like(weights[0]) # weight_averaged 초기화
    w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화
    
    # ----------- fed weights by simple avg ------------------
    loss_tensor = torch.tensor(loss_buffer)
    loss_inverse = 1 / (loss_tensor + 1e-12)
    loss_sum = sum(loss_inverse)
    fed_weights = loss_inverse / loss_sum
    # -----------------------------------------------------
    # print(f"fed weights => {fed_weights}")
    
    for k in w_avg.keys():
        for i in range(len(weights)): # client의 개수 - 1 만큼 반복
            w_avg[k] += weights[i][k].detach() * fed_weights[i].item()
    return w_avg