import os

from lib.utils.utils import ShellColors as sc
from lib.utils.utils import clone_parameters
from lib.utils.average_meter import AverageMeter
from lib.utils.timer import gpu_timer
import time
from datetime import datetime
import torch
from collections import OrderedDict
from lib.dataset.mpii import MPIIDataset
from lib.dataset.build_dataloader import build_split_union_dataloader

from lib.dataset.data_augmentation import AugmentData

class Trainer:
    def __init__(self, config, wdb, clients: list, server, gpu, proxy_client=None, glob_client_model=None) -> None:
        self.config = config
        self.wdb = wdb
        self.clients = clients
        self.proxy_client = proxy_client
        self.server = server
        self.gpu = gpu
        self.glob_client_model = glob_client_model
        
        if config.DATASET.AUGMENTATION != "":
            self.aug_fn = AugmentData(
                aug_type=config.DATASET.AUGMENTATION,
                clean_high=config.DATASET.CLEAN_HIGH,
            )
        else:
            self.aug_fn = None
        
        self.mu = config.FED.MU
        # self.client_params_dict: OrderedDict[str : torch.Tensor] = None
        self.trainable_global_client_params: list[torch.Tensor] = None

        # Data loading code
        self.train_loader = build_split_union_dataloader(
            config=config,
            dataset_class=MPIIDataset,
            root=config.DATASET_SETS[0].ROOT,
            image_set=config.DATASET_SETS[0].TRAIN_SET,
            image_size=config.MODEL.IMAGE_SIZE,
            heatmap_size=config.MODEL.HEATMAP_SIZE,
            is_train=True,
            split_data=True
        )
    
    def set_parameters(
        self,
        model_params: OrderedDict[str, torch.Tensor],
    ):
        print("setting global client parameters to proxy model ...")
        # self.trainable_global_client_params = list(
        #     filter(lambda p: p.requires_grad, model_params.values())
        # )
        self.trainable_global_client_params = list(model_params.values())
        # print(f"self.trainable_global_client_params: {self.trainable_global_client_params}")
    
    # full_shared version training
    def train_full_shared(self, device, logger, epoch):
        train_batch_size = len(self.train_loader)
        batch_time = AverageMeter()
        losses_buf = []
        acc_buf = []
        
        for client in self.clients:
            client.losses.reset()
            losses_buf.append(client.losses)
            client.acc.reset()
            acc_buf.append(client.acc)
            client.model.train()
            
            client.set_train_loader(self.train_loader)
        
        global_model_params = clone_parameters(OrderedDict(self.clients[0].model.state_dict()))
        global_model_params_list = list(global_model_params.values())

        epoch_start_time = datetime.now()
        
        # -------- train_kd 용 -----------
        train_iters = iter(self.train_loader)

        print("\n>>> Client Training")
        for train_batch_idx in range(train_batch_size):
            def train_step():
                activations_privacy = []

                # Step 1: 모든 client들의 activation 모으기.
                imgs, target_joint, target_joints_vis, heatmaps, heatmap_weights, meta = next(train_iters) # imgs, heatmaps는 list로 나옴 (각 client의 resolution 별로)
                
                # data augmentation
                if self.aug_fn is not None:
                    imgs = self.aug_fn(imgs)
                
                # samples to device
                imgs = [img.to(device) for img in imgs]
                heatmaps = [heatmap.to(device) for heatmap in heatmaps]
                heatmap_weights = [heatmap_weight.to(device) for heatmap_weight in heatmap_weights]
                
                # forward clients
                for client, img in zip(self.clients, imgs):
                    activation = client.forward_batch(img)
                    activations_privacy.append(activation)
                
                # forward and backward server
                gradients, loss_buffer, acc_buffer = self.server.train(
                    activations=activations_privacy,
                    gt_heatmaps=heatmaps,
                    heatmap_weights=heatmap_weights,
                    kd_use=True if self.config.KD_USE else False
                )
                # accuracy, loss meter update
                for idx, client in enumerate(self.clients):
                    client.acc.update(*acc_buffer[idx])
                    client.losses.update(loss_buffer[idx].item(), activation.size(0))
                
                # backward clients
                for client, gradient in zip(self.clients, gradients):
                    if self.config.FED.FEDPROX: # FedProx일 때, 적용
                        client.backward_batch(gradient, global_model_params_list=global_model_params_list, mu=self.mu)
                    else:
                        client.backward_batch(gradient)

            # measure elapsed time
            etime = gpu_timer(train_step)
            batch_time.update(etime)

            self._log_while_training(
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=logger,
                train_batch_idx=train_batch_idx,
                train_batch_size=train_batch_size,
                total_batch_time=batch_time,
            )        
    
    # proxy kd version split learning
    def train_proxy_kd(self, device, logger, epoch, mu_=None, momentum_scheduler=None):
        epoch_start_time = datetime.now()
        train_batch_size = len(self.proxy_client.train_loader)
        batch_time = AverageMeter()
        
        self.proxy_client.losses.reset()
        self.proxy_client.acc.reset()
        self.proxy_client.model.train()
        
        server_params_dict = OrderedDict(
            self.server.model.state_dict(keep_vars=True)
        )
        server_model_params = clone_parameters(server_params_dict)
        self.server.set_parameters(server_model_params)
        
        # -------- train_kd 용 -----------
        train_iters = iter(self.proxy_client.train_loader)
        
        print("\n>>> Proxy Training")
        # 기존대로 학습한번 가고
        for train_batch_idx in range(train_batch_size):
            def train_step():
                if momentum_scheduler is not None:
                    momentum = next(momentum_scheduler)
                activation_buf = []
                pseudo_activation_buf = []

                # Step 1: 모든 client들의 activation 모으기.
                imgs, target_joint, target_joints_vis, heatmaps, heatmap_weights, meta = next(train_iters) # imgs, heatmaps는 list로 나옴 (각 client의 resolution 별로)
                
                # data augmentation
                if self.aug_fn is not None:
                    imgs = self.aug_fn(imgs)
                
                imgs = [img.to(device) for img in imgs]
                heatmaps = [heatmap.to(device) for heatmap in heatmaps]
                heatmap_weights = [heatmap_weight.to(device) for heatmap_weight in heatmap_weights]
                
                for img in imgs:
                    activation = self.proxy_client.forward_batch(img)
                    activation_buf.append(activation)
                    # # pseudo activation for MOCO
                    # with torch.no_grad():
                    #     pseudo_activation = self.glob_client_model(img)
                    #     pseudo_activation_buf.append(pseudo_activation)
                
                
                #-------- server forward & backward prop -------------
                gradients, loss, acc, cnt = self.server.train_proxy(
                    activations=activation_buf,
                    gt_heatmaps=heatmaps,
                    heatmap_weights=heatmap_weights,
                    kd_use=True if self.config.KD_USE else False,
                    mu=mu_,
                    # momentum=momentum,
                    # pseudo_activations=pseudo_activation_buf
                )
                # record accuracy
                self.proxy_client.acc.update(acc, cnt)
                self.proxy_client.losses.update(loss.item(), activation.size(0))
                
                #-------- client backward prop -------------
                self.proxy_client.optimizer_client.zero_grad()
                for activation, gradient in zip(activation_buf, gradients):
                    activation.backward(gradient)
                
                # FedProx일 경우.
                if self.trainable_global_client_params is not None and mu_ is not None:
                    for w, w_g in zip(self.proxy_client.model.parameters(), self.trainable_global_client_params):
                        # w.grad.data += mu_ * (w_g.data - w.data) # 잘못됨
                        w.grad.data += mu_ * (w.data - w_g.data) # 이게 맞음.
                        print("Using alt proximal term!")
                
                self.proxy_client.optimizer_client.step()
                
                # EMA update
                if momentum_scheduler is not None:
                    print("Using ema update!")
                    # momentum = next(momentum_scheduler)
                    with torch.no_grad():
                        for param, param_g in zip(self.proxy_client.model.parameters(), self.trainable_global_client_params):
                            param.data.mul_(1. - momentum).add_((momentum) * param_g.detach().data)
            
            etime = gpu_timer(train_step)
            # measure elapsed time
            batch_time.update(etime)
            
            self._log_while_training(
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=logger,
                train_batch_idx=train_batch_idx,
                train_batch_size=train_batch_size,
                total_batch_time=batch_time,
                is_proxy=True
            )
    
    # Online Teacher Assistant Knowledge Distillation
    def train_proxy_takd(self, device, logger, epoch):
        epoch_start_time = datetime.now()
        train_batch_size = len(self.proxy_client.train_loader)
        batch_time = AverageMeter()
        
        self.proxy_client.losses.reset()
        self.proxy_client.acc.reset()
        self.proxy_client.model.train()
        
        server_params_dict = OrderedDict(
            self.server.model.state_dict(keep_vars=True)
        )
        server_model_params = clone_parameters(server_params_dict)
        self.server.set_parameters(server_model_params)
        
        # -------- train_kd 용 -----------
        train_iters = iter(self.proxy_client.train_loader)
        
        print("\n>>> Proxy Training")
        # 기존대로 학습한번 가고
        for train_batch_idx in range(train_batch_size):
            def train_step():
                # get items
                imgs, target_joint, target_joints_vis, heatmaps, heatmap_weights, meta = next(train_iters)
                
                imgs = [img.to(device) for img in imgs]
                heatmaps = [heatmap.to(device) for heatmap in heatmaps]
                heatmap_weights = [heatmap_weight.to(device) for heatmap_weight in heatmap_weights]
                
                for idx, img in enumerate(imgs):
                    # device 이동
                    if idx == 0:
                        activation_0 = self.proxy_client.forward_batch(img)
                        heatmap_0 = heatmaps[idx]
                        heatmap_weight_0 = heatmap_weights[idx]
                        
                    else:
                        img_teacher, img_student = imgs[idx - 1], img
                    
                        activation_teacher = self.proxy_client.forward_batch(img_teacher).detach()
                        activation_student = self.proxy_client.forward_batch(img_student)
                        activation_pair = [activation_teacher, activation_student]
                        heatmap_pair = [heatmaps[idx - 1], heatmaps[idx]]
                        heatmap_weight_pair = [heatmap_weights[idx - 1], heatmap_weights[idx]]
                    
                    #-------- server forward & backward prop -------------
                    gradient, loss, acc, cnt = self.server.train_proxy_takd(
                        activation_pair=activation_0 if idx==0 else activation_pair,
                        gt_heatmap_pair=heatmap_0 if idx==0 else heatmap_pair,
                        heatmap_weight_pair=heatmap_weight_0 if idx==0 else heatmap_weight_pair,
                        kd_use=False if idx==0 else True,
                    )
                    # record accuracy
                    self.proxy_client.acc.update(acc, cnt)
                    if idx == 0:
                        self.proxy_client.losses.update(loss.item(), activation_0.size(0))
                    else:
                        self.proxy_client.losses.update(loss.item(), activation_student.size(0))
                    
                    #-------- client backward prop -------------
                    self.proxy_client.optimizer_client.zero_grad()
                    if idx == 0:
                        activation_0.backward(gradient)
                    else:
                        activation_student.backward(gradient)
                    
                    # 이 optimizer 부분이 좀 걸리긴 함...
                    self.proxy_client.optimizer_client.step()
            
            etime = gpu_timer(train_step)
            # measure elapsed time
            batch_time.update(etime)
            
            self._log_while_training(
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=logger,
                train_batch_idx=train_batch_idx,
                train_batch_size=train_batch_size,
                total_batch_time=batch_time,
                is_proxy=True
            )
    
    # proxy is not multi-resolution. proxy is just high or mid or low.
    def train_split_agent(self, logger, epoch):
        epoch_start_time = datetime.now()
        agent = self.proxy_client
        batch_num = len(agent.train_loader)
        total_batch_time = AverageMeter()
        
        agent.losses.reset()
        agent.acc.reset()
        agent.model.train()
        
        agent.set_train_iters()

        print("\n>>> Agent Training")
        for train_batch_idx in range(batch_num):
            def train_step():
                # client model forward
                activation, heatmap, heatmap_weight = agent.forward_step()
                
                # server model forward and backeward
                gradient, grad_norm, loss, avg_acc, cnt = self.server.forward_and_backward(
                    activation=activation,
                    gt_heatmap=heatmap,
                    heatmap_weight=heatmap_weight,
                )
                # record accuracy
                agent.acc.update(avg_acc, cnt)
                agent.losses.update(loss.item(), activation.size(0))
                
                agent.backward_batch(gradient)
            # train_step()
            etime = gpu_timer(train_step)
            # measure elapsed time
            total_batch_time.update(etime)

            self._log_while_training(
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=logger,
                train_batch_idx=train_batch_idx,
                train_batch_size=batch_num,
                total_batch_time=total_batch_time,
            )
    
    # disjoint version training
    def train_each_client(self, logger, epoch):
        epoch_start_time = datetime.now()
        batch_num = min([len(client.train_loader) for client in self.clients])
        total_batch_time = AverageMeter()
        
        for client in self.clients:
            client.losses.reset()
            client.acc.reset()
            client.model.train()
            
            client.set_train_iters()
        
        global_model_params = clone_parameters(OrderedDict(self.clients[0].model.state_dict()))
        global_model_params_list = list(global_model_params.values())

        print("\n>>> Client Training")
        for train_batch_idx in range(batch_num):
            def train_step():
                for idx, client in enumerate(self.clients):
                    # client model forward
                    activation, heatmap, heatmap_weight = client.forward_step()
                    
                    # server model forward and backeward
                    gradient, grad_norm, loss, avg_acc, cnt = self.server.forward_and_backward(
                        activation=activation,
                        gt_heatmap=heatmap,
                        heatmap_weight=heatmap_weight,
                    )
                    # record accuracy
                    client.acc.update(avg_acc, cnt)
                    client.losses.update(loss.item(), activation.size(0))
                    
                    # client backward prop
                    if self.config.FED.FEDPROX: # FedProx일 때, 적용
                        client.backward_batch(gradient, global_model_params_list=global_model_params_list, mu=self.mu)
                    else:
                        client.backward_batch(gradient)
            # train_step()
            etime = gpu_timer(train_step)
            # measure elapsed time
            total_batch_time.update(etime)

            self._log_while_training(
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=logger,
                train_batch_idx=train_batch_idx,
                train_batch_size=batch_num,
                total_batch_time=total_batch_time,
            )
    
    def _log_while_training(self, epoch, epoch_start_time, logger, train_batch_idx, train_batch_size, total_batch_time, is_proxy=False):
        if train_batch_idx % self.config.PRINT_FREQ == 0:
            msg = f"\tEpoch[{epoch}][{train_batch_idx}/{train_batch_size}] "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Time{sc.ENDC} {total_batch_time.avg:.3f}(s) "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Loss{sc.ENDC} "
            if is_proxy:
                msg += f"{self.proxy_client.losses.avg:.4f} "
            else:
                for client in self.clients:
                    msg += f"{client.losses.avg:.4f} "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Accuracy{sc.ENDC} "
            if is_proxy:
                msg += f"{self.proxy_client.acc.avg:.3f} "
            else:
                for client in self.clients:
                    msg += f"{client.acc.avg:.3f} "
            
            elapsed_time = str((datetime.now()-epoch_start_time) / (train_batch_idx+1) * (train_batch_size-train_batch_idx-1)).split('.')[0].split(':')
            msg += f"| {sc.COLOR_LIGHT_BLUE}ETA{sc.ENDC} {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s |"
            logger.info(msg)
            
            # prefix = f"{os.path.join(output_dir, 'train')}_train_batch_idx"
            # save_debug_images(self.config, img, meta, target_joint, pred * 4, outputs[0], prefix)
            for idx in range(len(self.clients)):
                if self.wdb:
                    if is_proxy:
                        self.wdb.log({f"Proxy Avg loss": self.proxy_client.losses.avg})
                        self.wdb.log({f"Proxy Accuracy": self.proxy_client.acc.avg})
                    else:
                        # self.wdb.log({f"Client [{idx}] Avg loss": client.losses.avg})
                        # self.wdb.log({f"Client [{idx}] Accuracy": client.acc.avg})
                        self.wdb.log({f"Client [{idx}] Avg loss": self.clients[idx].losses.avg})
                        self.wdb.log({f"Client [{idx}] Accuracy": self.clients[idx].acc.avg})