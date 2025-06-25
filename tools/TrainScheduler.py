from lib.utils.utils import ShellColors as sc
from copy import deepcopy
from lib.utils.average_meter import AverageMeter
import time
from datetime import datetime
import torch
from itertools import cycle

import dataset
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

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
        
        self.proxy_dataset = eval(f'dataset.{config.DATASET.DATASET}')(
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

        self.proxy_sampler = DistributedSampler(self.proxy_dataset, shuffle=True)

        self.proxy_loader = DataLoader(
            self.proxy_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=(self.proxy_sampler is None),
            num_workers=config.WORKERS,
            pin_memory=True,
            sampler=self.proxy_sampler,
        )
        # # iterator로 만들어주기.
        # self.proxy_iter = iter(self.proxy_loader)
        
        # self.trainloaders_iter = []
        # self.proxyloaders_iter = []

        # for client in self.clients:
        #     self.trainloaders_iter.append(iter(client.train_loader))

    def train(self, device, logger, epoch):

        train_batch_size = len(self.clients[0].train_loader)
        # proxy_batch_size = len(self.proxy_loader)
        # train_proxy_rate = train_batch_size // proxy_batch_size

        # Proxy Cycle!!!!!!!
        # self.proxy_loader = cycle(iter(self.proxy_loader))
        
        self.trainloaders_iter = []
        for client in self.clients:
            # shuffle 설정
            # client.train_sampler.set_epoch(epoch=epoch)
            self.trainloaders_iter.append(iter(client.train_loader))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        end = time.time()
        epoch_start_time = datetime.now()
        
        # iterator로 만들어주기.
        proxy_iter = iter(self.proxy_loader)

        # 기존대로 학습한번 가고
        for train_batch_idx in range(train_batch_size):

            Hps = []
            gt_heatmaps = []
            heatmap_weights = []
            activations_privacy = []

            # Step 1: 모든 client들의 activation 모으기.

            #--------forward prop -------------
            for client_idx, client in enumerate(self.clients):
                if self.gpu == 0 and train_batch_idx == 0:
                    print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
                    print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_BLUE}Training{sc.ENDC} ------------")
                    print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
                
                client.model.train()

                # train_loader의 mini batch 값들
                imgs, target_joint, target_joints_vis, heatmaps, heatmap_weight, meta = next(self.trainloaders_iter[client_idx])
                # imgs, target_joint, target_joints_vis, heatmaps, heatmap_weight, meta = next(client.train_loader)
                
                # print(f"train loader [{train_batch_idx}/{train_batch_size}] filename: {meta['image'][0]}")
                
                # measure data loading time
                torch.autograd.set_detect_anomaly(True)

                img = imgs[client_idx]
                heatmap = heatmaps[client_idx]
                
                img, heatmap, heatmap_weight = img.to(device), heatmap.to(device), heatmap_weight.to(device)

                gt_heatmaps.append(heatmap)
                heatmap_weights.append(heatmap_weight)

                data_time.update(time.time() - end)

                client.optimizer.zero_grad()
                activation, Hp = client.model(img)

                client_activation = activation.clone().detach().requires_grad_(True)

                activations_privacy.append(client_activation)
                Hps.append(Hp)
            
            # Step 2: 모든 client들의 proxy activation 모으기.

            # if self.config.TRAIN.USE_PROXY and train_batch_idx % (train_proxy_rate + 1) == 0: # train:proxy = 5:1 비율로 뽑도록 하기.
            if self.config.TRAIN.USE_PROXY: # train:proxy = 1:1 비율로 뽑도록 하기.
                
                # print(f"USE Proxy [{train_batch_idx}/{train_batch_size}] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                try:
                    proxy_batch = next(proxy_iter)
                except StopIteration:
                    # print(f"StopIteration! batch index: [{train_batch_idx}]")
                    # print(f"Shffle with new sampler seed, {train_batch_idx + epoch * train_batch_size}")
                    self.proxy_sampler.set_epoch(train_batch_idx + epoch * train_batch_size)
                    proxy_iter = iter(self.proxy_loader)
                    proxy_batch = next(proxy_iter)

                imgs_pr, target_joint_pr, target_joints_vis_pr, heatmaps_pr, heatmap_weight_pr, meta_pr = proxy_batch
                # imgs_pr, target_joint_pr, target_joints_vis_pr, heatmaps_pr, heatmap_weight_pr, meta_pr = next(self.proxy_loader)
                # print(f"proxy loader [{train_batch_idx}/{train_batch_size}] filename: {meta_pr['image'][0]}")

                Hps_pr = []
                gt_heatmaps_pr = []
                heatmap_weights_pr = []
                activations_proxy = []

                # Step 1: 모든 client들의 activation 모으기.
                #--------forward prop -------------
                for client_idx, client in enumerate(self.clients):
                    if self.gpu == 0 and train_batch_idx == 0:
                        print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
                        print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_BLUE}Proxy Distillation{sc.ENDC} ------------")
                        print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
                    
                    # torch.autograd.set_detect_anomaly(True)

                    img_pr = imgs_pr[client_idx]
                    heatmap_pr = heatmaps_pr[client_idx]

                    img_pr, heatmap_pr, heatmap_weight_pr = img_pr.to(device), heatmap_pr.to(device), heatmap_weight_pr.to(device)

                    gt_heatmaps_pr.append(heatmap_pr)
                    heatmap_weights_pr.append(heatmap_weight_pr)

                    activation_pr, Hp_pr = client.model(img_pr)
                    
                    # 역전파를 위한 retain_grad 설정
                    client_proxy_activation_pr = activation_pr.clone().detach().requires_grad_(True)
                    activations_proxy.append(client_proxy_activation_pr)
                    Hps_pr.append(Hp_pr)
                
                activations_total = [activations_privacy, activations_proxy]
                Hps_total = [Hps, Hps_pr]
                gt_heatmaps_total = [gt_heatmaps, gt_heatmaps_pr]
                heatmap_weights_total = [heatmap_weights, heatmap_weights_pr]

            # Sending activations to server and receiving gradients from server

            # if self.config.TRAIN.USE_PROXY and train_batch_idx % (train_proxy_rate + 1) == 0: # train:proxy = 5:1 비율로 뽑도록 하기.
            if self.config.TRAIN.USE_PROXY: # train:proxy = 1:1 비율로 뽑도록 하기.
                gradients, grad_norms = self.server.train(
                    activations=activations_total,
                    Hps=Hps_total,
                    gt_heatmaps=gt_heatmaps_total,
                    heatmap_weights=heatmap_weights_total,
                    batch_idx=train_batch_idx,
                    wdb=self.wdb,
                    criterion=self.criterion,
                    losses=losses,
                    acc=acc,
                    device=device,
                    is_proxy=True,
                )
            else:
                gradients, grad_norms = self.server.train(
                    activations=activations_privacy,
                    Hps=Hps,
                    gt_heatmaps=gt_heatmaps,
                    heatmap_weights=heatmap_weights,
                    batch_idx=train_batch_idx,
                    wdb=self.wdb,
                    criterion=self.criterion,
                    losses=losses,
                    acc=acc,
                    device=device,
                    is_proxy=False,
                )

            #--------backward prop -------------
            for client_idx, client in enumerate(self.clients):
                activations_privacy[client_idx].backward(gradients[client_idx])
                client.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train_batch_idx % self.config.PRINT_FREQ == 0 and self.gpu == 0:
                msg = f"Epoch[{epoch}][{train_batch_idx}/{train_batch_size}] "
                msg += f"Time[{batch_time.val:.3f}/{batch_time.avg:.3f}] "
                msg += f"Speed[{img.size(0)/batch_time.val:.1f} samples/s] "
                msg += f"Data[{data_time.val:.3f}/{data_time.avg:.3f}] "
                msg += f"Loss[{losses.val:.4f}/{losses.avg:.4f}] "
                msg += f"Grad Norm[{grad_norms[0]:.4f}] "
                msg += f"Accuracy[{acc.val:.3f}/{acc.avg:.3f}] "
                
                msg += f"ETA[{str((datetime.now()-epoch_start_time) / (train_batch_idx+1) * (train_batch_size-train_batch_idx-1)).split('.')[0]}]"
                logger.info(msg)
                # prefix = "{}_{}".format(os.path.join(output_dir, "train"), batch_idx)
                # save_debug_images(self.config, img, meta, target_joint, pred * 4, outputs[0], prefix)
                if self.wdb:
                    self.wdb.log({"Avg loss": losses.avg})
                    self.wdb.log({"Accuracy": acc.avg})
        
        print(f"Epoch[{epoch}]: Proxy Distillation Success!")

        # After serving all clients for its local epochs------------
        # Federation process at Client-Side------------------------
        if self.gpu == 0:
            print(f"{sc.COLOR_RED}------------------------------------------------------------{sc.ENDC}")
            print(f"{sc.COLOR_RED}------ Fed Server: Federation process at Client-Side -------{sc.ENDC}")
            print(f"{sc.COLOR_RED}------------------------------------------------------------{sc.ENDC}")
        
        w_glob_client = FedAvg([client.model.state_dict() for client in self.clients]) # 각 client에서 update된 weight를 받아서 FedAvg로 합쳐줌.
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
def FedAvg(weights):
    w_avg = deepcopy(weights[0]) # weight_averaged 초기화
    # w_avg = torch.zeros_like(weights[0])
    
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg