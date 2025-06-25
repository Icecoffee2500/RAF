from copy import deepcopy
import torch
from functools import partial

class AugmentData:
    def __init__(self, aug_type, clean_high, MASK_HOLES_NUM=2):
        self.clean_high = clean_high
        if aug_type == "cutmix":
            self.aug_fn = partial(self.random_cutmix_same_pos, MASK_HOLES_NUM=MASK_HOLES_NUM)
        elif aug_type == "cutout":
            self.aug_fn = partial(self.random_cutmix, MASK_HOLES_NUM=MASK_HOLES_NUM)
        else:
            self.aug_fn = None
    
    def __call__(self, images):
        if self.aug_fn is not None:
            return self.aug_fn(images)
        return images

    def random_cutmix_same_pos(self, images, MASK_HOLES_NUM=2):
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
            if idx == 0 and self.clean_high:
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
            img_rand = img[rand_index]

            # CutMix 적용
            for i in range(N):
                for j in range(MASK_HOLES_NUM):
                    img[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = img_rand[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]]

            updated_images.append(img)

        return updated_images

    def random_cutout_same_pos(self, images, MASK_HOLES_NUM=2):
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

        for idx, img in enumerate(images_out):
            if self.clean_high and idx == 0:
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


    def random_cutmix(self, image, MASK_HOLES_NUM=2):
        N, _, W_i, H_i = image.shape
        
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
        
        rand_index = torch.randperm(N).cuda()
        image_rand = image[rand_index]
        
        for i in range(N):
            for j in range(MASK_HOLES_NUM):
                image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = image_rand[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]]
        
        return image

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