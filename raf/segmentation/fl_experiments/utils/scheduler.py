from torch.optim.lr_scheduler import _LRScheduler

# Custom Poly LR Scheduler 클래스 (openmmlab 없이)
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter: int, power: float = 1.0, last_epoch: int = -1, min_lr: float=0.0):
        """
        Poly LR Scheduler.
        - max_iter: 총 iteration 수 (e.g., 160000 for Cityscapes).
        - power: 논문 factor (e.g., 1.0).
        """
        self.max_iter = max_iter
        self.power = power
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Poly 공식: lr = base_lr * (1 - (current_iter / max_iter)) ^ power
        factor = (1 - self.last_epoch / self.max_iter) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]