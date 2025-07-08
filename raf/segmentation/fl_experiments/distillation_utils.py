"""Multi-resolution knowledge distillation utilities."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def kd_loss(outputs: List[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    """Simple KL-based distillation across list of segmentation logits."""
    # pick highest resolution as teacher
    teacher = outputs[0]  # assume outputs sorted high→low res
    loss = torch.zeros(1, device=teacher.device)
    for student in outputs[1:]:
        # upsample student to teacher size
        student_up = F.interpolate(student, size=teacher.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + F.kl_div(
            F.log_softmax(student_up / temperature, dim=1),
            F.softmax(teacher.detach() / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)
    return loss / max(1, len(outputs) - 1) 