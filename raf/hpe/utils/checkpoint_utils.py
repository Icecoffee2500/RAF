# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import os
import os.path as osp
import re
import copy
import torch
from collections import OrderedDict


def save_checkpoint(states, output_dir, filename="checkpoint.pth.tar"):
    torch.save(states, os.path.join(output_dir, filename)) # resume를 위해서 모든 설정 저장.
    
    # best model state dict 저장
    if "client_state_dict" in states and "server_state_dict" in states:
        # torch.save(states["client_state_dict"], os.path.join(output_dir, f"model_client_{states['client_idx']}_best.pt"))
        torch.save(states["client_state_dict"], os.path.join(output_dir, f"model_client_best.pt"))
        torch.save(states["server_state_dict"], os.path.join(output_dir, "model_server_best.pt"))
    elif "state_dict" in states:
        torch.save(states["state_dict"], os.path.join(output_dir, "model_best.pth"))

def save_checkpoint_fedbn(states, output_dir, filename="checkpoint.pth.tar"):
    torch.save(states, os.path.join(output_dir, filename)) # resume를 위해서 모든 설정 저장.
    
    # best model state dict 저장
    if "high_state_dict" in states:
        torch.save(states["high_state_dict"], os.path.join(output_dir, "high_model_best.pth"))
    if "mid_state_dict" in states:
        torch.save(states["mid_state_dict"], os.path.join(output_dir, "mid_model_best.pth"))
    if "low_state_dict" in states:
        torch.save(states["low_state_dict"], os.path.join(output_dir, "low_model_best.pth"))


def _load_checkpoint(filename, map_location=None):
    if not osp.isfile(filename):
        raise IOError(f"{filename} is not a checkpoint file")
    # checkpoint = torch.load(filename, map_location=map_location)
    checkpoint = torch.load(filename, map_location=map_location, weights_only=True)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )


def load_checkpoint(
    model,
    filename,
    map_location="cpu",
    strict=False,
    logger=None,
    patch_padding="pad",
    part_features=None,
):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        patch_padding (str): 'pad' or 'bilinear' or 'bicubic', used for interpolate patch embed from 14x14 to 16x16

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    # encoder로 시작하는 파라미터들은 앞에 encoder 떼어버림
    if sorted(list(state_dict.keys()))[0].startswith("encoder"):
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

    if ("patch_embed.proj.weight" in state_dict) and ("patch_embed.proj.weight" in model.state_dict()):
        proj_weight = state_dict["patch_embed.proj.weight"]
        orig_size = proj_weight.shape[2:]
        current_size = model.patch_embed.proj.weight.shape[2:]
        if orig_size != current_size:
            padding_size = current_size[0] - orig_size[0]
            padding_l = padding_size // 2
            padding_r = padding_size - padding_l
            if "pad" in patch_padding:
                proj_weight = torch.nn.functional.pad(
                    proj_weight, (padding_l, padding_r, padding_l, padding_r)
                )
            elif "bilinear" in patch_padding:
                proj_weight = torch.nn.functional.interpolate(
                    proj_weight, size=current_size, mode="bilinear", align_corners=False
                )
            elif "bicubic" in patch_padding:
                proj_weight = torch.nn.functional.interpolate(
                    proj_weight, size=current_size, mode="bicubic", align_corners=False
                )
            state_dict["patch_embed.proj.weight"] = proj_weight

    if ("pos_embed" in state_dict) and ("pos_embed" in model.state_dict()):
        pos_embed_checkpoint = state_dict["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        H, W = model.patch_embed.patch_shape
        n_patches = model.patch_embed.n_patches
        num_extra_tokens = model.pos_embed.shape[-2] - n_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(H, W), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict["pos_embed"] = new_pos_embed

    new_state_dict = copy.deepcopy(state_dict)
    
    # MoE를 위한 수정 (expert 빼고 공통 fc만 가져옴 (아마도?))
    if part_features is not None:
        current_keys = list(model.state_dict().keys())
        for key in current_keys:
            if "mlp.experts" in key:
                source_key = re.sub(r"experts.\d+.", "fc2.", key)
                new_state_dict[key] = state_dict[source_key][-part_features:]
            elif "fc2" in key:
                new_state_dict[key] = state_dict[key][:-part_features]

    # load state_dict
    load_state_dict(model, new_state_dict, strict, logger)
    
    print("mae success!!")
    return checkpoint 