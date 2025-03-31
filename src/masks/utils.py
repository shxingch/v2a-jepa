# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    :param concat: whether to concatenate the outputs or return a list
    """
    if not isinstance(masks, list):
        masks = [masks]
    
    all_x = []
    B, N, D = x.shape
    
    for mask in masks:
        # 确保掩码索引不超出范围
        safe_mask = torch.clamp(mask, 0, N-1)
        
        # 如果掩码被截断，打印警告
        if not torch.equal(mask, safe_mask):
            print(f"警告: 掩码索引超出范围，已被截断。原始范围: [{mask.min()}, {mask.max()}], 有效范围: [0, {N-1}]")
        
        # 使用安全的掩码索引
        mask_keep = safe_mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)
