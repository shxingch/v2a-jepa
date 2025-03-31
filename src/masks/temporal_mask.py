# 创建新文件 src/masks/temporal_mask.py

from multiprocessing import Value
from logging import getLogger

import torch
import numpy as np

_GLOBAL_SEED = 0
logger = getLogger()

class TemporalMaskCollator(object):
    """
    生成时间维度掩码：当前帧保持不变，预测下一帧
    """
    
    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
        predict_next_ratio=1.0,  # 下一时间步预测比例：1.0表示全部掩码
        spatial_mask_ratio=0.0,  # 可选的空间掩码比例
    ):
        super(TemporalMaskCollator, self).__init__()
        
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
            
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // patch_size[0], crop_size[1] // patch_size[1]
        self.duration = num_frames // tubelet_size
        
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_patches_spatial = self.height * self.width
        self.num_patches_total = self.num_patches_spatial * self.duration
        
        # 预测时间窗口参数
        self.predict_next_ratio = predict_next_ratio
        self.spatial_mask_ratio = spatial_mask_ratio
        
        # 计算时间分割点
        self.time_split = self.duration // 2
        
        # 迭代计数器
        self._itr_counter = Value('i', -1)
    
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def __call__(self, batch):
        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)
        
        collated_masks_pred, collated_masks_enc = [], []
        
        for _ in range(batch_size):
            # 生成用于编码器的掩码（当前帧）
            # 全部不掩码或部分掩码
            mask_e = torch.ones(self.num_patches_total, dtype=torch.int32)
            
            # 可选地对当前帧应用部分空间掩码
            if self.spatial_mask_ratio > 0:
                num_to_mask = int(self.num_patches_spatial * self.spatial_mask_ratio)
                mask_indices = np.random.choice(
                    self.num_patches_spatial, 
                    size=num_to_mask, 
                    replace=False
                )
                for t in range(self.time_split):
                    start_idx = t * self.num_patches_spatial
                    for idx in mask_indices:
                        mask_e[start_idx + idx] = 0
            
            # 生成用于预测器的掩码（下一帧）
            mask_p = torch.zeros(self.num_patches_total, dtype=torch.int32)
            
            # 对下一帧应用掩码（全部或部分）
            num_to_mask = int(self.num_patches_spatial * self.predict_next_ratio)
            if num_to_mask > 0:
                mask_indices = np.random.choice(
                    self.num_patches_spatial, 
                    size=num_to_mask, 
                    replace=False
                )
                for t in range(self.time_split, self.duration):
                    start_idx = t * self.num_patches_spatial
                    for idx in mask_indices:
                        mask_p[start_idx + idx] = 1
            
            # 转换为索引格式
            mask_e = torch.nonzero(mask_e).squeeze()
            mask_p = torch.nonzero(mask_p).squeeze()
            
            collated_masks_enc.append(mask_e)
            collated_masks_pred.append(mask_p)
        
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        
        return collated_batch, collated_masks_enc, collated_masks_pred