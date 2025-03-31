# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

class ActionPredictor(nn.Module):
    """动作预测模型，将V-JEPA的视觉表征转换为机器人动作。
    
    Args:
        input_dim (int): 输入特征维度 (视觉编码器的输出维度)
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出动作维度 (默认7，对应Libero动作空间)
        num_layers (int): MLP层数
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=7, num_layers=2):
        super().__init__()
        
        # 创建MLP网络
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        # 输出层
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 视觉特征，形状为 [B, N, D]，其中:
               B = 批量大小
               N = 时间步数
               D = 特征维度
               
        Returns:
            动作序列，形状为 [B, N, output_dim]
        """
        return self.mlp(x)


class DynamicPredictor(nn.Module):
    """动态预测模型，预测下一个时间步的潜在向量。
    
    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出特征维度 (通常与input_dim相同)
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
            
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 当前时间步的特征，形状为 [B, D]
               
        Returns:
            下一个时间步的预测特征，形状为 [B, D]
        """
        return self.mlp(x)