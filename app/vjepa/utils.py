# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
import warnings
import yaml


import torch

import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
    )
from src.utils.tensors import trunc_normal_


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# 在utils.py中添加或修改load_checkpoint函数
def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')

    epoch = 0
    try:
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(
                f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}'
            )

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )

    # # 加载动作预测器
    # if action_predictor is not None and 'action_predictor' in checkpoint:
    #     logger.info(f'Loading action_predictor from {r_path}')
    #     try:
    #         action_predictor.load_state_dict(checkpoint['action_predictor'], strict=True)
    #     except Exception as e:
    #         logger.info(f'Unable to load action_predictor weights: {e}.')
    #         logger.info('Continuing with random action_predictor weights.')
    # elif action_predictor is not None:
    #     logger.info(f'No action_predictor weights available, continuing with random init.')

def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    encoder = MultiMaskWrapper(encoder)
    predictor = vit_pred.__dict__['vit_predictor'](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
    )
    predictor = PredictorMultiMaskWrapper(predictor)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    logger.info(f'Predictor number of parameters: {count_parameters(predictor)}')

    return encoder, predictor

def init_opt(encoder, predictor, wd, final_wd, start_lr, ref_lr, final_lr, iterations_per_epoch, warmup, num_epochs, ipe_scale, mixed_precision, betas, eps, action_predictor=None):
    # 合并所有参数
    param_groups = [
        {'params': [p for p in encoder.parameters() if p.requires_grad]},
        {'params': [p for p in predictor.parameters() if p.requires_grad]},
    ]
    
    # 添加动作预测器参数
    if action_predictor is not None:
        param_groups.append({'params': [p for p in action_predictor.parameters() if p.requires_grad]})
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=betas,
        lr=start_lr,
        weight_decay=wd,
        eps=eps)
    
    # 创建学习率调度器
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup * iterations_per_epoch,
        start_lr=start_lr,
        ref_lr=ref_lr,
        T_max=num_epochs * iterations_per_epoch * ipe_scale,
        final_lr=final_lr
    )
    
    # 创建权重衰减调度器
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        T_max=num_epochs * iterations_per_epoch * ipe_scale,
        final_wd=final_wd
    )
    
    # 创建梯度缩放器
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    return optimizer, scaler, scheduler, wd_scheduler

# def init_opt(
#     encoder,
#     predictor,
#     iterations_per_epoch,
#     start_lr,
#     ref_lr,
#     warmup,
#     num_epochs,
#     wd=1e-6,
#     final_wd=1e-6,
#     final_lr=0.0,
#     mixed_precision=False,
#     ipe_scale=1.25,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     zero_init_bias_wd=True,
# ):
#     param_groups = [
#         {
#             'params': (p for n, p in encoder.named_parameters()
#                        if ('bias' not in n) and (len(p.shape) != 1))
#         }, {
#             'params': (p for n, p in predictor.named_parameters()
#                        if ('bias' not in n) and (len(p.shape) != 1))
#         }, {
#             'params': (p for n, p in encoder.named_parameters()
#                        if ('bias' in n) or (len(p.shape) == 1)),
#             'WD_exclude': zero_init_bias_wd,
#             'weight_decay': 0,
#         }, {
#             'params': (p for n, p in predictor.named_parameters()
#                        if ('bias' in n) or (len(p.shape) == 1)),
#             'WD_exclude': zero_init_bias_wd,
#             'weight_decay': 0,
#         },
#     ]

#     logger.info('Using AdamW')
#     optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
#     scheduler = WarmupCosineSchedule(
#         optimizer,
#         warmup_steps=int(warmup * iterations_per_epoch),
#         start_lr=start_lr,
#         ref_lr=ref_lr,
#         final_lr=final_lr,
#         T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
#     )
#     wd_scheduler = CosineWDSchedule(
#         optimizer,
#         ref_wd=wd,
#         final_wd=final_wd,
#         T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
#     )
#     scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
#     return optimizer, scaler, scheduler, wd_scheduler

# 在文件末尾添加
def init_libero_model(encoder, action_predictor_cfg, device=None):
    """
    初始化Libero模型，添加动作预测头到V-JEPA编码器
    
    Args:
        encoder: V-JEPA编码器模型
        action_predictor_cfg: 动作预测头配置
        device: 使用的设备
        
    Returns:
        action_predictor: 动作预测模型
    """
    from src.models.action_predictor import ActionPredictor
    
    # 获取编码器输出维度
    encoder_output_dim = encoder.backbone.embed_dim
    
    # 创建动作预测头
    action_predictor = ActionPredictor(
        input_dim=encoder_output_dim,
        hidden_dim=action_predictor_cfg.get('hidden_dim', 512),
        output_dim=action_predictor_cfg.get('output_dim', 7),
        num_layers=action_predictor_cfg.get('num_layers', 2)
    )
    
    if device is not None:
        action_predictor = action_predictor.to(device)
        
    return action_predictor