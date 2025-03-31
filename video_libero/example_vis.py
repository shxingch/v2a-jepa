import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import os
import pandas as pd
from collections import defaultdict
import sys
from decord import VideoReader, cpu
import cv2
import random

# 添加项目路径以导入V-JEPA相关模块
sys.path.append("/root/jepa")
from src.models.vision_transformer import vit_base
from src.models.utils.multimask import MultiMaskWrapper
from src.models.utils.patch_embed import PatchEmbed3D

# 配置
batch_size = 4
num_frames = 16
tubelet_size = 2
patch_size = 16
crop_size = 224
feature_dim = 768  # ViT-Base的特征维度

# 加载CSV文件
train_csv = "/cephfs/chenshuaixing/data/libero_video/libero_10/libero_train.csv"
test_csv = "/cephfs/chenshuaixing/data/libero_video/libero_10/libero_test.csv"

# 读取视频路径
def load_video_paths(csv_file):
    paths_labels = []
    with open(csv_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(' ')
            if len(parts) < 2:
                continue
                
            path, label = parts[0], parts[1]
            # 从路径中提取场景编号
            if 'KITCHEN_SCENE3' in path:
                scene = 3
            elif 'KITCHEN_SCENE4' in path:
                scene = 4
            elif 'KITCHEN_SCENE6' in path:
                scene = 6
            elif 'KITCHEN_SCENE8' in path:
                scene = 8
            else:
                continue
            paths_labels.append((path, scene))
    return paths_labels

# 视频预处理函数
def preprocess_video(video_path, num_frames=16, sample_rate=2):
    try:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        
        # 采样帧
        total_frames = len(vr)
        indices = np.linspace(0, total_frames-1, num_frames*sample_rate)
        indices = indices[::sample_rate][:num_frames]
        indices = np.clip(indices, 0, total_frames-1).astype(np.int64)
        
        # 获取帧
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        
        # 调整大小并进行中心裁剪
        processed_frames = []
        for frame in frames:
            # 调整为目标尺寸并保持纵横比
            h, w, _ = frame.shape
            if h > w:
                new_h, new_w = int(crop_size * h / w), crop_size
            else:
                new_h, new_w = crop_size, int(crop_size * w / h)
            
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 中心裁剪
            y_offset = (new_h - crop_size) // 2
            x_offset = (new_w - crop_size) // 2
            frame = frame[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
            
            # 确保尺寸正确
            if frame.shape[0] != crop_size or frame.shape[1] != crop_size:
                frame = cv2.resize(frame, (crop_size, crop_size))
                
            processed_frames.append(frame)
        
        # 转换为张量格式
        video_tensor = np.stack(processed_frames)  # [T, H, W, C]
        video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))  # [C, T, H, W]
        video_tensor = video_tensor / 255.0  # 归一化到[0,1]
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        return torch.FloatTensor(video_tensor)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

# 修复状态字典中的键名问题
def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除"module."前缀
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# 构建模型
def build_model(checkpoint_path):
    # 创建模型架构
    encoder = vit_base(
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
    )
    encoder = MultiMaskWrapper(encoder)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 使用目标编码器，它通常更稳定
    if 'target_encoder' in checkpoint:
        encoder_state = checkpoint['target_encoder']
    elif 'encoder' in checkpoint:
        encoder_state = checkpoint['encoder']
    else:
        raise ValueError(f"No encoder found in checkpoint. Keys: {checkpoint.keys()}")
    
    # 修复状态字典
    fixed_state_dict = fix_state_dict(encoder_state)
    
    # 加载权重
    encoder.load_state_dict(fixed_state_dict)
    
    # 设置为评估模式
    encoder.eval()
    
    return encoder

# 提取特征
def extract_features():
    # 加载模型
    model_path = "/cephfs/chenshuaixing/vjepa_logs/libero_kitchen/jepa_libero-latest.pth.tar"
    print(f"加载模型: {model_path}")
    model = build_model(model_path)
    
    # 加载视频路径
    print("加载视频路径...")
    train_paths = load_video_paths(train_csv)
    test_paths = load_video_paths(test_csv)
    print(f"找到 {len(train_paths)} 个训练视频和 {len(test_paths)} 个测试视频")
    
    all_paths = train_paths + test_paths
    
    # 限制视频数量以加快处理速度
    max_videos = 100
    if len(all_paths) > max_videos:
        random.seed(42)
        all_paths = random.sample(all_paths, max_videos)
    
    print(f"处理 {len(all_paths)} 个视频...")
    
    # 初始化存储
    all_features = []
    all_labels = []
    
    # 批处理提取特征
    with torch.no_grad():
        for i, (video_path, label) in enumerate(all_paths):
            if i % 10 == 0:
                print(f"处理视频 {i+1}/{len(all_paths)}: {os.path.basename(video_path)}")
            
            # 预处理视频
            video_tensor = preprocess_video(video_path)
            if video_tensor is None:
                continue
                
            # 添加批次维度
            video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
            
            # 提取特征
            features = model(video_tensor)  # [1, N, D]
            
            # 获取全局特征通过平均池化
            pooled_features = features.mean(dim=1)  # [1, D]
            
            # 保存特征和标签
            all_features.append(pooled_features.cpu().numpy())
            all_labels.append(label)
    
    # 转换为NumPy数组
    if all_features:
        features_array = np.vstack(all_features)
        labels_array = np.array(all_labels)
        
        print(f"提取了 {features_array.shape[0]} 个视频的特征")
        return features_array, labels_array
    else:
        print("没有成功提取任何特征!")
        return np.array([]), np.array([])

# 主脚本
def main():
    print("开始提取特征...")
    features, labels = extract_features()
    
    if len(features) == 0:
        print("没有特征可以可视化，退出程序")
        return
        
    print(f"提取完成! 特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    # 使用t-SNE降维
    print("执行t-SNE降维...")
    perplexity = min(30, len(features) - 1)
    print(f"使用perplexity={perplexity}")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)
    
    # 获取唯一场景标签
    unique_labels = np.unique(labels)
    
    # 可视化
    plt.figure(figsize=(12, 10))
    colors = ['red', 'blue', 'green', 'purple']
    cmap = ListedColormap(colors[:len(unique_labels)])
    
    # 绘制散点图
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, 
                          alpha=0.8, s=100, edgecolors='k')
    
    # 添加图例
    legend_labels = [f'Scene {label}' for label in unique_labels]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, fontsize=12)
    
    plt.title('t-SNE Visualization of V-JEPA Features', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    save_path = '/cephfs/chenshuaixing/vjepa_logs/libero_kitchen/feature_viz.png'
    plt.savefig(save_path, dpi=300)
    print(f"可视化已保存到: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()