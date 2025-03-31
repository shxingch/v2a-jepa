import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import glob

class LiberoDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, frame_skip=4, transform=None, view_type='agentview'):
        """
        Libero数据集加载器
        
        Args:
            data_dir: Libero HDF5文件的目录
            num_frames: 每个视频序列的帧数
            frame_skip: 连续帧之间的跳帧数
            transform: 视频帧的转换函数
            view_type: 使用哪种视图 ('agentview' 或 'eye_in_hand')
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.transform = transform
        self.view_type = view_type + '_rgb'
        
        # 收集所有HDF5文件
        self.hdf5_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.hdf5'):
                    self.hdf5_files.append(os.path.join(root, file))
        
        # 构建索引映射
        self.demo_mapping = []  # 存储(file_idx, demo_idx, start_idx)元组
        
        for file_idx, file_path in enumerate(self.hdf5_files):
            with h5py.File(file_path, 'r') as f:
                demos = list(f['data'].keys())
                for demo_idx, demo_key in enumerate(demos):
                    # 获取该演示的序列长度
                    seq_length = f['data'][demo_key]['actions'].shape[0]
                    
                    # 计算可以提取的视频片段数
                    max_start_idx = seq_length - num_frames * frame_skip
                    
                    if max_start_idx > 0:
                        # 以固定步长采样起始帧
                        sample_stride = max(1, max_start_idx // 10)  # 每个演示取约10个样本
                        for start_idx in range(0, max_start_idx, sample_stride):
                            self.demo_mapping.append((file_idx, demo_key, start_idx))
        
        print(f"共加载了{len(self.hdf5_files)}个HDF5文件，{len(self.demo_mapping)}个视频片段")
        
    def __len__(self):
        return len(self.demo_mapping)
    
    def __getitem__(self, idx):
        file_idx, demo_key, start_idx = self.demo_mapping[idx]
        file_path = self.hdf5_files[file_idx]
        
        with h5py.File(file_path, 'r') as f:
            demo_data = f['data'][demo_key]
            
            # 提取视频帧
            frames = []
            for i in range(self.num_frames):
                frame_idx = start_idx + i * self.frame_skip
                frame = demo_data['obs'][self.view_type][frame_idx]
                frames.append(frame)
            
            # 提取对应的动作和状态
            actions = []
            ee_states = []
            for i in range(self.num_frames):
                frame_idx = start_idx + i * self.frame_skip
                action = demo_data['actions'][frame_idx]
                ee_state = demo_data['obs']['ee_states'][frame_idx]
                actions.append(action)
                ee_states.append(ee_state)
            
            # 从文件名中提取任务描述
            task_desc = os.path.basename(file_path).split('_demo')[0]
            
            # 转换为张量
            frames = np.stack(frames)  # (T, H, W, C)
            
            # 应用变换 - 在转置前应用变换
            if self.transform:
                frames = self.transform(frames)
            
            # 检查 frames 的类型并相应地进行转置
            if isinstance(frames, np.ndarray):
                # 如果是 NumPy 数组，使用 NumPy 的 transpose
                frames = frames.transpose(0, 3, 1, 2)  # 转为(T, C, H, W)格式
                frames = torch.from_numpy(frames).float()
            else:
                # 如果已经是 PyTorch 张量，使用 PyTorch 的 permute
                frames = frames.permute(0, 3, 1, 2)  # 转为(T, C, H, W)格式
            
            # 归一化到[0,1]
            if frames.max() > 1.0:
                frames = frames / 255.0
            
            # 确保动作和状态张量的形状正确
            actions = np.stack(actions)  # (T, action_dim)
            ee_states = np.stack(ee_states)  # (T, state_dim)
            
            # 转换为PyTorch张量
            actions = torch.from_numpy(actions).float()
            ee_states = torch.from_numpy(ee_states).float()
            
            return {
                'frames': frames,
                'actions': actions,
                'ee_states': ee_states,
                'task_desc': task_desc,
                'file_path': file_path,
                'demo_key': demo_key
            }
            
def make_liberodataset(
    data_dir,
    batch_size,
    num_frames=16,
    frame_skip=4,
    transform=None,
    view_type='agentview',
    collator=None,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    pin_memory=True,
    persistent_workers=False,
):
    """
    创建Libero数据集和数据加载器
    
    Args:
        data_dir: HDF5文件目录
        batch_size: 每个批次的样本数
        num_frames: 每个视频片段的帧数
        frame_skip: 连续帧之间的跳帧数
        transform: 视频帧转换
        view_type: 使用的视角类型 ('agentview' 或 'eye_in_hand')
        collator: 用于合并样本的函数
        num_workers: 数据加载的工作进程数
        world_size: 分布式训练的总进程数
        rank: 当前进程的rank
        drop_last: 是否丢弃不足一个批次的样本
        pin_memory: 是否将数据固定在内存中加速GPU传输
        persistent_workers: 是否保持工作进程活跃
        
    Returns:
        dataset: Libero数据集实例
        data_loader: 数据加载器
        sampler: 分布式采样器
    """
    
    # 创建数据集
    dataset = LiberoDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        frame_skip=frame_skip,
        transform=transform,
        view_type=view_type
    )
    
    # 创建分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=drop_last
    )
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collator
    )
    
    return dataset, data_loader, sampler