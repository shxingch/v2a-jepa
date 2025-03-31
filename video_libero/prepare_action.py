import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 定义数据路径
base_dir = "/cephfs/chenshuaixing/data/libero_img/libero_10"
output_dir = "/cephfs/chenshuaixing/data/libero_action_data"
os.makedirs(output_dir, exist_ok=True)

def process_action_data():
    # 创建字典来存储每个视频的动作序列
    action_data = {}
    
    # 遍历所有任务目录
    for task_dir in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
            
        print(f"处理任务: {task_dir}")
        
        # 遍历每个任务下的所有演示
        for demo_id in tqdm(sorted(os.listdir(task_path))):
            demo_path = os.path.join(task_path, demo_id)
            if not os.path.isdir(demo_path):
                continue
                
            # 构建视频ID (任务名_演示ID)
            video_id = f"{task_dir}/{demo_id}"
            
            # 读取action.json文件
            action_file = os.path.join(demo_path, "action.json")
            if os.path.exists(action_file):
                with open(action_file, 'r') as f:
                    action_data_raw = json.load(f)
                
                # 将action数据转换为numpy数组
                # 假设action数据是一个包含动作向量的列表
                action_array = np.array(action_data_raw)
                
                # 存储到字典中
                action_data[video_id] = action_array
    
    # 保存处理后的动作数据
    action_data_path = os.path.join(output_dir, "libero_10_actions.npy")
    np.save(action_data_path, action_data)
    print(f"动作数据已保存至: {action_data_path}")
    
    return action_data

def create_action_paths_csv():
    """创建包含视频路径和对应动作路径的CSV文件"""
    # 读取你之前创建的训练和测试集CSV
    train_csv_path = "/cephfs/chenshuaixing/data/libero_video/libero_10/libero_train.csv"
    test_csv_path = "/cephfs/chenshuaixing/data/libero_video/libero_10/libero_test.csv"
    
    # 创建新的CSV，添加动作路径列
    action_data_path = os.path.join(output_dir, "libero_10_actions.npy")
    
    # 处理训练集
    if os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path, header=None, sep=" ")
        train_df[2] = action_data_path  # 添加动作数据路径列
        new_train_csv = os.path.join(output_dir, "libero_train_with_actions.csv")
        train_df.to_csv(new_train_csv, header=False, index=False, sep=" ")
        print(f"训练集带动作路径CSV已保存至: {new_train_csv}")
    
    # 处理测试集
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path, header=None, sep=" ")
        test_df[2] = action_data_path  # 添加动作数据路径列
        new_test_csv = os.path.join(output_dir, "libero_test_with_actions.csv")
        test_df.to_csv(new_test_csv, header=False, index=False, sep=" ")
        print(f"测试集带动作路径CSV已保存至: {new_test_csv}")

# 主执行逻辑
if __name__ == "__main__":
    # 处理动作数据
    action_data = process_action_data()
    
    # 打印一些统计信息
    print(f"处理的视频数量: {len(action_data)}")
    
    # 为一个示例视频打印动作维度
    sample_key = next(iter(action_data.keys()))
    sample_action = action_data[sample_key]
    print(f"示例视频 {sample_key} 的动作形状: {sample_action.shape}")
    
    # 创建包含动作路径的CSV
    create_action_paths_csv()