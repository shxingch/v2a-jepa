import h5py
import os
import numpy as np
from pprint import pprint

def explore_hdf5_structure(file_path):
    """探索HDF5文件的结构并打印其内容概述"""
    print(f"正在分析文件: {os.path.basename(file_path)}")
    
    with h5py.File(file_path, 'r') as f:
        # 打印顶层键
        print("\n顶层键:")
        top_keys = list(f.keys())
        print(top_keys)
        
        # 递归探索结构
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"数据集: {name}, 形状: {obj.shape}, 类型: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"组: {name}")
        
        print("\n详细结构:")
        f.visititems(print_structure)
        
        # 检查第一个演示的具体数据
        if len(top_keys) > 0:
            first_demo = top_keys[0]
            print(f"\n第一个演示 '{first_demo}' 的内容:")
            
            demo_group = f[first_demo]
            demo_keys = list(demo_group.keys())
            print(f"子键: {demo_keys}")
            
            # 检查一些关键数据
            print("\n关键数据样本:")
            for key in demo_keys:
                if key in ['frames', 'actions', 'ee_states', 'language']:
                    data = demo_group[key]
                    print(f"\n{key}:")
                    print(f"  形状: {data.shape}")
                    print(f"  类型: {data.dtype}")
                    
                    # 如果是小型数据集，显示内容
                    if data.size < 10 or key == 'language':
                        try:
                            value = data[()]
                            if isinstance(value, bytes):
                                value = value.decode('utf-8')
                            print(f"  数据: {value}")
                        except:
                            print("  无法显示数据内容")
                    # 对于视频帧，显示一些统计信息
                    elif key == 'frames':
                        print(f"  帧数: {data.shape[0]}")
                        if len(data.shape) > 1:
                            print(f"  分辨率: {data.shape[1:]}")
                        print(f"  值范围: [{np.min(data[0])}, {np.max(data[0])}]")
                    # 对于其他数组，显示前几个元素
                    else:
                        try:
                            print(f"  前几个元素: {data[:2]}")
                        except:
                            print("  无法显示前几个元素")

# 使用示例
file_path = "/cephfs/chenshuaixing/data/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
explore_hdf5_structure(file_path)