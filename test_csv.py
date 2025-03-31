# 创建CSV文件的脚本
import os
import glob

def create_libero_csv(data_dir, output_path):
    """
    创建包含所有Libero HDF5文件路径的CSV文件
    
    Args:
        data_dir: Libero数据集目录
        output_path: 输出CSV文件路径
    """
    # 收集所有HDF5文件
    hdf5_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    
    # 写入CSV文件
    with open(output_path, 'w') as f:
        for file_path in hdf5_files:
            # 使用文件名作为虚拟类标签，这对于无监督训练无关紧要
            f.write(f"{file_path} 0\n")
    
    print(f"Created CSV file at {output_path} with {len(hdf5_files)} entries")

# 示例使用
data_dir = "/cephfs/chenshuaixing/data/datasets/libero_10/"
output_path = "/cephfs/chenshuaixing/data/datasets/libero_files.csv"
create_libero_csv(data_dir, output_path)