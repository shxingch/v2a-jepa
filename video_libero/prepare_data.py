import os
import pandas as pd

# 定义数据路径
data_dir = "/cephfs/chenshuaixing/data/libero_video/libero_10"

# 创建训练集索引文件
train_files = []
for filename in os.listdir(data_dir):
    # 只选择场景3、4和6的视频
    if any(scene in filename for scene in ["KITCHEN_SCENE3", "KITCHEN_SCENE4", "KITCHEN_SCENE6"]):
        abs_path = os.path.join(data_dir, filename)
        train_files.append([abs_path, 0])  # 标签设为0（预训练不关心标签）

# 创建测试集索引文件
test_files = []
for filename in os.listdir(data_dir):
    # 只选择场景8的视频
    if "KITCHEN_SCENE8" in filename:
        abs_path = os.path.join(data_dir, filename)
        test_files.append([abs_path, 0])

# 保存训练集索引
train_df = pd.DataFrame(train_files)
train_csv_path = os.path.join(data_dir, "libero_train.csv")
train_df.to_csv(train_csv_path, header=False, index=False, sep=" ")

# 保存测试集索引
test_df = pd.DataFrame(test_files)
test_csv_path = os.path.join(data_dir, "libero_test.csv")
test_df.to_csv(test_csv_path, header=False, index=False, sep=" ")

print(f"训练集索引保存至: {train_csv_path}")
print(f"测试集索引保存至: {test_csv_path}")