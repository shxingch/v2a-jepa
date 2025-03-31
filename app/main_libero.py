# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import argparse
import yaml
from app.vjepa.train_libero import main as train_main

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume_preempt', action='store_true', help='Resume from preemption')
    args = parser.parse_args()
    
    # 读取配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 调用训练函数
    train_main(args=config, resume_preempt=args.resume_preempt)