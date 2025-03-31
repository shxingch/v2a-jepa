#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p ./logs

# 生成时间戳格式的日志文件名
LOG_FILE="./logs/libero_vjepa_$(date +%Y%m%d_%H%M%S).log"

# 输出开始信息
echo "Starting command with nohup: python -m app.main_libero --config configs/pretrain/libero_vjepa.yaml"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"

# 使用nohup运行命令，确保在退出终端后继续运行
nohup python -m app.main_libero --config configs/pretrain/libero_vjepa.yaml > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!

# 输出进程ID信息
echo "Process started with PID: $PID"
echo "You can check the log file with: tail -f $LOG_FILE"
echo "To kill the process later if needed: kill $PID"