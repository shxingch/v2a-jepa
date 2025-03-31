import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import seaborn as sns
from tqdm import tqdm

# LIBERO相关导入
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# 创建结果保存目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"libero_eval_results_{timestamp}"
os.makedirs(RESULTS_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
VIDEOS_DIR = os.path.join(RESULTS_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

# 随机策略
class RandomPolicy:
    def __init__(self, action_dim=7):
        self.action_dim = action_dim
    
    def __call__(self, observation):
        # 随机动作,范围[-1, 1]
        return np.random.uniform(-1, 1, self.action_dim)

def save_observation_image(obs, path):
    """保存观察图像"""
    if isinstance(obs, dict) and "agentview_image" in obs:
        # 如果观察是字典并包含图像
        img = obs["agentview_image"]
    elif isinstance(obs, np.ndarray) and len(obs.shape) == 3:
        # 如果观察本身是图像
        img = obs
    else:
        print(f"无法从观察中提取图像: {type(obs)}")
        return False, None
    
    # 转换为uint8类型
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # 保存图像
    try:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True, img
    except Exception as e:
        print(f"保存图像失败: {e}")
        return False, None

def create_episode_video(images_dir, video_path, fps=5):
    """将目录中的图像合成为MP4视频"""
    try:
        # 获取所有图像文件（包括初始状态和步骤图像）
        image_files = []
        # 首先添加初始状态图像
        if os.path.exists(os.path.join(images_dir, "initial_state.png")):
            image_files.append("initial_state.png")
        
        # 然后添加按步骤排序的图像
        step_images = sorted([f for f in os.listdir(images_dir) 
                             if f.startswith('step_') and f.endswith('.png')],
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
        image_files.extend(step_images)
        
        if not image_files:
            print(f"没有找到图像文件: {images_dir}")
            return False
        
        print(f"找到 {len(image_files)} 个图像文件用于生成视频")
        
        # 读取第一张图像获取尺寸
        first_img = cv2.imread(os.path.join(images_dir, image_files[0]))
        if first_img is None:
            print(f"无法读取图像: {os.path.join(images_dir, image_files[0])}")
            return False
            
        height, width, _ = first_img.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # 添加每一帧
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"警告: 无法读取图像 {img_path}，跳过")
                continue
            
            # 添加文本标签以显示步骤信息
            if "step_" in img_file:
                step_num = img_file.split("_")[1].split(".")[0]
                cv2.putText(img, f"Step {step_num}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif "initial" in img_file:
                cv2.putText(img, "Initial State", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 为了让短视频更有观察价值，每帧重复写入多次
            for _ in range(3):  # 每帧重复3次，延长观察时间
                video.write(img)
        
        # 释放视频写入器
        video.release()
        print(f"视频已保存: {video_path}")
        
        # 验证视频文件是否成功创建
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            return True
        else:
            print(f"视频文件创建失败或大小为0: {video_path}")
            return False
            
    except Exception as e:
        print(f"创建视频时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_task(task_suite, task_id, num_episodes=5):
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    print(f"=== 评估任务 {task_id}: {task_name} ===")
    print(f"描述: {task_description}")
    
    # 为此任务创建图像目录
    task_images_dir = os.path.join(IMAGES_DIR, f"task_{task_id}")
    os.makedirs(task_images_dir, exist_ok=True)
    
    # 环境参数
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
        "has_offscreen_renderer": True,
        "use_camera_obs": True
    }
    
    # 创建环境
    env = OffScreenRenderEnv(**env_args)
    
    # 获取初始状态
    init_states = task_suite.get_task_init_states(task_id)
    
    # 创建随机策略
    policy = RandomPolicy()
    
    # 评估结果
    results = {
        "task_id": task_id,
        "task_name": task_name,
        "task_description": task_description,
        "episodes": [],
        "success_rate": 0.0,
        "avg_reward": 0.0,
        "avg_steps": 0.0,
        "completion_time": 0.0,  # 新增: 完成任务的平均时间步
        "failed_attempts": 0,    # 新增: 失败尝试次数
        "action_efficiency": 0.0 # 新增: 动作效率(每步平均奖励)
    }
    
    successful_episodes = 0
    total_reward = 0
    total_steps = 0
    completion_times = []
    
    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}")
        
        # 重置环境
        obs = env.reset()
        
        # 设置初始状态(如果有的话)
        if init_states is not None and len(init_states) > 0:
            init_state_id = ep % len(init_states)  # 循环使用初始状态
            env.set_init_state(init_states[init_state_id])
            print(f"Using initial state {init_state_id}")
        
        # 保存初始观察图像
        ep_dir = os.path.join(task_images_dir, f"episode_{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        initial_img_path = os.path.join(ep_dir, "initial_state.png")
        saved, _ = save_observation_image(obs, initial_img_path)
        
        # 为此episode创建图像存储列表，用于后续生成视频
        episode_frames = []
        
        # 评估
        done = False
        episode_reward = 0
        step_count = 0
        rewards = []
        actions = []
        all_frames = []  # 记录所有帧
        
        # 设置最大步数,防止陷入死循环
        max_steps = 200
        
        # 保存初始帧
        initial_img_path = os.path.join(ep_dir, "initial_state.png")
        saved, img = save_observation_image(obs, initial_img_path)
        if saved:
            all_frames.append(initial_img_path)
        
        while not done and step_count < max_steps:
            # 使用策略获取动作
            action = policy(obs)
            actions.append(action.tolist())  # 记录动作
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))
            episode_reward += reward
            step_count += 1
            
            # 保存每一步的观察图像
            img_path = os.path.join(ep_dir, f"step_{step_count}.png")
            saved, img = save_observation_image(obs, img_path)
            if saved:
                all_frames.append(img_path)  # 添加到所有帧列表
        
        # 如果任务成功，记录完成时间
        if episode_reward > 0:
            completion_times.append(step_count)
        
        # 记录此次episode的结果
        episode_data = {
            "rewards": rewards,
            "actions": actions,
            "total_reward": float(episode_reward),
            "steps": step_count,
            "success": episode_reward > 0,
            "images_dir": ep_dir,
            "video_path": os.path.join(VIDEOS_DIR, f"task_{task_id}_episode_{ep}.mp4")
        }
        results["episodes"].append(episode_data)
        
        # 更新统计数据
        if episode_reward > 0:
            successful_episodes += 1
        total_reward += episode_reward
        total_steps += step_count
        
        print(f"Episode {ep+1} - Steps: {step_count}, Reward: {episode_reward}, Success: {episode_reward > 0}")
        
        # 生成本次episode的奖励曲线
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, marker='o')
        plt.title(f"Task {task_id} - Episode {ep+1} Rewards")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(ep_dir, "rewards_plot.png"))
        plt.close()
        
        # 生成任务执行视频
        video_path = os.path.join(VIDEOS_DIR, f"task_{task_id}_episode_{ep}.mp4")
        video_success = create_episode_video(ep_dir, video_path)
        
        if not video_success:
            print(f"警告: 为任务 {task_id} 的 episode {ep} 创建视频失败")
            
            # 尝试直接从收集的帧创建视频
            if len(all_frames) > 0:
                print(f"尝试使用 {len(all_frames)} 个直接收集的帧创建视频...")
                alternate_video_path = os.path.join(VIDEOS_DIR, f"task_{task_id}_episode_{ep}_alt.mp4")
                
                try:
                    # 读取第一张图像获取尺寸
                    first_img = cv2.imread(all_frames[0])
                    if first_img is not None:
                        height, width, _ = first_img.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video = cv2.VideoWriter(alternate_video_path, fourcc, 5, (width, height))
                        
                        for frame_path in all_frames:
                            img = cv2.imread(frame_path)
                            if img is not None:
                                # 添加多次同一帧以延长观察时间
                                for _ in range(3):
                                    video.write(img)
                        
                        video.release()
                        print(f"替代视频已保存: {alternate_video_path}")
                        video_path = alternate_video_path  # 更新视频路径
                    else:
                        print("无法读取第一帧图像")
                except Exception as e:
                    print(f"创建替代视频时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
    
    # 计算整体统计数据
    results["success_rate"] = successful_episodes / num_episodes
    results["avg_reward"] = total_reward / num_episodes
    results["avg_steps"] = total_steps / num_episodes
    results["failed_attempts"] = num_episodes - successful_episodes
    
    # 计算动作效率和平均完成时间
    if total_steps > 0:
        results["action_efficiency"] = total_reward / total_steps
    if len(completion_times) > 0:
        results["completion_time"] = sum(completion_times) / len(completion_times)
    
    # 关闭环境
    env.close()
    
    # 保存结果
    results_file = os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # 生成任务的汇总可视化
    visualize_task_results(results, task_id)
    
    print(f"Task {task_id} Results:")
    print(f"Success Rate: {results['success_rate']:.2f}")
    print(f"Average Reward: {results['avg_reward']:.4f}")
    print(f"Average Steps: {results['avg_steps']:.1f}")
    print(f"Average Completion Time: {results['completion_time']:.1f}")
    print(f"Results saved to {results_file}")
    
    return results

def visualize_task_results(results, task_id):
    """为单个任务生成详细可视化"""
    # 提取每个episode的成功/失败信息
    successes = [ep["success"] for ep in results["episodes"]]
    steps = [ep["steps"] for ep in results["episodes"]]
    
    # 创建成功/失败柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(successes)), [1 if s else 0 for s in successes], 
                   color=['green' if s else 'red' for s in successes])
    
    # 添加步骤数标签
    for i, (bar, step) in enumerate(zip(bars, steps)):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + 0.05, 
                 f"{step} steps", 
                 ha='center', va='bottom')
    
    plt.title(f"Task {task_id}: Success/Failure by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Success (1) / Failure (0)")
    plt.ylim(0, 1.3)
    plt.xticks(range(len(successes)), [f"Ep {i+1}" for i in range(len(successes))])
    plt.savefig(os.path.join(PLOTS_DIR, f"task_{task_id}_success_failure.png"))
    plt.close()
    
    # 如果有多个episode，创建奖励分布图
    if len(results["episodes"]) > 1:
        all_rewards = [ep["total_reward"] for ep in results["episodes"]]
        plt.figure(figsize=(10, 6))
        sns.histplot(all_rewards, kde=True)
        plt.title(f"Task {task_id}: Reward Distribution")
        plt.xlabel("Total Reward")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(PLOTS_DIR, f"task_{task_id}_reward_distribution.png"))
        plt.close()

def main():
    # 获取LIBERO-10基准
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10"
    task_suite = benchmark_dict[task_suite_name]()
    
    # 获取任务数量
    num_tasks = task_suite.n_tasks
    print(f"Found {num_tasks} tasks in {task_suite_name}")
    
    # 所有任务的汇总结果
    all_results = {
        "task_suite": task_suite_name,
        "timestamp": timestamp,
        "tasks": [],
        "overall_success_rate": 0.0,
        "overall_avg_reward": 0.0,
        "overall_avg_steps": 0.0,
        "overall_completion_time": 0.0,
        "overall_action_efficiency": 0.0
    }
    
    # 评估每个任务
    total_success_rate = 0.0
    total_avg_reward = 0.0
    total_avg_steps = 0.0
    total_completion_time = 0.0
    total_action_efficiency = 0.0
    valid_completion_times = 0
    
    for task_id in range(num_tasks):
        # 评估当前任务
        task_results = evaluate_task(task_suite, task_id, num_episodes=5)
        
        # 添加到汇总结果
        all_results["tasks"].append({
            "task_id": task_id,
            "task_name": task_results["task_name"],
            "success_rate": task_results["success_rate"],
            "avg_reward": task_results["avg_reward"],
            "avg_steps": task_results["avg_steps"],
            "completion_time": task_results["completion_time"],
            "action_efficiency": task_results["action_efficiency"]
        })
        
        # 更新总体统计数据
        total_success_rate += task_results["success_rate"]
        total_avg_reward += task_results["avg_reward"]
        total_avg_steps += task_results["avg_steps"]
        
        if task_results["completion_time"] > 0:
            total_completion_time += task_results["completion_time"]
            valid_completion_times += 1
        
        total_action_efficiency += task_results["action_efficiency"]
    
    # 计算整体统计数据
    all_results["overall_success_rate"] = total_success_rate / num_tasks
    all_results["overall_avg_reward"] = total_avg_reward / num_tasks
    all_results["overall_avg_steps"] = total_avg_steps / num_tasks
    
    if valid_completion_times > 0:
        all_results["overall_completion_time"] = total_completion_time / valid_completion_times
    
    all_results["overall_action_efficiency"] = total_action_efficiency / num_tasks
    
    # 保存汇总结果
    summary_file = os.path.join(RESULTS_DIR, f"{task_suite_name}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n=== Overall Results ===")
    print(f"Overall Success Rate: {all_results['overall_success_rate']:.2f}")
    print(f"Overall Average Reward: {all_results['overall_avg_reward']:.4f}")
    print(f"Overall Average Steps: {all_results['overall_avg_steps']:.1f}")
    if valid_completion_times > 0:
        print(f"Overall Completion Time: {all_results['overall_completion_time']:.1f}")
    print(f"Overall Action Efficiency: {all_results['overall_action_efficiency']:.4f}")
    print(f"Summary saved to {summary_file}")
    
    # 生成多个汇总可视化
    plot_results(all_results)

def plot_results(all_results):
    # 1. 成功率柱状图
    task_names = [task["task_name"] for task in all_results["tasks"]]
    success_rates = [task["success_rate"] for task in all_results["tasks"]]
    
    # 截取任务名称(可能太长)
    short_names = [name[:20] + "..." if len(name) > 20 else name for name in task_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(short_names)), success_rates, color='skyblue')
    
    # 添加任务ID作为文本标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 0.05, 
                 f"Task {all_results['tasks'][i]['task_id']}", 
                 ha='center', color='black', rotation=90)
    
    plt.title(f"Success Rates for {all_results['task_suite']} Tasks")
    plt.xlabel("Tasks")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.1)
    plt.xticks(range(len(short_names)), short_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{all_results['task_suite']}_success_rates.png"))
    plt.close()
    
    # 2. 平均步数柱状图
    avg_steps = [task["avg_steps"] for task in all_results["tasks"]]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(short_names)), avg_steps, color='lightgreen')
    plt.title(f"Average Steps for {all_results['task_suite']} Tasks")
    plt.xlabel("Tasks")
    plt.ylabel("Average Steps")
    plt.xticks(range(len(short_names)), short_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{all_results['task_suite']}_avg_steps.png"))
    plt.close()
    
    # 3. 平均奖励柱状图
    avg_rewards = [task["avg_reward"] for task in all_results["tasks"]]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(short_names)), avg_rewards, color='salmon')
    plt.title(f"Average Rewards for {all_results['task_suite']} Tasks")
    plt.xlabel("Tasks")
    plt.ylabel("Average Reward")
    plt.xticks(range(len(short_names)), short_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{all_results['task_suite']}_avg_rewards.png"))
    plt.close()
    
    # 4. 任务难度散点图 (成功率vs平均步数)
    plt.figure(figsize=(10, 8))
    plt.scatter(success_rates, avg_steps, s=100, alpha=0.7)
    
    # 添加任务ID标签
    for i, (x, y) in enumerate(zip(success_rates, avg_steps)):
        plt.annotate(f"Task {i}", (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.title(f"Task Difficulty Map: Success Rate vs Average Steps")
    plt.xlabel("Success Rate")
    plt.ylabel("Average Steps")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task_difficulty_map.png"))
    plt.close()
    
    # 5. 综合性能热图
    # 准备热图数据
    task_ids = [task["task_id"] for task in all_results["tasks"]]
    metrics = ["success_rate", "avg_reward", "avg_steps", "completion_time", "action_efficiency"]
    metric_names = ["Success Rate", "Avg Reward", "Avg Steps", "Completion Time", "Action Efficiency"]
    
    # 创建数据矩阵
    data = np.zeros((len(metrics), len(task_ids)))
    for i, metric in enumerate(metrics):
        for j, task in enumerate(all_results["tasks"]):
            data[i, j] = task[metric]
    
    # 对步数和完成时间进行规范化处理（越低越好）
    if len(task_ids) > 0:
        data[2, :] = 1 - (data[2, :] / np.max(data[2, :]) if np.max(data[2, :]) > 0 else 0)  # 步数
        if np.max(data[3, :]) > 0:
            data[3, :] = 1 - (data[3, :] / np.max(data[3, :]))  # 完成时间
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(data, annot=True, cmap="YlGnBu", 
                xticklabels=[f"Task {i}" for i in task_ids],
                yticklabels=metric_names)
    plt.title("Performance Metrics Across Tasks")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "performance_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    main()