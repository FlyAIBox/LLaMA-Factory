#!/usr/bin/env python3
"""
GPU内存清理脚本
用于在训练前清理GPU内存，确保有足够的显存空间
"""

import gc
import torch
import subprocess
import time

def clear_gpu_memory():
    """清理GPU内存"""
    print("正在清理GPU内存...")
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 强制垃圾回收
    gc.collect()
    
    print("GPU内存清理完成")

def check_gpu_memory():
    """检查GPU内存使用情况"""
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        print("当前GPU内存使用情况:")
        print(result.stdout)
    except FileNotFoundError:
        print("无法找到rocm-smi命令")

def kill_existing_processes():
    """杀死可能占用GPU的进程"""
    try:
        # 查找Python进程
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        python_processes = []
        for line in lines:
            if 'python' in line and 'llamafactory' in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    python_processes.append(pid)
        
        # 杀死相关进程
        for pid in python_processes:
            try:
                subprocess.run(['kill', '-9', pid], check=False)
                print(f"已杀死进程 {pid}")
            except:
                pass
                
    except Exception as e:
        print(f"清理进程时出错: {e}")

if __name__ == "__main__":
    print("开始清理GPU资源...")
    
    # 杀死可能的残留进程
    kill_existing_processes()
    
    # 等待一下
    time.sleep(2)
    
    # 清理GPU内存
    clear_gpu_memory()
    
    # 再等待一下
    time.sleep(2)
    
    # 检查内存状态
    check_gpu_memory()
    
    print("清理完成！") 