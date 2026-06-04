#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-ASR 优化配置模块
提供方案1（移动缓存目录）和方案2（预热模型）的组合优化
"""

import os
import sys

# === 缓存目录配置 ===
# 优先级: 环境变量 HF_HOME > 配置文件 > 默认值
DEFAULT_HF_CACHE_DIR = "D:/huggingface_cache"
FALLBACK_CACHE_DIR = "E:/huggingface_cache"


def setup_huggingface_cache(custom_cache_dir=None):
    """
    方案1: 设置HuggingFace缓存目录到空间充足的磁盘
    
    Args:
        custom_cache_dir: 自定义缓存目录路径
        
    Returns:
        str: 实际使用的缓存目录
    """
    # 1. 检查环境变量
    if os.environ.get("HF_HOME"):
        hf_cache_dir = os.environ["HF_HOME"]
        print(f"[Qwen3-Config] 使用环境变量 HF_HOME: {hf_cache_dir}")
        return hf_cache_dir
    
    # 2. 使用自定义路径
    if custom_cache_dir:
        hf_cache_dir = custom_cache_dir
    # 3. 优先使用D盘
    elif os.path.exists("D:/"):
        hf_cache_dir = DEFAULT_HF_CACHE_DIR
    # 4. 备选E盘
    elif os.path.exists("E:/"):
        hf_cache_dir = FALLBACK_CACHE_DIR
    # 5. 默认回到用户目录
    else:
        hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    
    # 创建目录
    os.makedirs(hf_cache_dir, exist_ok=True)
    
    # 设置环境变量
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    print(f"[Qwen3-Config] HuggingFace缓存目录已设置: {hf_cache_dir}")
    return hf_cache_dir


def preload_model_files(hf_cache_dir=None):
    """
    方案2: 预热模型文件到系统缓存
    
    在模型正式加载前，预先读取模型文件头到内存，
    强制操作系统将文件缓存到page cache中，
    减少后续加载时的磁盘I/O等待。
    
    Args:
        hf_cache_dir: HuggingFace缓存目录
    """
    if hf_cache_dir is None:
        hf_cache_dir = os.environ.get("HF_HOME", DEFAULT_HF_CACHE_DIR)
    
    print("[Qwen3-Config] 开始预热模型文件...")
    
    try:
        from transformers.utils.hub import cached_file
        import glob
        
        models = [
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ForcedAligner-0.6B"
        ]
        
        for model_name in models:
            try:
                # 获取配置文件路径（会自动下载到缓存）
                config_path = cached_file(model_name, "config.json")
                if config_path:
                    print(f"[Qwen3-Config] 预加载配置: {model_name}")
                
                # 查找safetensors文件
                model_cache_pattern = f"models--{model_name.replace('/', '--')}"
                safetensors_pattern = os.path.join(
                    hf_cache_dir,
                    "**",
                    model_cache_pattern,
                    "**",
                    "*.safetensors"
                )
                
                safetensors_files = glob.glob(safetensors_pattern, recursive=True)
                
                for safetensor_file in safetensors_files[:2]:  # 最多预热2个分片
                    if os.path.exists(safetensor_file):
                        # 读取文件头1MB到内存，触发page cache
                        with open(safetensor_file, 'rb') as f:
                            f.read(1024 * 1024)
                        print(f"[Qwen3-Config] 预热模型文件: {os.path.basename(safetensor_file)}")
                        
            except Exception as e:
                print(f"[Qwen3-Config] 预热 {model_name} 失败: {e}")
                
        print("[Qwen3-Config] 模型预热完成")
        
    except ImportError:
        print("[Qwen3-Config] transformers库未安装，跳过预热")
    except Exception as e:
        print(f"[Qwen3-Config] 预加载过程跳过: {e}")


def optimize_disk_usage():
    """
    综合优化: 方案1 + 方案2
    
    1. 设置缓存目录到空间充足的磁盘
    2. 预热模型文件到系统缓存
    
    Returns:
        str: 使用的缓存目录
    """
    print("[Qwen3-Config] 启动综合优化: 方案1 + 方案2")
    
    # 方案1: 移动缓存目录
    cache_dir = setup_huggingface_cache()
    
    # 方案2: 预热模型文件（延迟到实际使用时执行，避免启动耗时）
    
    return cache_dir


if __name__ == "__main__":
    # 单独运行时执行完整优化
    optimize_disk_usage()