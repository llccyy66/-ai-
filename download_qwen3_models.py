"""
Qwen3-ASR 和 Qwen3-ForcedAligner 模型下载脚本

使用方法：
    python download_qwen3_models.py

模型将下载到 D:/huggingface_cache 目录
"""

import os
import sys

# 设置缓存目录
HF_CACHE_DIR = "D:/huggingface_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# 设置环境变量
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内镜像加速

print(f"[下载] 缓存目录: {HF_CACHE_DIR}")
print(f"[下载] 使用镜像: {os.environ['HF_ENDPOINT']}")
print("=" * 60)


def download_model(model_name, model_type="model"):
    """
    下载单个模型
    """
    print(f"\n[下载] 开始下载: {model_name}")
    print(f"[下载] 模型类型: {model_type}")
    
    try:
        from huggingface_hub import snapshot_download
        
        local_dir = os.path.join(
            HF_CACHE_DIR, 
            "transformers", 
            f"models--{model_name.replace('/', '--')}"
        )
        
        print(f"[下载] 目标路径: {local_dir}")
        
        # 下载模型
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False,
            etag_timeout=30,
        )
        
        print(f"[OK] {model_name} 下载完成!")
        return True
        
    except Exception as e:
        print(f"[ERROR] {model_name} 下载失败: {e}")
        return False


def main():
    models = [
        ("Qwen/Qwen3-ASR-1.7B", "ASR语音识别模型 (~2.4GB)"),
        ("Qwen/Qwen3-ForcedAligner-0.6B", "强制对齐模型 (~1.2GB)"),
    ]
    
    print("\n" + "=" * 60)
    print("Qwen3 模型下载工具")
    print("=" * 60)
    print("\n待下载模型列表:")
    for i, (name, desc) in enumerate(models, 1):
        print(f"  {i}. {name} - {desc}")
    print("\n总大小约 3.6GB，请确保网络稳定")
    print("=" * 60 + "\n")
    
    success_count = 0
    for model_name, desc in models:
        if download_model(model_name, desc):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"[完成] 成功下载 {success_count}/{len(models)} 个模型")
    
    if success_count == len(models):
        print("[OK] 所有模型下载完成，可以正常使用离线模式!")
    else:
        print("[WARNING] 部分模型下载失败，请检查网络后重试")
    print("=" * 60)


if __name__ == "__main__":
    main()
