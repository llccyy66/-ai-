import sys
import json
import os
import subprocess

DEFAULT_HF_CACHE_DIR = "D:/huggingface_cache"


def _setup_huggingface_cache(force_offline=False):
    if os.path.exists("D:/"):
        hf_cache_dir = DEFAULT_HF_CACHE_DIR
    elif os.path.exists("E:/"):
        hf_cache_dir = "E:/huggingface_cache"
    else:
        hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    
    os.makedirs(hf_cache_dir, exist_ok=True)
    
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    
    if force_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.pop("HF_ENDPOINT", None)
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
        print(f"[Qwen3-Align] HuggingFace缓存目录已设置(离线模式): {hf_cache_dir}", file=sys.stderr)
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        print(f"[Qwen3-Align] HuggingFace缓存目录已设置: {hf_cache_dir}", file=sys.stderr)
    
    return hf_cache_dir


def _preload_model_files():
    try:
        import glob
        
        model_dirs = [
            os.path.join(DEFAULT_HF_CACHE_DIR, "transformers", "models--Qwen--Qwen3-ASR-1.7B"),
            os.path.join(DEFAULT_HF_CACHE_DIR, "transformers", "models--Qwen--Qwen3-ForcedAligner-0.6B")
        ]
        
        for model_dir in model_dirs:
            try:
                config_path = os.path.join(model_dir, "config.json")
                if os.path.exists(config_path):
                    print(f"[Qwen3-Align] 找到配置文件: {os.path.basename(model_dir)}", file=sys.stderr)
                
                safetensors_files = glob.glob(os.path.join(
                    model_dir,
                    "**",
                    "*.safetensors"
                ))
                
                for safetensor_file in safetensors_files[:2]:
                    if os.path.exists(safetensor_file):
                        with open(safetensor_file, 'rb') as f:
                            f.read(1024 * 1024)
                        print(f"[Qwen3-Align] 预热模型文件: {os.path.basename(safetensor_file)}", file=sys.stderr)
                        
            except Exception as e:
                print(f"[Qwen3-Align] 预热 {os.path.basename(model_dir)} 失败: {e}", file=sys.stderr)
                
    except Exception as e:
        print(f"[Qwen3-Align] 预加载过程跳过: {e}", file=sys.stderr)


_setup_huggingface_cache(force_offline=True)

import warnings
warnings.filterwarnings("ignore")

import gc
import re


def _split_text_by_punctuation(text):
    sentences = []
    current = ""
    for ch in text:
        current += ch
        if ch in '。！？；\n':
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    
    if not sentences:
        sentences = [text]
    
    final = []
    for s in sentences:
        while len(s) > 20:
            split_pos = 15
            for j in range(15, min(20, len(s))):
                if s[j] in '，、：':
                    split_pos = j + 1
                    break
            final.append(s[:split_pos])
            s = s[split_pos:]
        if s:
            final.append(s)
    
    return final


def _merge_char_timestamps(items, original_text="", max_chars_per_subtitle=15):
    char_ts = []
    for item in items:
        t = item.text.strip() if hasattr(item, 'text') else str(item)
        start = float(item.start_time) if hasattr(item, 'start_time') else 0.0
        end = float(item.end_time) if hasattr(item, 'end_time') else 0.0
        if t:
            char_ts.append({"text": t, "start": start, "end": end})

    if not char_ts:
        return []

    sentences = _split_text_by_punctuation(original_text)
    
    total_chars = sum(len(s.replace('，', '').replace('。', '').replace('！', '').replace('？', '').replace('；', '').replace('、', '').replace('：', '')) for s in sentences)
    asr_chars = len(char_ts)
    
    if total_chars == 0 or asr_chars == 0:
        return [{"text": original_text, "start": char_ts[0]["start"], "end": char_ts[-1]["end"]}]
    
    char_ratio = asr_chars / total_chars
    
    merged = []
    char_idx = 0
    
    for sent in sentences:
        sent_clean_len = len(sent.replace('，', '').replace('。', '').replace('！', '').replace('？', '').replace('；', '').replace('、', '').replace('：', ''))
        expected_asr_chars = max(1, round(sent_clean_len * char_ratio))
        
        end_idx = min(char_idx + expected_asr_chars, len(char_ts))
        
        if end_idx <= char_idx:
            if merged:
                merged[-1]["text"] += sent
                continue
            end_idx = min(char_idx + 1, len(char_ts))
        
        if char_idx >= len(char_ts):
            if merged:
                merged[-1]["text"] += sent
            continue
        
        start_time = char_ts[char_idx]["start"]
        end_time = char_ts[end_idx - 1]["end"] if end_idx > char_idx else char_ts[char_idx]["end"]
        
        merged.append({
            "text": sent,
            "start": start_time,
            "end": end_time
        })
        char_idx = end_idx
    
    while char_idx < len(char_ts):
        last_end = char_ts[char_idx]["end"]
        if merged:
            merged[-1]["end"] = last_end
        char_idx += 1
    
    return merged


def _download_progress_hook(repo_id, filename, bytes_downloaded, bytes_total):
    """
    下载进度回调函数，显示下载状态
    """
    if bytes_total > 0:
        progress = (bytes_downloaded / bytes_total) * 100
        # 格式化显示大小
        def format_size(bytes_size):
            if bytes_size < 1024:
                return f"{bytes_size} B"
            elif bytes_size < 1024 * 1024:
                return f"{bytes_size / 1024:.1f} KB"
            elif bytes_size < 1024 * 1024 * 1024:
                return f"{bytes_size / (1024 * 1024):.1f} MB"
            else:
                return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"
        
        downloaded = format_size(bytes_downloaded)
        total = format_size(bytes_total)
        print(f"[Qwen3-Align] 正在下载: {filename} | {downloaded}/{total} ({progress:.1f}%)", file=sys.stderr)
    else:
        print(f"[Qwen3-Align] 正在下载: {filename} | {bytes_downloaded} bytes", file=sys.stderr)


def _load_model_with_fallback():
    """
    加载Qwen3-ASR模型，支持离线/在线模式自动切换
    
    先尝试离线模式加载，如果失败则切换到在线模式下载模型。
    使用4-bit量化，RTX 4060 8GB显卡可运行。
    """
    import torch
    from qwen_asr import Qwen3ASRModel
    from accelerate import init_empty_weights
    from accelerate.utils import set_seed
    
    asr_model_dir = os.path.join(DEFAULT_HF_CACHE_DIR, "transformers", "models--Qwen--Qwen3-ASR-1.7B")
    aligner_model_dir = os.path.join(DEFAULT_HF_CACHE_DIR, "transformers", "models--Qwen--Qwen3-ForcedAligner-0.6B")
    
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.pop("HF_ENDPOINT", None)
    
    print(f"[Qwen3-Align] ASR模型目录: {asr_model_dir}", file=sys.stderr)
    print(f"[Qwen3-Align] 对齐器模型目录: {aligner_model_dir}", file=sys.stderr)
    
    if not os.path.exists(asr_model_dir):
        raise Exception(f"ASR模型目录不存在: {asr_model_dir}")
    if not os.path.exists(aligner_model_dir):
        raise Exception(f"对齐器模型目录不存在: {aligner_model_dir}")
    
    strategies = [
        {
            "name": "4-bit量化（推荐）",
            "kwargs": {
                "device_map": "auto",
                "load_in_4bit": True,
                "local_files_only": True,
                "max_memory": {0: "6GB", "cpu": "10GB"}
            }
        },
        {
            "name": "float16精度",
            "kwargs": {
                "device_map": "auto",
                "dtype": torch.float16,
                "local_files_only": True,
                "max_memory": {0: "6GB", "cpu": "10GB"}
            }
        },
        {
            "name": "CPU模式（较慢）",
            "kwargs": {
                "device_map": "cpu",
                "local_files_only": True
            }
        }
    ]
    
    for strategy in strategies:
        try:
            print(f"[Qwen3-Align] 尝试{strategy['name']}加载模型...", file=sys.stderr)
            model = Qwen3ASRModel.from_pretrained(
                asr_model_dir,
                forced_aligner=aligner_model_dir,
                **strategy["kwargs"]
            )
            print(f"[Qwen3-Align] {strategy['name']}加载成功", file=sys.stderr)
            return model
        except Exception as e:
            print(f"[Qwen3-Align] {strategy['name']}加载失败: {str(e)[:150]}", file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    raise Exception("所有加载策略均失败，请检查显存或模型文件")


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "need input json path"}))
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2] if len(sys.argv) >= 3 else None

    def _output(data):
        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            print(json.dumps(data, ensure_ascii=False))

    if not os.path.exists(input_json_path):
        _output({"success": False, "error": f"input file not found: {input_json_path}"})
        sys.exit(1)

    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception as e:
        _output({"success": False, "error": f"read input failed: {e}"})
        sys.exit(1)

    try:
        print("[Qwen3-Align] 开始预热模型文件...", file=sys.stderr)
        _preload_model_files()
        
        import torch
        from qwen_asr import Qwen3ASRModel

        print("[Qwen3-Align] 加载Qwen3-ASR模型...", file=sys.stderr)
        model = _load_model_with_fallback()
        print("[Qwen3-Align] 模型加载完成", file=sys.stderr)

        results = []
        for seg in segments:
            audio_path = seg["audio_path"]
            text = seg["text"]
            index = seg["index"]

            if not os.path.exists(audio_path):
                results.append({
                    "index": index,
                    "success": False,
                    "error": f"audio file not found: {audio_path}"
                })
                continue

            try:
                result = model.forced_aligner.align(audio_path, text, language="zh")

                if result and len(result) > 0:
                    align_result = result[0]
                    if hasattr(align_result, 'items') and align_result.items:
                        timestamps = _merge_char_timestamps(align_result.items, original_text=text)
                        aligned_text_parts = [ts["text"] for ts in timestamps]

                        results.append({
                            "index": index,
                            "success": True,
                            "timestamps": timestamps,
                            "aligned_text": "".join(aligned_text_parts)
                        })

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                results.append({
                    "index": index,
                    "success": False,
                    "error": "alignment result empty"
                })

            except Exception as e:
                results.append({
                    "index": index,
                    "success": False,
                    "error": str(e)
                })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output = {"success": True, "results": results}
        _output(output)

    except Exception as e:
        _output({"success": False, "error": str(e)})


if __name__ == "__main__":
    main()