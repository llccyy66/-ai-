#!/usr/bin/env python3
"""启动脚本 - 任务调度器"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.scheduler.scheduler import get_scheduler

if __name__ == '__main__':
    print("启动任务调度器...")
    
    try:
        scheduler = get_scheduler()
        scheduler.start()
        
        print("任务调度器已启动")
        print("按 Ctrl+C 停止服务")
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止任务调度器...")
        scheduler.stop()
        print("任务调度器已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)