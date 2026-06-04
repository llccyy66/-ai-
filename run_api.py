#!/usr/bin/env python3
"""启动脚本 - API服务"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import create_app, create_socketio

if __name__ == '__main__':
    app = create_app()
    socketio = create_socketio(app)
    
    print("启动Flask API服务...")
    print("访问地址: http://127.0.0.1:5000")
    print("按 Ctrl+C 停止服务")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)