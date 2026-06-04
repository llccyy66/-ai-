#!/usr/bin/env python3
"""启动脚本 - GUI应用"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui.main import VideoWorkflowGUI
import tkinter as tk

if __name__ == '__main__':
    print("启动GUI应用...")
    
    try:
        root = tk.Tk()
        app = VideoWorkflowGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)