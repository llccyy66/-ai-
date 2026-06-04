# AI 视频生成工作流系统

一个基于 AI 的视频内容生成工作流系统，支持视频脚本生成、图片生成、语音合成等功能。

## 功能特性

- 🎬 **视频脚本生成** - 基于大语言模型自动生成视频脚本
- 🖼️ **图片生成** - 支持多种图片生成模型
- 🔊 **语音合成** - 支持多个 TTS 服务商
- ⚡ **工作流引擎** - 自动化的任务调度和执行
- 🌐 **Web API** - RESTful API 接口
- 🖥️ **图形界面** - 桌面端 GUI 应用

## 技术栈

- Python 3.10+
- Flask - Web 框架
- MiniMax - TTS 服务
- 各种 AI 模型集成

## 快速开始

### 环境要求

```bash
Python >= 3.10
pip >= 23.0
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加您的 API 密钥：

```env
# 硅基流动 API 配置
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
SILICONFLOW_API_URL=https://api.siliconflow.cn/v1/chat/completions

# ModelScope API 配置
MODELSCOPE_TOKEN=your_modelscope_token_here

# MiniMax TTS API 配置
MINIMAX_API_KEY=your_minimax_api_key_here
```

### 配置文件

编辑 `config.yaml` 配置应用参数。

### 启动服务

**启动 Web API**
```bash
python run_api.py
```

**启动 GUI 界面**
```bash
python run_gui.py
```

**启动任务调度器**
```bash
python run_scheduler.py
```

## 项目结构

```
├── config.yaml          # 应用配置文件
├── .env.example         # 环境变量示例
├── requirements.txt     # Python 依赖
├── run_api.py           # API 启动脚本
├── run_gui.py           # GUI 启动脚本
├── run_scheduler.py     # 调度器启动脚本
└── README.md            # 项目说明
```

## API 接口

### 工作流接口

- `POST /api/workflow/run` - 执行工作流
- `GET /api/workflow/status/{task_id}` - 查询任务状态

### 模板接口

- `GET /api/templates` - 获取模板列表
- `POST /api/templates` - 创建模板

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 注意事项

- 请确保在上传到 GitHub 前移除所有硬编码的 API 密钥
- 敏感配置应通过环境变量或 `.env` 文件管理
- `.env` 文件已添加到 `.gitignore`，不会被提交
