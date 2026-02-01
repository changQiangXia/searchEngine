<div align="center">

# 🔮 NexusMind

**下一代多模态语义搜索引擎**

[![CI](https://github.com/changQiangXia/searchEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/changQiangXia/searchEngine/actions)
[![Docker](https://github.com/changQiangXia/searchEngine/actions/workflows/docker.yml/badge.svg)](https://github.com/changQiangXia/searchEngine/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- TODO: 发布到 PyPI 后更新此链接 [![PyPI](https://img.shields.io/pypi/v/nexus-mind.svg)](https://pypi.org/project/nexus-mind/) -->

**[中文](#-中文文档) | [English](#-english-documentation)**

</div>

---

<details open>
<summary><h2>🇨🇳 中文文档</h2></summary>

## 📖 目录

- [简介](#-简介)
- [核心特性](#-核心特性)
- [快速开始](#-快速开始)
- [使用方法](#-使用方法)
- [GPU内存安全](#-gpu内存安全)
- [系统架构](#-系统架构)
- [性能表现](#-性能表现)
- [部署方式](#-部署方式)
- [开发指南](#-开发指南)
- [更新日志](#-更新日志)

---

## 🎯 简介

**NexusMind** 是一个基于 CLIP 和 FAISS 构建的下一代多模态语义搜索引擎，专为消费级 GPU（如 RTX 3080ti 12GB）优化，具备智能内存管理和自动降级机制。

### 为什么选择 NexusMind？

- 🚀 **开箱即用** - 一行命令启动，无需复杂配置
- 🛡️ **内存安全** - 三级熔断机制，12GB显存也能畅玩大模型
- 🎨 **可视化** - 3D语义星系、概念插值漫游等酷炫功能
- 🔌 **可扩展** - 插件架构，轻松扩展功能
- 💪 **高性能** - 支持INT4量化，动态批处理优化

---

## ✨ 核心特性

### 搜索能力
| 功能 | 描述 | 命令示例 |
|------|------|----------|
| 🔍 **语义搜索** | 文本搜图、以图搜图 | `nexus search "夕阳"` |
| 🚫 **负面搜索** | 排除不想要的内容 | `nexus negative "海滩" "人群"` |
| 🎲 **多样性排序** | MMR算法平衡相关性和多样性 | `--diverse` |
| 🎭 **概念插值** | 发现概念间的中间态 | `nexus interpolate "复古" "未来"` |
| 🔄 **概念混合** | 多概念加权融合 | `nexus blend "圆形:0.6" "红色:0.4"` |
| 🔗 **跨模态链** | Image→Text→Image 探索链 | `nexus chain "cat.jpg" 4` |

### 可视化
| 功能 | 描述 |
|------|------|
| 🌌 **语义星系** | 3D降维可视化，探索语义空间 |
| 🎭 **概念漫游** | 逐步展示概念插值过程 |
| 🔥 **注意力热力图** | 查看CLIP关注的图像区域 |

### 性能优化
| 功能 | 效果 |
|------|------|
| ⚡ **模型量化** | FP16(2x) / INT8(4x) / INT4(8x) 内存节省 |
| 🚀 **动态批处理** | 自动调整batch size，避免OOM |
| 💾 **三级缓存** | L1(GPU) / L2(SSD) / L3(Disk) |
| 📊 **性能监控** | 实时吞吐量、延迟监控 |

---

## 🚀 快速开始

### 方式一：pip安装（推荐）

```bash
# 安装
pip install nexus-mind

# 验证
nexus status
```

### 方式二：Docker（含GPU支持）

```bash
# 运行Web界面
docker run --gpus all -p 8501:8501 \
  ghcr.io/changqiangxia/searchengine:latest

# 访问 http://localhost:8501
```

### 方式三：源码安装

```bash
git clone https://github.com/changQiangXia/searchEngine.git
cd nexus-mind
pip install -e ".[all]"
```

---

## 📚 使用方法

### CLI命令行

```bash
# 1. 索引图像
nexus index ./photos --recursive

# 2. 文本搜索
nexus search "夕阳下的山脉"

# 3. 以图搜图
nexus search ./query.jpg

# 4. 负面搜索（不要人群）
nexus negative "夕阳海滩" "人群"

# 5. 概念插值（复古→未来）
nexus interpolate "复古" "未来" --steps 5

# 6. 概念混合（60%圆形 + 40%红色）
nexus blend "圆形:0.6" "红色:0.4"

# 7. 查看系统状态
nexus status
```

### Python API

```python
from nexus_mind import NexusEngine

# 初始化引擎
engine = NexusEngine()

# 索引图像
engine.index_images(["./photos"])

# 搜索
results = engine.search("可爱的猫咪")
for r in results:
    print(f"{r['metadata']['path']}: {r['score']:.3f}")

# 概念插值
path = engine.interpolate_concepts("猫", "老虎", steps=5)

# 跨模态链式探索
chain = engine.explore_chain("start.jpg", steps=4)
```

### Web界面

```bash
# 启动
nexus-web
# 或: cd apps/web && ./launch.sh

# 访问 http://localhost:8501
```

⚠️ **重要提示：Web界面需要设置Workspace路径**

如果你之前在CLI中创建了索引，需要在Web侧边栏中指定相同的工作空间：

1. 看网页**左侧边栏**的 **"📁 Workspace"** 部分
2. 输入之前CLI使用的工作空间路径（例如：`./my_workspace`）
3. **按回车键**确认
4. 看到 **"✅ Index: X vectors"** 表示加载成功

或者设置环境变量启动：
```bash
export NEXUS_WORKSPACE_DIR=./my_workspace
nexus-web
```

**Web界面功能：**
- 🔍 **搜索页面** - 支持文本/图像搜索，负面搜索，多样性排序
- 🌌 **语义星系** - 3D可视化，支持PCA/t-SNE/UMAP降维
- 🎭 **概念探索** - 交互式概念插值漫游
- 🔥 **注意力图** - 查看CLIP注意力热力图
- 📊 **系统统计** - 实时监控GPU内存和性能

---

## 🛡️ GPU内存安全

NexusMind专为**有限显存**优化，在RTX 3080ti (12GB)上也能流畅运行：

| GPU | CLIP | 索引 | 量化支持 |
|-----|------|------|----------|
| RTX 3080ti (12GB) | FP16 (~0.9GB) | GPU/CPU混合 | INT8/INT4 |
| RTX 4090 (24GB) | FP16 (~0.9GB) | 完整GPU | INT8/INT4 |

### 三级熔断保护

系统自动监控GPU内存，三级保护机制：

```
⚠️ WARNING  (>60%)  → 清理缓存
🔴 CRITICAL (>80%)  → 卸载非核心模型
💥 EMERGENCY (>90%) → 强制降级到CPU
```

**查看内存状态：**
```python
from nexus_mind.infrastructure.memory.manager import get_memory_manager

manager = get_memory_manager()
print(manager.get_stats())
# 输出: MemoryStats(gpu_used=2.5GB, gpu_total=12.0GB, usage=20.8%)
```

---

## 🏗️ 系统架构

```
nexus_mind/
├── core/                 # 核心引擎层
│   └── engine.py        # 主引擎
├── infrastructure/       # 基础设施层
│   ├── models/          # CLIP + 量化
│   ├── storage/         # FAISS + 缓存
│   ├── memory/          # GPU内存管理 ⭐
│   └── compute/         # 性能优化
├── application/         # 应用层
│   ├── use_cases/       # 搜索/插值/聚类
│   └── workflow/        # 工作流
├── plugins/             # 插件系统
│   ├── base.py          # 插件基类
│   └── builtin/         # 内置插件
└── interfaces/          # 接口层
    ├── cli/             # 命令行
    └── web/             # Web界面
```

---

## 📊 性能表现

RTX 3080ti (12GB) 实测数据：

| 数据集规模 | 索引时间 | 搜索延迟 | 显存占用 |
|-----------|---------|---------|---------|
| 1,000张 | 5秒 | 10ms | 0.5GB |
| 10,000张 | 45秒 | 15ms | 1.2GB |
| 100,000张 | 8分钟 | 50ms | 4GB (转CPU) |

**运行基准测试：**
```bash
python tools/benchmark.py --image-dir ./photos
```

---

## 🐳 部署方式

### Docker Compose（推荐）

```yaml
version: '3.8'
services:
  nexus-mind:
    image: ghcr.io/changqiangxia/searchengine:latest
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

启动：
```bash
docker-compose up -d
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `NEXUS_WORKSPACE_DIR` | 工作空间目录 | `./data/workspaces` |
| `NEXUS_CACHE_DIR` | 缓存目录 | `./data/cache` |
| `NEXUS_LOG_LEVEL` | 日志级别 | `INFO` |
| `CUDA_VISIBLE_DEVICES` | GPU选择 | `0` |

---

## 🛠️ 开发指南

```bash
# 克隆仓库
git clone https://github.com/changQiangXia/searchEngine.git
cd nexus-mind

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/unit -v

# 代码检查
ruff check src/
black src/
mypy src/nexus_mind/

# 构建Docker
 docker build -t searchengine:latest .
```

---

## 📝 更新日志

### [0.1.0] - 2024-02-01

**新增功能：**
- ✅ 核心语义搜索（CLIP + FAISS）
- ✅ GPU内存三级熔断保护
- ✅ CLI命令行界面
- ✅ 概念插值（SLERP/LEERP）
- ✅ 概念混合（多概念加权）
- ✅ 负面搜索（语义排除）
- ✅ MMR多样性排序
- ✅ 语义聚类（KMeans/HDBSCAN）
- ✅ 跨模态链式推理
- ✅ Streamlit Web界面
- ✅ 3D语义星系可视化
- ✅ 注意力热力图
- ✅ 插件系统架构
- ✅ 三级缓存（L1/L2/L3）
- ✅ 模型量化（FP16/INT8/INT4）
- ✅ 动态批处理优化
- ✅ 性能监控
- ✅ Docker支持
- ✅ CI/CD自动化

---

## 📄 许可证

[MIT License](LICENSE) © 2024 NexusMind Team

**[⬆ 回到顶部](#-nexusmind)**

</details>

---

<details>
<summary><h2>🇺🇸 English Documentation</h2></summary>

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [GPU Memory Safety](#-gpu-memory-safety)
- [Architecture](#-architecture)
- [Performance](#-performance)
- [Deployment](#-deployment)
- [Development](#-development)
- [Changelog](#-changelog)

---

## 🎯 Introduction

**NexusMind** is a next-generation multimodal semantic search engine built on CLIP and FAISS, optimized for consumer GPUs (like RTX 3080ti 12GB) with intelligent memory management and automatic fallback mechanisms.

### Why NexusMind?

- 🚀 **Out-of-the-box** - One command to start, no complex configuration
- 🛡️ **Memory Safe** - Three-level circuit breaker, run large models on 12GB VRAM
- 🎨 **Visualization** - 3D semantic galaxy, concept interpolation, and more
- 🔌 **Extensible** - Plugin architecture for easy feature expansion
- 💪 **High Performance** - INT4 quantization, dynamic batching optimization

---

## ✨ Features

### Search Capabilities
| Feature | Description | Example |
|---------|-------------|---------|
| 🔍 **Semantic Search** | Text-to-image, image-to-image | `nexus search "sunset"` |
| 🚫 **Negative Search** | Exclude unwanted content | `nexus negative "beach" "people"` |
| 🎲 **Diverse Results** | MMR algorithm balances relevance and diversity | `--diverse` |
| 🎭 **Concept Interpolation** | Discover intermediate concepts | `nexus interpolate "vintage" "futuristic"` |
| 🔄 **Concept Blending** | Multi-concept weighted fusion | `nexus blend "circle:0.6" "red:0.4"` |
| 🔗 **Cross-Modal Chain** | Image→Text→Image exploration | `nexus chain "cat.jpg" 4` |

### Visualization
| Feature | Description |
|---------|-------------|
| 🌌 **Semantic Galaxy** | 3D dimensionality reduction visualization |
| 🎭 **Concept Explorer** | Step-by-step concept interpolation |
| 🔥 **Attention Heatmap** | View CLIP attention regions |

### Performance Optimization
| Feature | Effect |
|---------|--------|
| ⚡ **Model Quantization** | FP16(2x) / INT8(4x) / INT4(8x) memory savings |
| 🚀 **Dynamic Batching** | Auto-adjust batch size, avoid OOM |
| 💾 **Tiered Cache** | L1(GPU) / L2(SSD) / L3(Disk) |
| 📊 **Performance Monitor** | Real-time throughput and latency tracking |

---

## 🚀 Quick Start

### Option 1: pip Install (Recommended)

```bash
# Install
pip install nexus-mind

# Verify
nexus status
```

### Option 2: Docker (with GPU Support)

```bash
# Run Web UI
docker run --gpus all -p 8501:8501 \
  ghcr.io/changqiangxia/searchengine:latest

# Access http://localhost:8501
```

### Option 3: Source Install

```bash
git clone https://github.com/changQiangXia/searchEngine.git
cd nexus-mind
pip install -e ".[all]"
```

---

## 📚 Usage

### CLI Commands

```bash
# 1. Index images
nexus index ./photos --recursive

# 2. Text search
nexus search "sunset over mountains"

# 3. Image search
nexus search ./query.jpg

# 4. Negative search (exclude people)
nexus negative "sunset beach" "people"

# 5. Concept interpolation (vintage→futuristic)
nexus interpolate "vintage" "futuristic" --steps 5

# 6. Concept blending (60% circle + 40% red)
nexus blend "circle:0.6" "red:0.4"

# 7. Check system status
nexus status
```

### Python API

```python
from nexus_mind import NexusEngine

# Initialize engine
engine = NexusEngine()

# Index images
engine.index_images(["./photos"])

# Search
results = engine.search("cute cat")
for r in results:
    print(f"{r['metadata']['path']}: {r['score']:.3f}")

# Concept interpolation
path = engine.interpolate_concepts("cat", "tiger", steps=5)

# Cross-modal chain exploration
chain = engine.explore_chain("start.jpg", steps=4)
```

### Web Interface

```bash
# Start
nexus-web
# Or: cd apps/web && ./launch.sh

# Access http://localhost:8501
```

⚠️ **Important: Web Interface Requires Workspace Configuration**

If you created an index using CLI, you need to specify the same workspace in the Web UI sidebar:

1. Look at the **left sidebar** in the web page, find **"📁 Workspace"**
2. Enter the workspace path used in CLI (e.g., `./my_workspace`)
3. **Press Enter** to confirm
4. You should see **"✅ Index: X vectors"** indicating successful loading

Or set environment variable before starting:
```bash
export NEXUS_WORKSPACE_DIR=./my_workspace
nexus-web
```

**Web Interface Features:**
- 🔍 **Search Page** - Text/image search, negative search, diversity ranking
- 🌌 **Semantic Galaxy** - 3D visualization with PCA/t-SNE/UMAP
- 🎭 **Concept Explorer** - Interactive concept interpolation
- 🔥 **Attention Map** - View CLIP attention heatmap
- 📊 **System Stats** - Real-time GPU memory and performance monitoring

---

## 🛡️ GPU Memory Safety

NexusMind is optimized for **limited VRAM**, running smoothly on RTX 3080ti (12GB):

| GPU | CLIP | Index | Quantization |
|-----|------|-------|--------------|
| RTX 3080ti (12GB) | FP16 (~0.9GB) | GPU/CPU hybrid | INT8/INT4 |
| RTX 4090 (24GB) | FP16 (~0.9GB) | Full GPU | INT8/INT4 |

### Three-Level Circuit Breaker

System automatically monitors GPU memory with three-level protection:

```
⚠️ WARNING  (>60%)  → Clean cache
🔴 CRITICAL (>80%)  → Offload non-essential models
💥 EMERGENCY (>90%) → Force fallback to CPU
```

**Check memory status:**
```python
from nexus_mind.infrastructure.memory.manager import get_memory_manager

manager = get_memory_manager()
print(manager.get_stats())
# Output: MemoryStats(gpu_used=2.5GB, gpu_total=12.0GB, usage=20.8%)
```

---

## 🏗️ Architecture

```
nexus_mind/
├── core/                 # Core engine layer
│   └── engine.py        # Main engine
├── infrastructure/       # Infrastructure layer
│   ├── models/          # CLIP + quantization
│   ├── storage/         # FAISS + cache
│   ├── memory/          # GPU memory management ⭐
│   └── compute/         # Performance optimization
├── application/         # Application layer
│   ├── use_cases/       # Search/interpolation/clustering
│   └── workflow/        # Workflows
├── plugins/             # Plugin system
│   ├── base.py          # Plugin base classes
│   └── builtin/         # Built-in plugins
└── interfaces/          # Interface layer
    ├── cli/             # Command line
    └── web/             # Web interface
```

---

## 📊 Performance

Benchmarks on RTX 3080ti (12GB):

| Dataset Size | Index Time | Search Latency | VRAM Usage |
|-------------|------------|----------------|------------|
| 1,000 images | 5s | 10ms | 0.5GB |
| 10,000 images | 45s | 15ms | 1.2GB |
| 100,000 images | 8min | 50ms | 4GB (CPU fallback) |

**Run benchmarks:**
```bash
python tools/benchmark.py --image-dir ./photos
```

---

## 🐳 Deployment

### Docker Compose (Recommended)

```yaml
version: '3.8'
services:
  nexus-mind:
    image: ghcr.io/changqiangxia/searchengine:latest
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Start:
```bash
docker-compose up -d
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXUS_WORKSPACE_DIR` | Workspace directory | `./data/workspaces` |
| `NEXUS_CACHE_DIR` | Cache directory | `./data/cache` |
| `NEXUS_LOG_LEVEL` | Log level | `INFO` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` |

---

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/changQiangXia/searchEngine.git
cd nexus-mind

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit -v

# Code linting
ruff check src/
black src/
mypy src/nexus_mind/

# Build Docker
docker build -t nexus-mind:latest .
```

---

## 📝 Changelog

### [0.1.0] - 2024-02-01

**New Features:**
- ✅ Core semantic search (CLIP + FAISS)
- ✅ GPU memory three-level circuit breaker
- ✅ CLI interface
- ✅ Concept interpolation (SLERP/LEERP)
- ✅ Concept blending (multi-concept weighted)
- ✅ Negative search (semantic exclusion)
- ✅ MMR diversity ranking
- ✅ Semantic clustering (KMeans/HDBSCAN)
- ✅ Cross-modal chain reasoning
- ✅ Streamlit web interface
- ✅ 3D semantic galaxy visualization
- ✅ Attention heatmap
- ✅ Plugin system architecture
- ✅ Tiered cache (L1/L2/L3)
- ✅ Model quantization (FP16/INT8/INT4)
- ✅ Dynamic batching optimization
- ✅ Performance monitoring
- ✅ Docker support
- ✅ CI/CD automation

---

## 📄 License

[MIT License](LICENSE) © 2024 NexusMind Team

**[⬆ Back to Top](#-nexusmind)**

</details>

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md).

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) - Vision-language model
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Streamlit](https://streamlit.io/) - Web interface

---

<div align="center">

Made with ❤️ by the NexusMind Team

**[中文](#-中文文档) | [English](#-english-documentation)**

</div>