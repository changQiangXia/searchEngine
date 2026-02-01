# FAISS-GPU 安装指南

NexusMind 默认使用 `faiss-cpu`，但在生产环境推荐使用 `faiss-gpu` 以获得更快的索引和搜索性能。

## 快速安装

```bash
# 方式1: 使用安装脚本（推荐）
bash scripts/install_faiss_gpu.sh

# 方式2: 手动安装
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 pytorch-cuda=11.8

# 验证安装
python scripts/verify_faiss_gpu.py
```

## 安装前准备

### 1. 检查 CUDA 版本

```bash
nvcc --version
# 或
nvidia-smi
```

NexusMind 已在以下 CUDA 版本测试：
- ✅ CUDA 11.8 (推荐，适用于 RTX 3080ti)
- ✅ CUDA 12.1

### 2. 确保 Conda 已安装

FAISS-GPU 最稳定的安装方式是通过 Conda：

```bash
# 检查 conda
conda --version

# 如果没有安装，下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## 显存需求

| 索引类型 | 10万向量 | 100万向量 | 备注 |
|---------|---------|----------|------|
| Flat    | ~300MB  | ~3GB     | 精确搜索，最快 |
| IVF     | ~330MB  | ~3.3GB   | 平衡速度/内存 |
| IVFPQ   | ~3MB    | ~30MB    | 压缩存储，适合大规模 |

**RTX 3080ti (12GB) 建议**：
- < 50万向量：使用 GPU Flat 或 IVF 索引
- 50万-500万向量：使用 GPU IVFPQ 索引
- > 500万向量：使用 CPU 索引或分片

## 验证安装

```bash
# 完整验证
python scripts/verify_faiss_gpu.py

# 快速检查
python -c "import faiss; print(f'GPUs: {faiss.get_num_gpus()}')"
```

## 使用 GPU 加速

安装完成后，NexusMind 会自动检测并使用 GPU：

```bash
# CLI 会自动使用 GPU（如果可用）
nexus index ./photos
nexus search "sunset beach"

# Python API
from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend

backend = FAISSBackend(dim=768, use_gpu=True)
backend.build(embeddings, metadata)
```

## 故障排除

### 问题1: `faiss-gpu` 导入失败

```bash
# 症状: ImportError: libfaiss.so: cannot open shared object file

# 解决: 设置环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 或重新安装
pip uninstall faiss-cpu faiss-gpu -y
conda install -c pytorch faiss-gpu
```

### 问题2: GPU 检测不到

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查 FAISS GPU
python -c "import faiss; print(faiss.get_num_gpus())"
```

### 问题3: CUDA 版本不匹配

```bash
# 查看当前 CUDA 版本
nvcc --version

# 安装对应版本的 FAISS
# CUDA 11.8
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 pytorch-cuda=11.8

# CUDA 12.1
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 pytorch-cuda=12.1
```

### 问题4: 显存不足 (OOM)

NexusMind 会自动处理显存不足的情况：
- 自动降级到 CPU 索引
- 三级熔断保护（警告/临界/紧急）

手动强制使用 CPU：
```python
backend = FAISSBackend(dim=768, use_gpu=False)
```

## 性能对比

在 RTX 3080ti 上的典型性能：

| 操作 | CPU | GPU | 加速比 |
|-----|-----|-----|-------|
| 索引 10万张图 | 45s | 8s | 5.6x |
| 搜索 (100 queries) | 120ms | 15ms | 8x |
| 批量搜索 | 1.2s | 0.1s | 12x |

## 相关文件

- `scripts/install_faiss_gpu.sh` - 自动安装脚本
- `scripts/verify_faiss_gpu.py` - 验证工具
- `src/nexus_mind/infrastructure/storage/vector/faiss_backend.py` - GPU 支持代码
