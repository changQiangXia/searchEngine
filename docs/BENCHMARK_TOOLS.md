# NexusMind 测试工具指南

## 快速参考

| 测试工具 | 规模 | 预计时间 | 用途 |
|---------|------|---------|------|
| `pytest tests/` | 单元测试 | 10s | 功能验证 |
| `verify_faiss_gpu.py` | 1K向量 | 5s | GPU验证 |
| `large_scale_test.py --scale 10000` | 1万向量 | 30s | 快速基准 |
| `large_scale_test.py --scale 100000` | 10万向量 | 3min | 标准基准 |
| `nprobe_tuner.py` | 可调 | 2min | nprobe调优 |
| `million_vector_benchmark.py` | 100万向量 | 15-20min | 极限测试 |

---

## 1. 单元测试

```bash
# 运行所有单元测试
pytest tests/ -v

# 仅 GPU 测试
pytest tests/unit/test_faiss_gpu.py -v

# 覆盖率报告
pytest tests/ --cov=src/nexus_mind --cov-report=html
```

---

## 2. GPU 验证

```bash
# 验证 FAISS-GPU 安装
python scripts/verify_faiss_gpu.py

# 预期输出:
# ✅ FAISS GPU 安装成功
# ✅ GPU 检测正常: 1 GPU(s)
# ✅ 速度提升: 29.1x
```

---

## 3. 大规模测试

### 3.1 基础规模测试 (1-10万向量)

```bash
# 1万向量快速测试
python tools/large_scale_test.py --scale 10000 --index-type ivf

# 5万向量标准测试
python tools/large_scale_test.py --scale 50000 --index-type ivf --output results_50k.json

# 10万向量完整测试
python tools/large_scale_test.py --scale 100000 --index-type ivfpq --output results_100k.json

# 全量测试 (多规模)
python tools/large_scale_test.py --all --output benchmark_results.json
```

### 3.2 参数说明

| 参数 | 说明 | 示例 |
|-----|------|------|
| `--scale` | 向量数量 | 10000, 50000, 100000 |
| `--index-type` | 索引类型 | flat, ivf, ivfpq, auto |
| `--cpu` | 强制使用 CPU | --cpu |
| `--output` | 结果输出文件 | results.json |
| `--quiet` | 减少输出 | --quiet |

---

## 4. nprobe 调优

```bash
# 自动调优 nprobe
python tools/nprobe_tuner.py --benchmark --scale 50000 --target-recall 0.95

# 输出示例:
# nprobe=  1: recall=0.125, P50=0.11ms
# nprobe=100: recall=0.515, P50=0.11ms
# nprobe=800: recall=0.938, P50=0.15ms  <-- 推荐值
```

---

## 5. 百万向量极限测试

```bash
# ⚠️ 注意: 此测试需要 15-20 分钟
python tools/million_vector_benchmark.py

# 后台运行 (推荐)
nohup python tools/million_vector_benchmark.py > million_test.log 2>&1 &
tail -f million_test.log  # 查看进度
```

### 百万向量预估性能

| 指标 | 预估 | 说明 |
|-----|------|------|
| 构建时间 | 10-15 min | IVFPQ 训练耗时 |
| 显存占用 | 4-6 GB | 压缩后 |
| 搜索 P50 | < 2ms | GPU 加速 |
| QPS | > 500 | 查询/秒 |
| 召回率 | > 90% | nprobe=100-200 |

---

## 6. 测试结果解读

### 标准输出格式

```
================================================================================
LARGE SCALE TEST SUMMARY
================================================================================
       Scale    Index Device   Build(s)    P50(ms)    P95(ms)   Recall   Status
--------------------------------------------------------------------------------
      10,000     flat    GPU       0.29       0.16       0.18    1.000   ✅ PASS
      50,000      ivf    GPU       1.36       0.35       1.46    0.938   ✅ PASS
     100,000     ivfpq    GPU      3.20       0.45       0.89    0.920   ✅ PASS
================================================================================
```

### 关键指标

| 指标 | 说明 | 优秀标准 |
|-----|------|---------|
| Build(s) | 索引构建时间 | < 5s (10万向量) |
| P50(ms) | 50%延迟 | < 1ms (GPU) |
| P95(ms) | 95%延迟 | < 5ms (GPU) |
| Recall | 召回率@10 | > 90% |

---

## 7. 硬件要求

| 规模 | GPU显存 | 系统内存 | 推荐索引 |
|-----|---------|---------|---------|
| < 1万 | 2GB | 4GB | Flat |
| 1-10万 | 4GB | 8GB | IVF |
| 10-50万 | 6GB | 16GB | IVFPQ |
| 50-100万 | 8GB+ | 32GB | IVFPQ |
| > 100万 | 12GB+ | 64GB | IVFPQ + 分片 |

---

## 8. 故障排除

### Q: 测试超时
**A**: 减小 `--scale` 或使用后台运行

### Q: OOM (显存不足)
**A**: 使用 `--cpu` 或改用 IVFPQ 索引

### Q: 召回率过低
**A**: 调整 nprobe，`backend.set_nprobe(100)`

### Q: 速度比预期慢
**A**: 检查 GPU 是否被使用，`nvidia-smi` 确认

---

## 9. 自动化测试脚本

```bash
#!/bin/bash
# run_all_tests.sh - 运行全部测试

set -e

echo "=== NexusMind 完整测试套件 ==="

# 1. 单元测试
echo "[1/5] 单元测试..."
pytest tests/ -v --tb=short

# 2. GPU 验证
echo "[2/5] GPU 验证..."
python scripts/verify_faiss_gpu.py

# 3. 小规模基准
echo "[3/5] 小规模基准..."
python tools/large_scale_test.py --scale 10000 --quiet

# 4. 标准基准
echo "[4/5] 标准基准..."
python tools/large_scale_test.py --scale 50000 --output results_50k.json

# 5. nprobe 调优
echo "[5/5] nprobe 调优..."
python tools/nprobe_tuner.py --benchmark --scale 50000 --target-recall 0.95

echo "=== 所有测试通过 ==="
```

---

*最后更新: 2026-02-01*
