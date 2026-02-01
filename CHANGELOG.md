# Changelog

All notable changes to NexusMind will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-02-01

### Added
- Core semantic search with CLIP and FAISS
- 3-level GPU memory protection circuit breaker
- CLI interface with rich output
- Concept interpolation (SLERP/LEERP)
- Concept blending (multi-concept weighted)
- Negative search (semantic exclusion)
- MMR diversity ranking
- Semantic clustering (KMeans/HDBSCAN)
- Cross-modal chain reasoning
- Streamlit web interface
- 3D semantic galaxy visualization (Plotly)
- Attention heatmap visualization
- Plugin system architecture
- Tiered cache (L1/L2/L3)
- Model quantization (FP16/INT8/INT4)
- Dynamic batching optimization
- Performance monitoring
- Docker support with CUDA
- GitHub Actions CI/CD

### Features
- Support for RTX 3080ti (12GB) and RTX 4090 (24GB)
- CPU fallback for OOM protection
- Export plugins (CSV, JSON, HTML)
- Async processing
- Streaming batch processing for large datasets
- Workspace management

[0.1.0]: https://github.com/changQiangXia/searchEngine/releases/tag/v0.1.0