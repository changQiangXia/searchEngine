#!/bin/bash
# FAISS GPU Installation Script for NexusMind
# Supports CUDA 11.x and 12.x

set -e

echo "=== NexusMind FAISS GPU Installation ==="

# Detect CUDA version
detect_cuda_version() {
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        echo "Detected CUDA version: $CUDA_VERSION"
    else
        echo "⚠️  nvcc not found, checking nvidia-smi..."
        if command -v nvidia-smi &> /dev/null; then
            CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p')
            echo "Detected CUDA version from nvidia-smi: $CUDA_VERSION"
        else
            echo "❌ Neither nvcc nor nvidia-smi found. Please install CUDA first."
            exit 1
        fi
    fi
}

# Install FAISS GPU based on CUDA version
install_faiss_gpu() {
    local cuda_major=$(echo $CUDA_VERSION | cut -d. -f1)
    
    echo ""
    echo "Uninstalling faiss-cpu (if installed)..."
    pip uninstall -y faiss-cpu 2>/dev/null || true
    
    echo ""
    if [ "$cuda_major" = "12" ]; then
        echo "Installing FAISS GPU for CUDA 12.x..."
        pip install faiss-gpu-cu12
    elif [ "$cuda_major" = "11" ]; then
        echo "Installing FAISS GPU for CUDA 11.x..."
        pip install faiss-gpu-cu11
    else
        echo "⚠️  Untested CUDA version: $CUDA_VERSION"
        echo "Attempting to install faiss-gpu-cu12..."
        pip install faiss-gpu-cu12
    fi
}

# Verify installation
verify_installation() {
    echo ""
    echo "=== Verifying Installation ==="
    
    python3 << 'EOF'
import sys
try:
    import faiss
    print(f"✅ FAISS version: {faiss.__version__}")
    
    # Check GPU availability
    if hasattr(faiss, 'StandardGpuResources'):
        print("✅ GPU support available")
        print(f"✅ Number of GPUs: {faiss.get_num_gpus()}")
        
        # Test GPU functionality
        import numpy as np
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(128)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        print("✅ GPU index creation successful")
        
        print("\n🎉 FAISS GPU installation verified!")
        sys.exit(0)
    else:
        print("❌ GPU support NOT available (still using CPU version)")
        sys.exit(1)
except Exception as e:
    print(f"❌ Verification failed: {e}")
    sys.exit(1)
EOF
}

# Main
main() {
    detect_cuda_version
    install_faiss_gpu
    verify_installation
    
    echo ""
    echo "=== Installation Complete ==="
    echo "You can now use GPU-accelerated FAISS in NexusMind!"
    echo ""
    echo "Quick test:"
    echo "  python -c \"import faiss; print('GPUs:', faiss.get_num_gpus())\""
}

main "$@"
