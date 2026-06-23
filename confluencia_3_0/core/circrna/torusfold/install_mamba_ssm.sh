#!/bin/bash
# install_mamba_ssm.sh — Install mamba-ssm with CUDA version compatibility fix
#
# Problem: mamba-ssm requires nvcc version to match PyTorch's CUDA version,
# but the system CUDA toolkit may be older (e.g., CUDA 12.1 nvcc vs
# PyTorch compiled with CUDA 13.0).
#
# Solution: Create a nvcc wrapper that reports the correct CUDA version
# for the version check, then passes through to the real nvcc for
# actual compilation.
#
# Usage:
#   bash confluencia_3_0/core/circrna/torusfold/install_mamba_ssm.sh
#
# Requirements:
#   - PyTorch with CUDA support already installed
#   - GCC with C++14 support
#   - NVIDIA GPU driver supporting the required CUDA version

set -e

echo "=== mamba-ssm Installation Script ==="

# Detect PyTorch CUDA version
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
if [ -z "$TORCH_CUDA" ]; then
    echo "ERROR: PyTorch not found or not built with CUDA"
    exit 1
fi
echo "PyTorch CUDA version: $TORCH_CUDA"

# Detect system nvcc version
SYS_CUDA=""
if command -v nvcc &> /dev/null; then
    SYS_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "")
fi
echo "System nvcc CUDA version: ${SYS_CUDA:-not found}"

# Detect cxx11 ABI
CXX11_ABI=$(python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" 2>/dev/null || echo "True")
echo "CXX11 ABI: $CXX11_ABI"

# Detect GPU architecture
GPU_ARCH=$(python3 -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'{cap[0]}.{cap[1]}')
else:
    print('8.0')
" 2>/dev/null || echo "8.0")
echo "GPU compute capability: $GPU_ARCH"

# Try downloading pre-built wheel from GitHub releases first
echo ""
echo "Step 1: Checking for pre-built wheel on GitHub..."

CUDA_VER_SHORT=$(echo "$TORCH_CUDA" | tr -d '.')
PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
ABI_FLAG=$([ "$CXX11_ABI" = "True" ] && echo "cxx11abiTRUE" || echo "cxx11abiFALSE")

WHEEL_URL=$(curl -sL https://api.github.com/repos/state-spaces/mamba/releases/latest 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
cuda_short = '${CUDA_VER_SHORT}'
py_ver = '${PY_VER}'
abi = '${ABI_FLAG}'.lower()
for a in data.get('assets', []):
    name = a['name'].lower()
    if f'cu{cuda_short}' in name and py_ver in name and abi in name:
        print(a['browser_download_url'])
        break
" 2>/dev/null || echo "")

if [ -n "$WHEEL_URL" ]; then
    echo "  Found pre-built wheel: $WHEEL_URL"
    echo "  Downloading..."
    pip install "$WHEEL_URL"
    echo ""
    echo "Verifying installation..."
    python3 -c "from mamba_ssm import Mamba; print('mamba-ssm installed successfully!')"
    exit 0
else
    echo "  No pre-built wheel found. Will build from source."
fi

# Build from source with nvcc wrapper
echo ""
echo "Step 2: Building from source with CUDA version fix..."

NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
if [ -z "$NVCC_PATH" ]; then
    echo "ERROR: nvcc not found. Install CUDA toolkit first."
    exit 1
fi

NVCC_DIR=$(dirname "$NVCC_PATH")

# Check if versions match
if [ "$SYS_CUDA" = "$TORCH_CUDA" ]; then
    echo "  CUDA versions match. Installing normally."
    TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install mamba-ssm --no-build-isolation --force-reinstall --no-binary mamba-ssm
else
    echo "  CUDA version mismatch (system=$SYS_CUDA, torch=$TORCH_CUDA). Applying fix."

    # Backup real nvcc
    if [ ! -f "${NVCC_PATH}.real" ]; then
        cp "$NVCC_PATH" "${NVCC_PATH}.real"
        echo "  Backed up nvcc to ${NVCC_PATH}.real"
    fi

    # Patch nvcc version check: mamba-ssm's setup.py calls
    # subprocess.check_output([nvcc, "--version"]) to get CUDA version.
    # We need nvcc --version to return PyTorch's CUDA version string.
    MAJOR=$(echo "$TORCH_CUDA" | cut -d. -f1)
    MINOR=$(echo "$TORCH_CUDA" | cut -d. -f2)

    cat > "$NVCC_PATH" << NVCCEOF
#!/bin/bash
# nvcc wrapper: make version check return torch CUDA version
# while passing actual compilation to the real nvcc.
# The torch.cpp_extension._check_cuda_version function reads
# the "release X.Y" line from nvcc --version output.
if [[ "\$*" == *--version* ]] || [[ "\$#" -eq 0 ]]; then
    /usr/local/cuda/bin/nvcc.real --version | sed "s/release [0-9.]*/release ${TORCH_CUDA}/"
else
    /usr/local/cuda/bin/nvcc.real "\$@"
fi
NVCCEOF
    chmod +x "$NVCC_PATH"
    echo "  Created nvcc wrapper at $NVCC_PATH"

    # Set environment and install
    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
    export CUDA_HOME="/usr/local/cuda"
    pip install mamba-ssm --no-build-isolation --force-reinstall --no-binary mamba-ssm

    # Restore real nvcc
    mv "${NVCC_PATH}.real" "$NVCC_PATH"
    echo "  Restored original nvcc."
fi

# Verify
echo ""
echo "Verifying installation..."
python3 -c "from mamba_ssm import Mamba; print('mamba-ssm installed successfully!')" || {
    echo ""
    echo "WARNING: mamba-ssm import failed."
    echo "The pure-Python SSM fallback in circrna_mamba_diffusion.py will be used instead."
    echo "This is slower but functional."
}
