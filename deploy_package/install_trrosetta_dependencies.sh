#!/bin/bash
# ============================================================
# 安装trRosettaRNA2依赖 - torch_geometric
# ============================================================

echo "======================================================================"
echo "  安装trRosettaRNA2缺失依赖"
echo "======================================================================"
echo ""

# 检查torch_geometric
python -c "import torch_geometric" 2>/dev/null && {
    echo "✓ torch_geometric已安装"
    python -c "import torch_geometric; print(f'版本: {torch_geometric.__version__}')"
} || {
    echo "✗ torch_geometric缺失，开始安装..."
    echo ""

    # 获取PyTorch和CUDA版本
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

    echo "PyTorch版本: $TORCH_VERSION"
    echo "CUDA版本: $CUDA_VERSION"
    echo ""

    # 安装torch_geometric及依赖
    echo "安装torch_geometric..."
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html || \
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv

    pip install torch-geometric

    echo ""
    echo "验证安装..."
    python -c "import torch_geometric; print('✓ torch_geometric安装成功')" || \
    echo "✗ 安装失败，尝试conda安装..."

    # 如果pip失败，尝试conda
    if [ $? -ne 0 ]; then
        echo "尝试conda安装..."
        conda install -c pyg pyg -y
    fi
}

# 检查其他可能缺失的依赖
echo ""
echo "检查其他依赖..."

python -c "import torch_scatter" 2>/dev/null && echo "✓ torch_scatter" || echo "✗ torch_scatter"
python -c "import torch_sparse" 2>/dev/null && echo "✓ torch_sparse" || echo "✗ torch_sparse"
python -c "import torch_cluster" 2>/dev/null && echo "✓ torch_cluster" || echo "✗ torch_cluster"

echo ""
echo "======================================================================"
echo "  安装完成"
echo "======================================================================"
echo ""
echo "下一步: 重新运行诊断脚本"
echo "  python diagnose_pipeline.py"