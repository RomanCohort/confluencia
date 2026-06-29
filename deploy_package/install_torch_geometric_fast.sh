#!/bin/bash
# ============================================================
# 快速安装torch_geometric（使用预编译wheel）
# ============================================================

echo "======================================================================"
echo "  快速安装torch_geometric（预编译wheel）"
echo "======================================================================"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# 获取PyTorch和CUDA版本
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")

echo "PyTorch版本: $TORCH_VER"
echo "CUDA版本: $CUDA_VER"
echo ""

# 使用预编译wheel（约1-2分钟）
echo "安装预编译wheel..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html

# 安装torch_geometric
echo ""
echo "安装torch_geometric..."
pip install torch-geometric

# 验证安装
echo ""
echo "验证安装..."

python << 'EOF'
import torch
try:
    import torch_scatter
    import torch_sparse
    import torch_cluster
    import torch_geometric

    print("✓ torch:", torch.__version__)
    print("✓ torch_scatter")
    print("✓ torch_sparse")
    print("✓ torch_cluster")
    print("✓ torch_geometric:", torch_geometric.__version__)
    print("\n所有依赖安装成功！")
except ImportError as e:
    print(f"✗ 安装失败: {e}")
    exit(1)
EOF

echo ""
echo "======================================================================"
echo "  安装完成！"
echo "======================================================================"
echo ""
echo "下一步: 运行诊断脚本验证Pipeline"
echo "  bash fix_pipeline.sh"