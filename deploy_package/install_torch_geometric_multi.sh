#!/bin/bash
# ============================================================
# 安装torch_geometric - 使用conda或直接pip
# ============================================================

echo "======================================================================"
echo "  安装torch_geometric（多种方法）"
echo "======================================================================"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# 方法1: 直接pip安装（最快）
echo "[方法1] 直接pip安装torch_geometric..."
pip install torch-geometric

# 验证
python -c "import torch_geometric" 2>/dev/null && {
    echo "✓ torch_geometric已安装"
    python -c "import torch_geometric; print('版本:', torch_geometric.__version__)"
    echo ""
    echo "安装成功！运行fix_pipeline.sh验证"
    bash fix_pipeline.sh
    exit 0
} || {
    echo "✗ 方法1失败"
}

# 方法2: conda安装
echo ""
echo "[方法2] conda安装..."
conda install -c pyg pyg -y

python -c "import torch_geometric" 2>/dev/null && {
    echo "✓ torch_geometric已安装（conda）"
    bash fix_pipeline.sh
    exit 0
} || {
    echo "✗ 方法2失败"
}

# 方法3: 从源安装（最后手段）
echo ""
echo "[方法3] 从源安装（可能很慢）..."
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

python -c "import torch_geometric" && {
    echo "✓ 安装成功"
    bash fix_pipeline.sh
    exit 0
} || {
    echo "✗ 所有方法失败"
    echo ""
    echo "手动安装建议:"
    echo "  pip install torch-geometric"
    echo "  或:"
    echo "  conda install -c pyg pyg"
    exit 1
}