#!/bin/bash
# ============================================================
# 安装trRosettaRNA2剩余依赖
# ============================================================

echo "======================================================================"
echo "  安装trRosettaRNA2剩余依赖"
echo "======================================================================"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# 根据trRosettaRNA2的environment.yml安装依赖
# 参考文件: trRosettaRNA2/environment.yml

echo "安装缺失的Python包..."

# 必需依赖
pip install ml_collections

# 检查其他可能缺失的依赖
python -c "import einops" 2>/dev/null || pip install einops
python -c "import tqdm" 2>/dev/null || pip install tqdm
python -c "import fairseq" 2>/dev/null || echo "fairseq可选，跳过"

# 验证安装
echo ""
echo "验证安装..."

python << 'EOF'
missing = []
try:
    import torch
    print("✓ torch:", torch.__version__)
except:
    missing.append("torch")

try:
    import torch_geometric
    print("✓ torch_geometric:", torch_geometric.__version__)
except:
    missing.append("torch_geometric")

try:
    import ml_collections
    print("✓ ml_collections")
except:
    missing.append("ml_collections")

try:
    import einops
    print("✓ einops")
except:
    missing.append("einops")

try:
    import tqdm
    print("✓ tqdm")
except:
    missing.append("tqdm")

if missing:
    print("\n缺失包:", ", ".join(missing))
    exit(1)
else:
    print("\n所有依赖安装成功！")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "  安装完成！"
    echo "======================================================================"
    echo ""
    echo "下一步: 测试Pipeline"
    echo "  bash fix_pipeline.sh"
else
    echo ""
    echo "有依赖缺失，请手动安装"
fi