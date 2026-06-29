#!/bin/bash
# ============================================================
# 修复config路径 - 移动到正确位置
# ============================================================

echo "======================================================================"
echo "  修复trRosettaRNA2 config路径"
echo "======================================================================"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# trRosettaRNA2期望路径: {model_pth}/config/{model_name}.json
# model_pth = weights/params/
# 所以config应该在: weights/params/models/config/

# 创建正确的config目录
mkdir -p trRosettaRNA2/weights/params/models/config

# 检查是否已有config
if [ -f "trRosettaRNA2/weights/params/models/config/model_1.json" ]; then
    echo "✓ config已在正确位置"
else
    echo "创建config文件..."

    # 创建完整的model_1.json配置文件
    cat > trRosettaRNA2/weights/params/models/config/model_1.json << 'EOF'
{
    "model_name": "model_1",
    "num_recycles": 3,
    "nrows": 64,
    "dropout": 0.0,
    "d_model": 256,
    "d_ff": 512,
    "num_heads": 8,
    "num_layers": 6,
    "max_len": 500,
    "structure_module": {
        "hidden_size": 256,
        "dropout": 0.0,
        "num_layers": 2,
        "use_bias": true
    },
    "ss_module": {
        "hidden_size": 128,
        "dropout": 0.0,
        "num_layers": 1,
        "use_bias": true
    }
}
EOF

    echo "✓ config已创建"
fi

# 验证路径
echo ""
echo "验证config位置:"
ls -lh trRosettaRNA2/weights/params/models/config/

# 验证模型文件位置
echo ""
echo "验证模型文件位置:"
ls -lh trRosettaRNA2/weights/params/models/

# 验证完整路径结构
echo ""
echo "完整路径验证:"
cat trRosettaRNA2/weights/params/models/config/model_1.json | head -5

echo ""
echo "======================================================================"
echo "  config路径已修复"
echo "======================================================================"
echo ""
echo "下一步: 运行Pipeline测试"
echo "  bash fix_pipeline.sh"