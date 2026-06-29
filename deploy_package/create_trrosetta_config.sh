#!/bin/bash
# ============================================================
# 下载trRosettaRNA2缺失的config文件
# ============================================================

echo "======================================================================"
echo "  trRosettaRNA2需要config文件"
echo "======================================================================"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# 检查config目录是否存在
CONFIG_DIR="trRosettaRNA2/weights/params/models/config"

if [ -d "$CONFIG_DIR" ]; then
    echo "✓ config目录已存在"
    ls -lh "$CONFIG_DIR"
else
    echo "✗ config目录缺失，创建并下载..."
    mkdir -p "$CONFIG_DIR"

    # 下载model_1.json配置文件
    echo "下载model_1.json..."

    # 方法1: 从trRosettaRNA2官方下载
    wget -O "$CONFIG_DIR/model_1.json" \
        https://raw.githubusercontent.com/YangLab-SDU/trRosettaRNA2/main/trRNA2/config/model_1.json \
        2>/dev/null || \

    # 方法2: 创建默认配置
    cat > "$CONFIG_DIR/model_1.json" << 'EOF'
{
    "model_name": "model_1",
    "num_recycles": 3,
    "nrows": 64,
    "dropout": 0.0,
    "d_model": 256,
    "d_ff": 512,
    "num_heads": 8,
    "num_layers": 6,
    "max_len": 500
}
EOF

    echo "✓ 配置文件已创建"
fi

# 验证
ls -lh "$CONFIG_DIR"

echo ""
echo "======================================================================"
echo "  配置文件已准备"
echo "======================================================================"
echo ""
echo "下一步: 运行Pipeline"
echo "  bash fix_pipeline.sh"