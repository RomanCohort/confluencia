#!/bin/bash
# ============================================================
# CircRNA Pipeline 一键修复脚本
# 自动修复诊断脚本发现的所有问题
# ============================================================

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "  CircRNA Pipeline 一键修复脚本"
echo "======================================================================"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# ============================================================
# 修复1: 安装OpenMM
# ============================================================
echo "[修复1] 安装OpenMM..."

python -c "import openmm" 2>/dev/null && {
    echo "  ✓ OpenMM已安装"
    python -c "import openmm; print(f'  版本: {openmm.__version__}')"
} || {
    echo "  安装OpenMM..."
    conda install -c conda-forge openmm -y || pip install openmm
    echo "  ✓ OpenMM安装完成"
}

# ============================================================
# 修复2: 检查并修复trRosettaRNA2
# ============================================================
echo ""
echo "[修复2] 检查trRosettaRNA2..."

if [ -f "trRosettaRNA2/predict.py" ]; then
    echo "  ✓ trRosettaRNA2/predict.py 已存在"
    ls -lh trRosettaRNA2/predict.py
else
    echo "  ✗ trRosettaRNA2/predict.py 缺失"

    # 检查是否有ZIP文件
    if [ -f "trRosettaRNA2-2.0.4.zip" ]; then
        echo "  解压 trRosettaRNA2-2.0.4.zip..."
        rm -rf trRosettaRNA2
        unzip -q trRosettaRNA2-2.0.4.zip
        mv trRosettaRNA2-2.0.4 trRosettaRNA2
        echo "  ✓ 解压完成"
    elif [ -f "trRosettaRNA2.tar.gz" ]; then
        echo "  解压 trRosettaRNA2.tar.gz..."
        rm -rf trRosettaRNA2
        tar -xzf trRosettaRNA2.tar.gz
        echo "  ✓ 解压完成"
    else
        echo "  ⚠️  未找到trRosettaRNA2压缩包，需要手动上传"
        echo ""
        echo "  上传方法（在本地Windows执行）:"
        echo "    scp D:/LENOVO/Documents/trRosettaRNA2-2.0.4.zip root@server:/root/autodl-tmp/confluencia/confluencia/deploy_package/"
        echo ""
        echo "  或从GitHub下载:"
        echo "    wget https://github.com/YangLab-SDU/trRosettaRNA2/archive/refs/heads/main.zip"
        echo "    unzip main.zip"
        echo "    mv trRosettaRNA2-main trRosettaRNA2"
        echo ""
        echo "  然后重新运行此脚本"
        # 不退出，继续修复其他问题
    fi
fi

# 验证关键文件
if [ -d "trRosettaRNA2" ]; then
    echo ""
    echo "  验证trRosettaRNA2关键文件:"

    [ -f "trRosettaRNA2/predict.py" ] && echo "    ✓ predict.py" || echo "    ✗ predict.py 缺失"
    [ -d "trRosettaRNA2/trRNA2" ] && echo "    ✓ trRNA2/" || echo "    ✗ trRNA2/ 缺失"
    [ -f "trRosettaRNA2/trRNA2/RNAformer.py" ] && echo "    ✓ RNAformer.py" || echo "    ✗ RNAformer.py 缺失"
fi

# ============================================================
# 修复3: 权重文件检查
# ============================================================
echo ""
echo "[修复3] 检查权重文件..."

if [ -d "weights/params/models" ]; then
    MODEL_COUNT=$(ls weights/params/models/*.pth.tar 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "  ✓ 权重文件已存在 ($MODEL_COUNT 个模型)"
        ls -lh weights/params/models/*.pth.tar | awk '{print "    " $9 " (" $5 ")"}'
    else
        echo "  ✗ 权重文件缺失"
        echo "  下载权重..."
        wget http://yanglab.qd.sdu.edu.cn/trRosettaRNA/download/params_trRNA2.tar.bz2 && \
        tar -jxvf params_trRNA2.tar.bz2 && \
        mkdir -p weights && \
        mv params weights/params && \
        echo "  ✓ 权重下载完成" || \
        echo "  ✗ 权重下载失败，需要手动下载"
    fi
else
    echo "  ✗ weights/params/models 目录不存在"
fi

# ============================================================
# 修复4: 更新配置文件路径
# ============================================================
echo ""
echo "[修复4] 更新配置文件路径..."

# 备份原配置
cp config_quality.yaml config_quality.yaml.bak 2>/dev/null || true
cp config.yaml config.yaml.bak 2>/dev/null || true

# 更新model_path
sed -i 's|model_path: models/rosettafold2na/|model_path: ./trRosettaRNA2/|g' config_quality.yaml
sed -i 's|model_path: models/rosettafold2na/|model_path: ./trRosettaRNA2/|g' config.yaml

# 添加use_gpu配置（如果缺失）
grep -q "use_gpu:" config_quality.yaml || sed -i '/device: cuda:0/a\  use_gpu: true' config_quality.yaml
grep -q "use_gpu:" config.yaml || sed -i '/device: cuda:0/a\  use_gpu: true' config.yaml

echo "  ✓ 配置文件已更新"
echo ""
echo "  验证配置:"
grep "model_path:" config_quality.yaml | head -1
grep "device:" config_quality.yaml | head -1
grep "use_gpu:" config_quality.yaml | head -1 || echo "  use_gpu: true (已添加)"

# ============================================================
# 修复5: 安装Python依赖
# ============================================================
echo ""
echo "[修复5] 检查Python依赖..."

MISSING_DEPS=""

python -c "import torch" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS torch"
python -c "import openmm" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS openmm"
python -c "import ray" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS ray"
python -c "import yaml" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS pyyaml"
python -c "import numpy" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS numpy"
python -c "import scipy" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS scipy"
python -c "import RNA" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS viennarna (conda install -c bioconda viennarna)"

if [ -z "$MISSING_DEPS" ]; then
    echo "  ✓ 所有Python依赖已安装"
else
    echo "  安装缺失的依赖: $MISSING_DEPS"
    pip install $MISSING_DEPS 2>/dev/null || conda install -y $MISSING_DEPS
fi

# ============================================================
# 修复6: 设置环境变量
# ============================================================
echo ""
echo "[修复6] 设置环境变量..."

export TRROSETTARNA2_HOME=/root/autodl-tmp/confluencia/confluencia/deploy_package/trRosettaRNA2

# 添加到.bashrc
if ! grep -q "TRROSETTARNA2_HOME" ~/.bashrc; then
    echo "export TRROSETTARNA2_HOME=/root/autodl-tmp/confluencia/confluencia/deploy_package/trRosettaRNA2" >> ~/.bashrc
    echo "  ✓ 已添加到 ~/.bashrc"
else
    echo "  ✓ 环境变量已设置"
fi

echo "  当前: TRROSETTARNA2_HOME=$TRROSETTARNA2_HOME"

# ============================================================
# 修复7: 验证Pipeline代码
# ============================================================
echo ""
echo "[修复7] 验证Pipeline代码..."

REQUIRED_FILES=(
    "circrna_3d_pipeline/pipeline.py"
    "circrna_3d_pipeline/stage1_vienna.py"
    "circrna_3d_pipeline/stage2_trrosetta.py"
    "circrna_3d_pipeline/stage3_cyclize.py"
    "circrna_3d_pipeline/stage4_md.py"
    "circrna_3d_pipeline/stage5_quality.py"
    "circrna_3d_pipeline/parallel_worker.py"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file 缺失"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "  从GitHub拉取最新代码..."
    git pull origin main
    cp circrna_3d_pipeline/*.py ./
fi

# ============================================================
# 最终验证
# ============================================================
echo ""
echo "======================================================================"
echo "  最终验证"
echo "======================================================================"
echo ""

echo "运行诊断脚本..."
python diagnose_pipeline.py

echo ""
echo "======================================================================"
echo "  修复完成！"
echo "======================================================================"
echo ""
echo "下一步:"
echo ""
echo "1. 如果trRosettaRNA2仍然缺失，上传完整包:"
echo "   scp D:/LENOVO/Documents/trRosettaRNA2-2.0.4.zip root@server:/root/autodl-tmp/confluencia/confluencia/deploy_package/"
echo "   然后重新运行此脚本"
echo ""
echo "2. 运行Pipeline:"
echo "   python parallel_worker.py --config config_quality.yaml --fasta filtered.fasta --num-workers 1 --output output_real"
echo ""
echo "3. 监控GPU使用（应该是80-100%）:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "======================================================================"