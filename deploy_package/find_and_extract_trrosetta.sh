#!/bin/bash
# ============================================================
# 查找并解压trRosettaRNA2源代码
# ============================================================

echo "======================================================================"
echo "  查找并解压trRosettaRNA2源代码"
echo "======================================================================"
echo ""

cd /root/autodl-tmp

# Step 1: 查找所有可能的zip文件
echo "[Step 1] 查找trRosettaRNA2 zip文件..."

POSSIBLE_ZIPS=(
    "trRosettaRNA2-2.0.4.zip"
    "trRosettaRNA2-main.zip"
    "trRosettaRNA2.zip"
)

FOUND_ZIP=""

# 检查当前目录
for zip_name in "${POSSIBLE_ZIPS[@]}"; do
    if [ -f "$zip_name" ]; then
        FOUND_ZIP="$zip_name"
        echo "  ✓ 找到: $zip_name"
        break
    fi
done

# 如果当前目录没找到，递归搜索
if [ -z "$FOUND_ZIP" ]; then
    echo "  当前目录未找到，递归搜索..."
    FOUND_ZIP=$(find . -name "*trRosetta*.zip" -o -name "*trrosetta*.zip" 2>/dev/null | head -1)
    if [ -n "$FOUND_ZIP" ]; then
        echo "  ✓ 找到: $FOUND_ZIP"
    fi
fi

# Step 2: 解压
if [ -n "$FOUND_ZIP" ]; then
    echo ""
    echo "[Step 2] 解压 $FOUND_ZIP ..."

    # 切换到zip所在目录
    ZIP_DIR=$(dirname "$FOUND_ZIP")
    ZIP_NAME=$(basename "$FOUND_ZIP")

    cd "$ZIP_DIR"
    echo "  当前目录: $(pwd)"
    echo "  解压: $ZIP_NAME"

    # 解压
    unzip -q "$ZIP_NAME"
    if [ $? -eq 0 ]; then
        echo "  ✓ 解压成功"
    else
        echo "  ✗ 解压失败"
        exit 1
    fi

    # Step 3: 查找解压后的目录
    echo ""
    echo "[Step 3] 查找解压后的目录..."

    POSSIBLE_DIRS=(
        "trRosettaRNA2-2.0.4"
        "trRosettaRNA2-main"
        "trRosettaRNA2"
    )

    EXTRACTED_DIR=""

    for dir_name in "${POSSIBLE_DIRS[@]}"; do
        if [ -d "$dir_name" ]; then
            EXTRACTED_DIR="$dir_name"
            echo "  ✓ 找到目录: $dir_name"
            break
        fi
    done

    if [ -z "$EXTRACTED_DIR" ]; then
        echo "  ✗ 未找到解压目录"
        ls -lh
        exit 1
    fi

    # Step 4: 验证关键文件
    echo ""
    echo "[Step 4] 验证关键文件..."

    PREDICT_PY="$EXTRACTED_DIR/predict.py"
    TRRNA2_DIR="$EXTRACTED_DIR/trRNA2"

    if [ -f "$PREDICT_PY" ]; then
        echo "  ✓ predict.py 存在"
        ls -lh "$PREDICT_PY"
    else
        echo "  ✗ predict.py 缺失"
        echo "  目录内容:"
        ls -lh "$EXTRACTED_DIR/" | head -20
    fi

    if [ -d "$TRRNA2_DIR" ]; then
        echo "  ✓ trRNA2/ 目录存在"
        RNAFORMER="$TRRNA2_DIR/RNAformer.py"
        if [ -f "$RNAFORMER" ]; then
            echo "  ✓ RNAformer.py 存在"
        fi
    else
        echo "  ✗ trRNA2/ 目录缺失"
    fi

    # Step 5: 移动到目标位置
    echo ""
    echo "[Step 5] 移动到目标位置..."

    TARGET_DIR="/root/autodl-tmp/confluencia/confluencia/deploy_package/trRosettaRNA2"

    # 删除旧的
    if [ -d "$TARGET_DIR" ]; then
        echo "  删除旧的trRosettaRNA2..."
        rm -rf "$TARGET_DIR"
    fi

    # 移动新的
    echo "  移动 $EXTRACTED_DIR -> $TARGET_DIR"
    mv "$EXTRACTED_DIR" "$TARGET_DIR"

    if [ -d "$TARGET_DIR" ]; then
        echo "  ✓ 移动成功"
    else
        echo "  ✗ 移动失败"
        exit 1
    fi

    # Step 6: 最终验证
    echo ""
    echo "[Step 6] 最终验证..."

    cd /root/autodl-tmp/confluencia/confluencia/deploy_package

    if [ -f "trRosettaRNA2/predict.py" ]; then
        echo "  ✓ trRosettaRNA2/predict.py 已就绪"
        ls -lh trRosettaRNA2/predict.py

        echo ""
        echo "======================================================================"
        echo "  trRosettaRNA2 安装完成！"
        echo "======================================================================"
        echo ""
        echo "关键文件:"
        echo "  - trRosettaRNA2/predict.py"
        echo "  - trRosettaRNA2/trRNA2/"
        echo "  - trRosettaRNA2/trRNA2/RNAformer.py"
        echo ""
        echo "下一步: 运行修复脚本验证所有配置"
        echo "  bash fix_pipeline.sh"
        echo ""
        echo "或直接运行Pipeline:"
        echo "  python parallel_worker.py --config config_quality.yaml --fasta filtered.fasta --num-workers 1 --output output_real"
        echo "======================================================================"

    else
        echo "  ✗ trRosettaRNA2/predict.py 仍然缺失"
        echo ""
        echo "trRosettaRNA2目录内容:"
        ls -lh trRosettaRNA2/ | head -20
        exit 1
    fi

else
    echo ""
    echo "======================================================================"
    echo "  ✗ 未找到trRosettaRNA2 zip文件"
    echo "======================================================================"
    echo ""
    echo "请上传trRosettaRNA2源代码到AutoDL:"
    echo ""
    echo "方法1: 使用JupyterLab上传"
    echo "  1. 打开AutoDL JupyterLab"
    echo "  2. 导航到 /root/autodl-tmp/"
    echo "  3. 上传 D:/LENOVO/Documents/trRosettaRNA2-2.0.4.zip"
    echo ""
    echo "方法2: 使用scp上传（在本地Windows执行）"
    echo "  scp D:/LENOVO/Documents/trRosettaRNA2-2.0.4.zip root@server:/root/autodl-tmp/"
    echo ""
    echo "方法3: 从GitHub下载"
    echo "  cd /root/autodl-tmp"
    echo "  wget https://github.com/YangLab-SDU/trRosettaRNA2/archive/refs/heads/main.zip"
    echo "  mv main.zip trRosettaRNA2-main.zip"
    echo ""
    echo "上传后重新运行此脚本:"
    echo "  bash find_and_extract_trrosetta.sh"
    echo "======================================================================"
    exit 1
fi