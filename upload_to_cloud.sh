#!/bin/bash
# ============================================================
# 一键上传脚本 - 从Windows上传到云GPU服务器
# ============================================================

echo "============================================================"
echo "  circRNA环化Pipeline - 上传到云GPU"
echo "============================================================"
echo ""

# 提示用户输入云GPU服务器信息
read -p "请输入云GPU服务器地址 (例如: root@192.168.1.100): " SERVER_ADDR
read -p "请输入上传目录 (默认: /root): " UPLOAD_DIR
UPLOAD_DIR=${UPLOAD_DIR:-/root}

echo ""
echo "服务器: $SERVER_ADDR"
echo "上传目录: $UPLOAD_DIR"
echo ""

# 检查部署包
if [ ! -f "deploy_package.tar.gz" ] || [ ! -f "pipeline_code.tar.gz" ]; then
    echo "[ERROR] 部署包不存在"
    echo "  请先运行打包脚本:"
    echo "    tar -czf deploy_package.tar.gz deploy_package/"
    echo "    tar -czf pipeline_code.tar.gz confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline/*.py ..."
    exit 1
fi

echo "[Step 1] 上传部署包..."
scp deploy_package.tar.gz pipeline_code.tar.gz $SERVER_ADDR:$UPLOAD_DIR/

echo ""
echo "[Step 2] 上传后在云GPU执行:"
echo "  ssh $SERVER_ADDR"
echo "  cd $UPLOAD_DIR"
echo "  tar -xzf deploy_package.tar.gz"
echo "  tar -xzf pipeline_code.tar.gz -C deploy_package/"
echo "  cd deploy_package"
echo "  python verify_dependencies.py"
echo "  ./deploy_full_pipeline.sh input.fasta"
echo ""
echo "============================================================"
echo "  上传完成！"
echo "============================================================"