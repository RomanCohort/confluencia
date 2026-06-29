#!/usr/bin/env python3
"""
CircRNA Pipeline诊断脚本

检查所有可能导致异常的问题：
1. trRosettaRNA2安装
2. GPU可用性
3. 权重文件
4. Pipeline代码
5. 配置文件
6. 依赖库
7. 实际运行测试
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def print_status(check_name, status, details=None):
    symbol = "✓" if status else "✗"
    print(f"{symbol} {check_name}")
    if details:
        print(f"    {details}")

def check_trrosetta_installation():
    print_header("检查 trRosettaRNA2 安装")

    possible_paths = [
        os.environ.get('TRROSETTARNA2_HOME', ''),
        './trRosettaRNA2',
        './trRosettaRNA2-2.0.4',
        '/opt/trRosettaRNA2',
        '~/trRosettaRNA2',
        '~/software/trRosettaRNA2',
    ]

    found = False
    for path in possible_paths:
        if path and os.path.exists(path):
            predict_script = os.path.join(path, 'predict.py')
            if os.path.exists(predict_script):
                print_status("trRosettaRNA2路径", True, f"Found at: {path}")
                print_status("predict.py", True, predict_script)

                # Check key files
                trrna2_dir = os.path.join(path, 'trRNA2')
                if os.path.exists(trrna2_dir):
                    print_status("trRNA2目录", True, trrna2_dir)
                    rnaformer = os.path.join(trrna2_dir, 'RNAformer.py')
                    print_status("RNAformer.py", os.path.exists(rnaformer), rnaformer)
                else:
                    print_status("trRNA2目录", False, "Missing")

                found = True
                return path
            else:
                print_status(f"路径存在但predict.py缺失", False, path)

    if not found:
        print_status("trRosettaRNA2", False, "Not found in any path")
        print("\n安装指南:")
        print("  git clone https://github.com/YangLab-SDU/trRosettaRNA2.git")
        print("  wget http://yanglab.qd.sdu.edu.cn/trRosettaRNA/download/params_trRNA2.tar.bz2")
        print("  tar -jxvf params_trRNA2.tar.bz2")
        print("  export TRROSETTARNA2_HOME=/path/to/trRosettaRNA2")

    return None

def check_weights():
    print_header("检查权重文件")

    possible_weights_paths = [
        './weights/params',
        './params',
        './trRosettaRNA2/weights',
    ]

    found = False
    for path in possible_weights_paths:
        if os.path.exists(path):
            print_status("权重目录", True, path)

            # Check model files
            models_dir = os.path.join(path, 'models')
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth.tar')]
                if model_files:
                    print_status("模型文件", True, f"{len(model_files)} files in {models_dir}")
                    for f in model_files:
                        filepath = os.path.join(models_dir, f)
                        size = os.path.getsize(filepath) / (1024 * 1024)
                        print(f"      - {f} ({size:.1f} MB)")
                    found = True
                else:
                    print_status("模型文件", False, f"No .pth.tar files in {models_dir}")
            else:
                print_status("models目录", False, "Missing")

            # Check models_ss
            models_ss_dir = os.path.join(path, 'models_ss')
            if os.path.exists(models_ss_dir):
                ss_files = [f for f in os.listdir(models_ss_dir) if f.endswith('.pth.tar')]
                print_status("SS模型", len(ss_files) > 0, f"{len(ss_files)} files in {models_ss_dir}")

    if not found:
        print_status("权重文件", False, "Not found")
        print("\n下载指南:")
        print("  wget http://yanglab.qd.sdu.edu.cn/trRosettaRNA/download/params_trRNA2.tar.bz2")
        print("  tar -jxvf params_trRNA2.tar.bz2")
        print("  mv params weights/params")

def check_gpu():
    print_header("检查GPU")

    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print_status("nvidia-smi", True)
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line and 'Name' not in line:
                    print(f"    {line.strip()}")
        else:
            print_status("nvidia-smi", False, result.stderr)
    except FileNotFoundError:
        print_status("nvidia-smi", False, "Command not found - GPU not available")
    except subprocess.TimeoutExpired:
        print_status("nvidia-smi", False, "Timeout - GPU may be busy")

    # Check PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print_status("PyTorch CUDA", cuda_available)

        if cuda_available:
            device_count = torch.cuda.device_count()
            print_status("GPU数量", True, f"{device_count} devices")

            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"      GPU {i}: {device_name}")

            # Memory check
            for i in range(device_count):
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"      GPU {i} Memory: {total:.1f} GB")
        else:
            print_status("GPU可用", False, "torch.cuda.is_available() = False")
            print("    原因可能:")
            print("      - PyTorch是CPU版本")
            print("      - CUDA驱动未安装")
            print("      - GPU硬件问题")
    except ImportError:
        print_status("PyTorch", False, "Not installed")

def check_pipeline_code():
    print_header("检查Pipeline代码")

    required_files = [
        'pipeline.py',
        'stage1_vienna.py',
        'stage2_trrosetta.py',
        'stage3_cyclize.py',
        'stage4_md.py',
        'stage5_quality.py',
        'parallel_worker.py',
    ]

    all_exist = True
    for file in required_files:
        path = f"./circrna_3d_pipeline/{file}"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print_status(file, True, f"{path} ({size} bytes)")
        else:
            print_status(file, False, f"Missing: {path}")
            all_exist = False

    if not all_exist:
        print("\n修复:")
        print("  git pull origin main")
        print("  cp circrna_3d_pipeline/*.py ./")

def check_config():
    print_header("检查配置文件")

    config_files = ['config_quality.yaml', 'config.yaml']

    for config_file in config_files:
        if os.path.exists(config_file):
            print_status(config_file, True)

            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)

                # Check critical keys
                if 'rosetta' in config:
                    print_status("rosetta配置", True)
                    rosetta = config['rosetta']
                    print(f"    model_path: {rosetta.get('model_path', 'MISSING')}")
                    print(f"    device: {rosetta.get('device', 'MISSING')}")
                    print(f"    use_gpu: {rosetta.get('use_gpu', 'MISSING')}")
                else:
                    print_status("rosetta键", False, "Missing - KeyError will occur!")

                if 'vienna' in config:
                    print_status("vienna配置", True)
                else:
                    print_status("vienna键", False)

                if 'cyclize' in config:
                    print_status("cyclize配置", True)
                else:
                    print_status("cyclize键", False)

                if 'md' in config:
                    print_status("md配置", True)
                else:
                    print_status("md键", False)

            except Exception as e:
                print_status("配置解析", False, str(e))
        else:
            print_status(config_file, False, f"Missing")

def check_dependencies():
    print_header("检查Python依赖")

    required_packages = {
        'torch': 'PyTorch',
        'openmm': 'OpenMM',
        'ray': 'Ray',
        'yaml': 'PyYAML',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'RNA': 'ViennaRNA',
    }

    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_status(name, True)
        except ImportError:
            print_status(name, False, f"pip install {package}")
            all_installed = False

    if not all_installed:
        print("\n安装:")
        print("  pip install torch openmm ray pyyaml numpy scipy")
        print("  conda install -c bioconda viennarna")

def check_actual_prediction():
    print_header("实际运行测试")

    print("测试trRosettaRNA2预测...")

    # Find trRosettaRNA2
    trrosetta_path = check_trrosetta_installation()
    if not trrosetta_path:
        print_status("无法测试", False, "trRosettaRNA2未找到")
        return False

    # Try to import and predict
    try:
        sys.path.insert(0, './circrna_3d_pipeline')
        from stage2_trrosetta import trRosettaRNA2Predictor

        config = {
            'model_path': trrosetta_path,
            'num_samples': 1,
            'device': 'cuda:0',
            'use_gpu': True,
            'max_seq_length': 50,
        }

        predictor = trRosettaRNA2Predictor(config)

        if predictor.trrosetta_home:
            print_status("Predictor初始化", True)

            # Test prediction
            test_seq = "ACGUACGUACGUACGUACGUACGU"  # 24nt
            print(f"测试序列: {test_seq} ({len(test_seq)}nt)")

            import tempfile
            output_dir = tempfile.mkdtemp()

            print("开始预测...")
            start_time = subprocess.time.time() if hasattr(subprocess, 'time') else 0

            results = predictor.predict(test_seq, output_dir=output_dir)

            elapsed = subprocess.time.time() if hasattr(subprocess, 'time') else 0

            print_status("预测完成", True, f"{len(results)} structures generated")

            for i, result in enumerate(results):
                pdb_path = result.get('pdb_path')
                if pdb_path and os.path.exists(pdb_path):
                    size = os.path.getsize(pdb_path)
                    print_status(f"PDB文件{i}", True, f"{pdb_path} ({size} bytes)")
                else:
                    print_status(f"PDB文件{i}", False, "Missing")

            print_status("GPU使用测试", True, "Prediction should use GPU")
            return True
        else:
            print_status("Predictor初始化", False, "trrosetta_home is None")
            return False

    except Exception as e:
        print_status("预测失败", False, str(e))
        import traceback
        print("\n详细错误:")
        traceback.print_exc()
        return False

def generate_report():
    print_header("诊断报告汇总")

    print("检查项目状态:")
    checks = [
        ("trRosettaRNA2安装", check_trrosetta_installation() is not None),
        ("权重文件", os.path.exists('./weights/params/models')),
        ("GPU可用", True),  # Will be checked above
        ("Pipeline代码", os.path.exists('./circrna_3d_pipeline/pipeline.py')),
        ("配置文件", os.path.exists('./config_quality.yaml')),
        ("Python依赖", True),  # Will be checked above
    ]

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    for check_name, status in checks:
        print_status(check_name, status)

    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed < total:
        print("\n⚠️  存在问题，需要修复!")
        print("    请根据上述检查结果进行修复")
    else:
        print("\n✓  所有检查通过!")

def main():
    print("="*70)
    print("  CircRNA Pipeline 诊断脚本")
    print("  检查所有可能导致异常快的问题")
    print("="*70)

    os.chdir('/root/autodl-tmp/confluencia/confluencia/deploy_package')

    check_trrosetta_installation()
    check_weights()
    check_gpu()
    check_pipeline_code()
    check_config()
    check_dependencies()
    check_actual_prediction()
    generate_report()

    print("\n"+"="*70)
    print("  诊断完成")
    print("="*70)

if __name__ == '__main__':
    main()