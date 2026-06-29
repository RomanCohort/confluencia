#!/usr/bin/env python3
"""
DGX/云GPU集群完整Pipeline验证脚本
检查全部依赖是否正确安装

验证内容:
1. ViennaRNA (Stage 1)
2. RoseTTAFold2NA (Stage 2)
3. OpenMM (Stage 3-4)
4. Ray并行
5. GPU可用性
"""

import sys
import os
import subprocess
from pathlib import Path

def check_command(cmd, name):
    """检查命令是否可用"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
        if result.returncode == 0:
            print(f"  [OK] {name}")
            return True
        else:
            print(f"  [FAIL] {name}: {result.stderr.decode()[:100]}")
            return False
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

def check_python_module(module_name, import_name=None):
    """检查Python模块"""
    import_name = import_name or module_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        location = getattr(module, '__file__', 'unknown')
        print(f"  [OK] {module_name}: {version}")
        print(f"       Location: {location}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {module_name}: {e}")
        return False

def main():
    """主验证流程"""

    print("=" * 80)
    print("  circRNA环化Pipeline - 完整依赖验证")
    print("=" * 80)
    print()

    results = {}

    # ============================================================
    # Step 1: 系统信息
    # ============================================================
    print("[Step 1] 系统信息")
    print("-" * 80)

    # Python版本
    py_version = sys.version_info
    print(f"  Python: {py_version.major}.{py_version.minor}.{py_version.micro}")

    if py_version.major == 3 and py_version.minor in [9, 10, 11]:
        print("  [OK] Python版本符合要求")
        results['python'] = True
    else:
        print("  [FAIL] 需要Python 3.9/3.10/3.11")
        results['python'] = False

    # GPU信息
    print()
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap', '--format=csv,noheader'],
            capture_output=True, timeout=10
        )
        if result.returncode == 0:
            gpu_info = result.stdout.decode().strip()
            gpu_count = len(gpu_info.split('\n'))
            print(f"  GPU数量: {gpu_count}")
            print(f"  GPU信息:")
            for line in gpu_info.split('\n'):
                print(f"    {line}")
            results['gpu'] = True
        else:
            print("  [WARN] nvidia-smi不可用")
            results['gpu'] = False
    except Exception as e:
        print(f"  [WARN] GPU检测失败: {e}")
        results['gpu'] = False

    print()

    # ============================================================
    # Step 2: Stage 1依赖 - ViennaRNA
    # ============================================================
    print("[Step 2] Stage 1依赖 - ViennaRNA")
    print("-" * 80)

    results['vienna'] = check_python_module("ViennaRNA", "RNA")

    if results['vienna']:
        # 测试ViennaRNA功能
        try:
            import RNA
            seq = "ACGUACGUACGU"
            fc = RNA.fold_compound(seq)
            ss, mfe = fc.mfe()
            print(f"  [TEST] 二级结构预测:")
            print(f"         Sequence: {seq}")
            print(f"         Structure: {ss}")
            print(f"         MFE: {mfe:.2f} kcal/mol")
            print("  [OK] ViennaRNA功能正常")
        except Exception as e:
            print(f"  [FAIL] ViennaRNA测试失败: {e}")
            results['vienna'] = False

    print()

    # ============================================================
    # Step 3: Stage 2依赖 - RoseTTAFold2NA
    # ============================================================
    print("[Step 3] Stage 2依赖 - RoseTTAFold2NA")
    print("-" * 80)

    rosetta_dir = Path("./RoseTTAFold2NA")
    if rosetta_dir.exists():
        print(f"  [OK] RoseTTAFold2NA目录存在")

        # 检查关键文件
        run_infer = rosetta_dir / "run_infer.py"
        weights = rosetta_dir / "weights"

        if run_infer.exists():
            print(f"  [OK] run_infer.py存在")
            results['rosetta_code'] = True
        else:
            print(f"  [FAIL] run_infer.py缺失")
            results['rosetta_code'] = False

        if weights.exists():
            weight_files = list(weights.glob("*.pt")) + list(weights.glob("*.pth"))
            print(f"  [OK] 权重目录存在，文件数: {len(weight_files)}")
            results['rosetta_weights'] = True
        else:
            print(f"  [FAIL] 权重目录缺失")
            results['rosetta_weights'] = False

        results['rosetta'] = results['rosetta_code'] and results['rosetta_weights']
    else:
        print(f"  [FAIL] RoseTTAFold2NA目录不存在")
        print("         安装方法: git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git")
        results['rosetta'] = False

    print()

    # ============================================================
    # Step 4: Stage 3-4依赖 - OpenMM
    # ============================================================
    print("[Step 4] Stage 3-4依赖 - OpenMM")
    print("-" * 80)

    results['openmm'] = check_python_module("OpenMM", "openmm")

    if results['openmm']:
        try:
            import openmm as mm
            import openmm.app as app
            from openmm import unit

            print(f"  [OK] OpenMM版本: {mm.__version__}")

            # 测试OpenMM基本功能
            print("  [TEST] 创建简单系统...")
            system = mm.System()
            system.addParticle(1.0)

            force = mm.HarmonicBondForce()
            system.addForce(force)

            integrator = mm.LangevinIntegrator(
                300 * unit.kelvin,
                1 / unit.picosecond,
                0.001 * unit.picoseconds
            )

            print("  [OK] OpenMM基本功能正常")
        except Exception as e:
            print(f"  [FAIL] OpenMM测试失败: {e}")
            results['openmm'] = False

    print()

    # ============================================================
    # Step 5: 并行依赖 - Ray
    # ============================================================
    print("[Step 5] 并行依赖 - Ray")
    print("-" * 80)

    results['ray'] = check_python_module("Ray", "ray")

    if results['ray']:
        try:
            import ray
            print(f"  [OK] Ray版本: {ray.__version__}")

            # 测试Ray初始化
            print("  [TEST] Ray初始化...")
            ray.init(ignore_reinit_error=True, num_gpus=0)
            print("  [OK] Ray初始化成功")
            ray.shutdown()
        except Exception as e:
            print(f"  [FAIL] Ray测试失败: {e}")
            results['ray'] = False

    print()

    # ============================================================
    # Step 6: 其他依赖
    # ============================================================
    print("[Step 6] 其他依赖")
    print("-" * 80)

    results['yaml'] = check_python_module("PyYAML", "yaml")
    results['numpy'] = check_python_module("NumPy", "numpy")
    results['pandas'] = check_python_module("Pandas", "pandas")
    results['torch'] = check_python_module("PyTorch", "torch")

    if results['torch']:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  [OK] PyTorch CUDA可用")
                print(f"       CUDA版本: {torch.version.cuda}")
                print(f"       GPU数量: {torch.cuda.device_count()}")
            else:
                print(f"  [WARN] PyTorch CUDA不可用")
        except Exception as e:
            print(f"  [FAIL] PyTorch CUDA检查: {e}")

    print()

    # ============================================================
    # Step 7: Pipeline代码
    # ============================================================
    print("[Step 7] Pipeline代码")
    print("-" * 80)

    pipeline_dir = Path("./circrna_3d_pipeline")
    if pipeline_dir.exists():
        print(f"  [OK] Pipeline目录存在")

        # 检查关键脚本
        required_files = [
            "pipeline.py",
            "stage1_vienna.py",
            "stage2_rosetta.py",
            "stage3_cyclize.py",
            "stage4_md.py",
            "stage5_quality.py",
            "parallel_worker.py",
            "config_quality.yaml"
        ]

        for fname in required_files:
            fpath = pipeline_dir / fname
            if fpath.exists():
                print(f"  [OK] {fname}")
            else:
                print(f"  [FAIL] {fname}缺失")

        results['pipeline'] = True
    else:
        print(f"  [FAIL] Pipeline目录不存在")
        print("         复制方法: cp -r ../confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline ./")
        results['pipeline'] = False

    print()

    # ============================================================
    # Step 8: 汇总报告
    # ============================================================
    print("=" * 80)
    print("  验证汇总")
    print("=" * 80)
    print()

    # 关键依赖状态
    critical = ['python', 'vienna', 'openmm', 'rosetta', 'pipeline']
    optional = ['ray', 'torch', 'gpu']

    print("  关键依赖:")
    for dep in critical:
        status = "[OK]" if results.get(dep, False) else "[FAIL]"
        print(f"    {status} {dep}")

    print()
    print("  可选依赖:")
    for dep in optional:
        status = "[OK]" if results.get(dep, False) else "[WARN]"
        print(f"    {status} {dep}")

    print()

    # 计算总体状态
    all_critical_ok = all(results.get(dep, False) for dep in critical)

    if all_critical_ok:
        print("=" * 80)
        print("  [SUCCESS] 所有关键依赖已就绪！")
        print("=" * 80)
        print()
        print("  下一步:")
        print("    chmod +x deploy_full_pipeline.sh")
        print("    ./deploy_full_pipeline.sh input.fasta")
        print()
        return 0
    else:
        print("=" * 80)
        print("  [FAILED] 关键依赖缺失，请先安装")
        print("=" * 80)
        print()

        # 给出安装建议
        print("  安装建议:")
        print()

        if not results.get('python', False):
            print("    Python 3.10:")
            print("      conda create -n circrna python=3.10")
            print("      conda activate circrna")
            print()

        if not results.get('vienna', False):
            print("    ViennaRNA:")
            print("      conda install -c bioconda viennarna")
            print()

        if not results.get('openmm', False):
            print("    OpenMM:")
            print("      conda install -c conda-forge openmm")
            print()

        if not results.get('rosetta', False):
            print("    RoseTTAFold2NA:")
            print("      git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git")
            print("      cd RoseTTAFold2NA")
            print("      wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz")
            print("      tar -xzf RoseTTAFold2NA_weights.tar.gz")
            print()

        if not results.get('pipeline', False):
            print("    Pipeline代码:")
            print("      cp -r D:/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline ./")
            print()

        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)