#!/usr/bin/env python3
"""
简化版Pipeline测试脚本 - 无需ViennaRNA/RoseTTAFold2NA
使用几何约束求解器直接生成环化3D结构

适用场景：快速测试、云GPU验证、Pipeline调试
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "confluencia_3_0" / "core" / "circrna" / "torusfold"))

try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    HAS_OPENMM = True
    print("[OK] OpenMM已安装")
except ImportError:
    HAS_OPENMM = False
    print("[ERROR] OpenMM未安装，请运行: pip install openmm")
    sys.exit(1)

# ============================================================
# 简化版环化器（无需ViennaRNA）
# ============================================================

class SimpleCircRNA3DGenerator:
    """简化的circRNA 3D结构生成器"""

    def __init__(self):
        self.bsj_target_distance = 3.5  # Å
        self.bsj_restraint_k = 1000.0   # kJ/mol/nm²

    def generate_circular_rna(self, sequence, output_pdb=None):
        """
        从序列生成环化RNA的3D结构

        流程：
        1. 生成线性C3'坐标（简化的螺旋几何）
        2. 使用OpenMM进行BSJ环化
        3. 能量最小化

        Args:
            sequence: RNA序列字符串（ACGU）
            output_pdb: 输出PDB文件路径

        Returns:
            dict with 'pdb_path', 'bsj_distance', 'energy'
        """
        L = len(sequence)
        print(f"  生成 {L} nt circRNA结构...")

        # 1. 构建线性坐标（简化螺旋）
        coords = self._build_linear_helix(sequence)

        # 2. 创建PDB文件
        if output_pdb is None:
            output_pdb = f"circrna_{L}nt_linear.pdb"

        self._write_pdb(coords, sequence, output_pdb)

        # 3. OpenMM环化
        cyclized_pdb = output_pdb.replace('_linear.pdb', '_cyclized.pdb')
        result = self._cyclize_with_openmm(output_pdb, cyclized_pdb)

        return result

    def _build_linear_helix(self, sequence):
        """构建简化的RNA螺旋坐标"""
        L = len(sequence)

        # RNA-A螺旋参数（简化）
        rise_per_nt = 2.8  # Å (A-form rise)
        twist_per_nt = 32.7  # degrees

        coords = []
        for i in range(L):
            # C3'原子坐标（简化）
            angle = i * twist_per_nt * np.pi / 180
            x = 10.0 * np.cos(angle)  # radius ~10 Å
            y = 10.0 * np.sin(angle)
            z = i * rise_per_nt

            coords.append([x, y, z])

        return np.array(coords)

    def _write_pdb(self, coords, sequence, pdb_path):
        """写入简化PDB文件"""
        with open(pdb_path, 'w') as f:
            f.write("HEADER    circRNA linear structure\n")

            for i, (coord, base) in enumerate(zip(coords, sequence)):
                res_name = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'U': 'URA'}[base]

                # ATOM record (简化：仅C3'原子)
                f.write(f"ATOM  {i+1:5d}  C3' {res_name} A{i+1:4d}    ")
                f.write(f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}")
                f.write("  1.00  0.00           C\n")

            f.write("END\n")

        print(f"    写入线性PDB: {pdb_path}")

    def _cyclize_with_openmm(self, linear_pdb, output_pdb):
        """使用OpenMM进行BSJ环化"""
        print(f"    OpenMM环化...")

        # 加载PDB
        pdb = app.PDBFile(linear_pdb)

        # 创建力场（简化：仅使用 harmonic bonds）
        forcefield = app.ForceField('amber14-all.xml')

        # 创建系统
        try:
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds
            )
        except Exception as e:
            print(f"    [WARN] 力场创建失败: {e}")
            print(f"    使用简化系统...")

            # 创建简化系统（仅约束）
            system = mm.System()
            for pos in pdb.positions:
                system.addParticle(1.0)  # 单位质量

            # 添加谐波键（连接相邻核苷酸）
            bond_force = mm.HarmonicBondForce()
            bond_force.setBondParameters(0, 0.5, 0.5)  # k=0.5, r0=0.5 nm

            # 连接C3'原子（简化）
            for i in range(len(pdb.positions) - 1):
                dist = np.linalg.norm(
                    np.array(pdb.positions[i]) - np.array(pdb.positions[i+1])
                ) * 0.1  # Å to nm
                bond_force.addBond(i, i+1, dist, 100.0)  # k=100

            # BSJ连接
            bsj_dist = np.linalg.norm(
                np.array(pdb.positions[-1]) - np.array(pdb.positions[0])
            ) * 0.1
            bond_force.addBond(len(pdb.positions)-1, 0, self.bsj_target_distance * 0.1, self.bsj_restraint_k)

            system.addForce(bond_force)

        # 添加BSJ距离约束（正式方法）
        try:
            bsj_force = mm.CustomBondForce('k*(r - r0)^2')
            bsj_force.addPerBondParameter('k')
            bsj_force.addPerBondParameter('r0')

            # 连接最后一个和第一个核苷酸
            bsj_force.addBond(
                len(pdb.positions) - 1,  # 最后一个C3'
                0,                       # 第一个C3'
                [self.bsj_restraint_k, self.bsj_target_distance * 0.1]  # nm
            )

            system.addForce(bsj_force)
            print(f"    [OK] BSJ约束已添加 (k={self.bsj_restraint_k}, r0={self.bsj_target_distance} A)")
        except Exception as e:
            print(f"    [WARN] BSJ约束添加失败: {e}")

        # 创建模拟
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.001 * unit.picoseconds
        )

        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        # 能量最小化
        print(f"    能量最小化...")
        simulation.minimizeEnergy(maxIterations=500)

        # 获取最小化后的状态
        state = simulation.context.getState(
            getPositions=True,
            getEnergy=True
        )

        # 计算BSJ距离
        positions = state.getPositions()
        bsj_dist_nm = np.linalg.norm(
            np.array(positions[-1]) - np.array(positions[0])
        )
        bsj_dist_angstrom = bsj_dist_nm * 10

        energy = state.getPotentialEnergy()

        # 保存结果
        app.PDBFile.writeFile(pdb.topology, positions, open(output_pdb, 'w'))

        print(f"    [OK] 环化完成")
        print(f"      BSJ距离: {bsj_dist_angstrom:.2f} A")
        print(f"      能量: {energy._value:.1f} kJ/mol")

        return {
            'pdb_path': output_pdb,
            'bsj_distance': float(bsj_dist_angstrom),
            'energy_kjmol': float(energy._value),
            'sequence_length': len(pdb.positions)
        }

# ============================================================
# 主测试流程
# ============================================================

def run_test_pipeline():
    """运行完整测试流程"""

    print("=" * 60)
    print("  circRNA环化Pipeline测试")
    print("=" * 60)
    print()

    # 测试序列
    test_sequences = [
        ("ACGUACGUACGUACGUACGU" * 4, "circ_test_001"),  # 80 nt
        ("GCUAGCUAGCUAGCUAGCUA" * 5, "circ_test_002"),  # 100 nt
        ("AUGCAUGCAUGCAUGC" * 4, "circ_test_003"),      # 64 nt
    ]

    # 输出目录
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # 初始化生成器
    generator = SimpleCircRNA3DGenerator()

    results = []

    for seq, name in test_sequences:
        print(f"\n处理序列: {name} ({len(seq)} nt)")

        output_pdb = str(output_dir / f"{name}_cyclized.pdb")

        try:
            result = generator.generate_circular_rna(seq, output_pdb)
            result['name'] = name
            results.append(result)

            print(f"  [OK] 成功")
        except Exception as e:
            print(f"  [ERROR] 失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总报告
    print()
    print("=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    print()

    report = {
        'total_structures': len(results),
        'successful': len([r for r in results if 'error' not in r]),
        'failed': len([r for r in results if 'error' in r]),
        'average_bsj_distance': np.mean([r['bsj_distance'] for r in results if 'bsj_distance' in r]),
        'average_energy': np.mean([r['energy_kjmol'] for r in results if 'energy_kjmol' in r]),
        'details': results
    }

    print(f"  总结构数: {report['total_structures']}")
    print(f"  成功数: {report['successful']}")
    print(f"  平均BSJ距离: {report['average_bsj_distance']:.2f} A")
    print(f"  平均能量: {report['average_energy']:.1f} kJ/mol")

    # 保存报告
    report_path = output_dir / "test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  报告已保存: {report_path}")
    print(f"  PDB文件目录: {output_dir}")

    # 检查BSJ质量
    good_bsj = [r for r in results if r.get('bsj_distance', 999) < 4.0]

    print()
    print(f"  BSJ质量评估:")
    print(f"    理想范围: < 4.0 A")
    print(f"    达标数: {len(good_bsj)}/{len(results)}")

    if len(good_bsj) == len(results):
        print(f"    [OK] 全部BSJ达标！Pipeline工作正常")
    else:
        print(f"    [WARN] 部分BSJ超标，可能需要调整约束强度")

    print()
    print("=" * 60)
    print("  测试完成！")
    print("=" * 60)

    return report

# ============================================================
# 执行入口
# ============================================================

if __name__ == "__main__":
    print()
    print("开始运行简化版Pipeline测试...")
    print()

    try:
        report = run_test_pipeline()

        print()
        print("下一步:")
        print("  1. 查看生成的PDB文件:")
        print("     ls test_output/*.pdb")
        print()
        print("  2. 使用可视化软件查看结构:")
        print("     PyMOL: load test_output/circ_test_001_cyclized.pdb")
        print("     Chimera: open test_output/circ_test_001_cyclized.pdb")
        print()
        print("  3. 扩展到完整Pipeline:")
        print("     安装ViennaRNA: conda install -c bioconda viennarna")
        print("     安装RoseTTAFold2NA: git clone https://github.com/baker-laboratory/RoseTTAFold2NA")
        print("     运行完整pipeline: python pipeline.py --config config_quality.yaml")
        print()

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)