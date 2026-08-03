"""
statistical_potential.py — RNA 三级接触统计势

从 PDB RNA 结构提取非 WC 接触统计，构建知识驱动统计势，
用于 CG 力场的三级接触项。

统计势公式:
    E(d) = -kT * ln(P_observed(d) / P_reference(d))

其中:
    P_observed(d): 从 PDB 统计的距离分布
    P_reference(d): 参考分布（随机期望）

作者: TorusFold Team
日期: 2026-08-02
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class StatisticalPotential:
    """RNA 三级接触统计势

    从 PDB RNA 结构统计非 WC 碱基对接触，构建距离依赖的统计势。

    碱基对类型:
        - WC (Watson-Crick): A-U, G-C
        - Non-WC: 所有其他碱基对 (A-A, G-G, A-G, U-U, G-U wobble, etc.)

    距离范围: 0-15 Å, 分为 30 个 bin (0.5Å/bin)
    """

    def __init__(
        self,
        distance_bins: int = 30,
        max_distance: float = 15.0,
        kT: float = 2.479,  # kJ/mol at 298K
    ):
        """初始化统计势

        Args:
            distance_bins: 距离 bin 数量
            max_distance: 最大距离 (Å)
            kT: 热能 (kJ/mol)
        """
        self.distance_bins = distance_bins
        self.max_distance = max_distance
        self.kT = kT

        # 距离 bin 边界
        self.bins = np.linspace(0, max_distance, distance_bins + 1)

        # 统计势表: (base1, base2) -> energy array
        # base1, base2 排序后作为 key (对称)
        self.potential: Dict[Tuple[str, str], np.ndarray] = {}

        # 计数表: (base1, base2) -> count array
        self.counts: Dict[Tuple[str, str], np.ndarray] = {}

        # 参考计数 (用于归一化)
        self.reference_counts: np.ndarray = np.zeros(distance_bins)

    def extract_contacts_from_pdb(self, pdb_file: Path) -> List[Tuple[str, str, float]]:
        """从 PDB 文件提取非 WC 碱基对接触

        Args:
            pdb_file: PDB 文件路径

        Returns:
            接触列表: [(base1, base2, distance), ...]
        """
        from Bio.PDB import PDBParser, Selection

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('rna', str(pdb_file))

        contacts = []

        for model in structure:
            for chain in model:
                residues = [res for res in chain if res.id[0] == ' ']  # 标准残基

                for i, res1 in enumerate(residues):
                    for res2 in residues[i+1:]:
                        # 跳过相邻残基 (i, i+1, i+2)
                        seq_dist = abs(res1.id[1] - res2.id[1])
                        if seq_dist < 3:
                            continue

                        # 获取碱基类型
                        base1 = self._get_base_type(res1)
                        base2 = self._get_base_type(res2)

                        if base1 is None or base2 is None:
                            continue

                        # 计算 P 原子距离 (粗粒近似)
                        if 'P' in res1 and 'P' in res2:
                            p1 = res1['P'].coord
                            p2 = res2['P'].coord
                            distance = np.linalg.norm(p1 - p2)

                            if distance < self.max_distance:
                                contacts.append((base1, base2, distance))

        return contacts

    def _get_base_type(self, residue) -> str:
        """获取残基的碱基类型

        Args:
            residue: Biopython Residue 对象

        Returns:
            碱基类型 (A, U, G, C) 或 None
        """
        resname = residue.get_resname().strip()

        # RNA 标准残基
        rna_mapping = {
            'A': 'A', 'ADE': 'A', 'DA': 'A', 'DADE': 'A',
            'U': 'U', 'URA': 'U', 'DT': 'U', 'DURA': 'U', 'DTT': 'U',
            'G': 'G', 'GUA': 'G', 'DG': 'G', 'DGUA': 'G',
            'C': 'C', 'CYT': 'C', 'DC': 'C', 'DCYT': 'C',
        }

        return rna_mapping.get(resname)

    def build_from_pdb_files(self, pdb_files: List[Path]):
        """从多个 PDB 文件构建统计势

        Args:
            pdb_files: PDB 文件路径列表
        """
        print(f"从 {len(pdb_files)} 个 PDB 文件构建统计势...")

        total_contacts = 0

        for i, pdb_file in enumerate(pdb_files, 1):
            print(f"  [{i}/{len(pdb_files)}] 处理 {pdb_file.name}...", end=" ")

            contacts = self.extract_contacts_from_pdb(pdb_file)
            total_contacts += len(contacts)

            # 更新计数
            for base1, base2, distance in contacts:
                key = tuple(sorted([base1, base2]))
                if key not in self.counts:
                    self.counts[key] = np.zeros(self.distance_bins)

                bin_idx = np.digitize(distance, self.bins) - 1
                if 0 <= bin_idx < self.distance_bins:
                    self.counts[key][bin_idx] += 1

                # 参考计数 (所有接触)
                self.reference_counts[bin_idx] += 1

            print(f"{len(contacts)} 个接触")

        print(f"\n总计: {total_contacts} 个接触")

        # 构建统计势
        self._build_potential()

    def _build_potential(self):
        """从计数构建统计势

        E(d) = -kT * ln(P_observed(d) / P_reference(d))
        """
        print("构建统计势...")

        for key, counts in self.counts.items():
            # 归一化
            total = counts.sum()
            if total == 0:
                continue

            p_observed = counts / (total + 1e-10)

            # 参考分布 (均匀分布近似)
            total_ref = self.reference_counts.sum()
            p_reference = self.reference_counts / (total_ref + 1e-10)

            # 统计势
            # E = -kT * ln(P_obs / P_ref)
            # 避免 log(0)
            ratio = p_observed / (p_reference + 1e-10)
            energy = -self.kT * np.log(ratio + 1e-10)

            self.potential[key] = energy

        print(f"完成: {len(self.potential)} 个碱基对类型")

    def get_energy(self, base1: str, base2: str, distance: float) -> float:
        """获取统计势能量

        Args:
            base1: 碱基 1 (A, U, G, C)
            base2: 碱基 2 (A, U, G, C)
            distance: 距离 (Å)

        Returns:
            统计势能量 (kJ/mol)
        """
        key = tuple(sorted([base1, base2]))

        if key not in self.potential:
            return 0.0

        bin_idx = np.digitize(distance, self.bins) - 1

        if 0 <= bin_idx < len(self.potential[key]):
            return float(self.potential[key][bin_idx])

        return 0.0

    def save(self, output_file: Path):
        """保存统计势到文件

        Args:
            output_file: 输出文件路径
        """
        import pickle

        # 计算通用统计势 (所有碱基对类型平均)
        all_energies = []
        for key, energy in self.potential.items():
            all_energies.append(energy)

        if len(all_energies) > 0:
            avg_energy = np.mean(all_energies, axis=0)
        else:
            avg_energy = np.zeros(self.distance_bins)

        data = {
            'distance_bins': self.distance_bins,
            'max_distance': self.max_distance,
            'kT': self.kT,
            'bins': self.bins,
            'potential': self.potential,
            'counts': self.counts,
            'reference_counts': self.reference_counts,
            'avg_energy': avg_energy,  # 通用统计势
        }

        with open(str(output_file), 'wb') as f:
            pickle.dump(data, f)

        print(f"统计势已保存到: {output_file}")

    def load(self, input_file: Path):
        """从文件加载统计势

        Args:
            input_file: 输入文件路径
        """
        import pickle

        with open(str(input_file), 'rb') as f:
            data = pickle.load(f)

        self.distance_bins = data['distance_bins']
        self.max_distance = data['max_distance']
        self.kT = data['kT']
        self.bins = data['bins']
        self.potential = data['potential']
        self.counts = data['counts']
        self.reference_counts = data['reference_counts']

        print(f"统计势已加载: {input_file}")
        print(f"  碱基对类型: {len(self.potential)}")
        print(f"  距离 bins: {self.distance_bins} (0-{self.max_distance}Å)")


def main():
    """主函数: 下载 PDB + 构建统计势"""
    from pathlib import Path

    # PDB 目录
    pdb_dir = Path("data/pdb_rna")

    if not pdb_dir.exists():
        print(f"PDB 目录不存在: {pdb_dir}")
        print("请先运行 download_pdb_rna.py 下载 PDB 结构")
        return

    # 获取 PDB 文件列表
    pdb_files = list(pdb_dir.glob("*.ent")) + list(pdb_dir.glob("*.pdb"))

    if len(pdb_files) == 0:
        print(f"未找到 PDB 文件: {pdb_dir}")
        return

    print(f"找到 {len(pdb_files)} 个 PDB 文件")

    # 构建统计势
    stat_pot = StatisticalPotential()
    stat_pot.build_from_pdb_files(pdb_files)

    # 保存
    output_file = Path("data/statistical_potential.pkl")
    stat_pot.save(output_file)

    # 测试
    print("\n测试统计势:")
    test_cases = [
        ('A', 'U', 5.0),
        ('G', 'C', 6.0),
        ('A', 'A', 7.0),
        ('G', 'G', 8.0),
    ]

    for base1, base2, dist in test_cases:
        energy = stat_pot.get_energy(base1, base2, dist)
        print(f"  {base1}-{base2} @ {dist}Å: {energy:.2f} kJ/mol")


if __name__ == "__main__":
    main()
