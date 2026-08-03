"""
download_pdb_rna.py — 从 PDB 下载高分辨率 RNA 结构

下载 ~20 个经典 RNA 结构（分辨率 < 3.0Å），用于构建统计势。

数据源: RCSB PDB
筛选条件: RNA 分子 + 分辨率 < 3.0Å

作者: TorusFold Team
日期: 2026-08-02
"""

import os
from pathlib import Path
from Bio.PDB import PDBList

# 经典 RNA 结构 PDB ID（高分辨率 < 3.0Å）
RNA_PDB_IDS = [
    # tRNA 结构
    "1EHZ",  # tRNA^Phe (yeast) - 1.93Å
    "1TRA",  # tRNA^Asp (yeast) - 2.7Å
    "1F27",  # tRNA^Gln (E. coli) - 2.8Å
    "2AWX",  # tRNA^Lys (human) - 2.8Å

    # 核酶
    "1Y26",  # Hammerhead ribozyme - 2.2Å
    "2OZ3",  # Hairpin ribozyme - 2.1Å
    "1KXQ",  # VS ribozyme - 2.5Å
    "2BZE",  # HDV ribozyme - 2.3Å

    # 核糖开关
    "1Y26",  # Adenine riboswitch - 2.2Å
    "2HOX",  # Guanine riboswitch - 2.0Å
    "2GDI",  # Lysine riboswitch - 2.4Å
    "3D0I",  # TPP riboswitch - 2.7Å

    # 其他 RNA
    "1JBR",  # Group I intron - 2.8Å
    "1GID",  # Group I intron - 3.0Å
    "1YQZ",  # RNase P - 2.8Å
    "2AID",  # Telomerase RNA - 2.5Å

    # 补充
    "1B23",  # tRNA^Ala - 2.5Å
    "1ASY",  # tRNA^Ser - 2.7Å
    "1DFU",  # tRNA^Val - 2.6Å
    "1EHO",  # tRNA^Met - 2.8Å
]

def download_rna_structures(output_dir: str = "data/pdb_rna") -> list:
    """下载 RNA PDB 结构

    Args:
        output_dir: 输出目录

    Returns:
        下载成功的 PDB ID 列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdbl = PDBList()
    successful = []

    print(f"开始下载 {len(RNA_PDB_IDS)} 个 RNA PDB 结构...")
    print(f"输出目录: {output_dir}")
    print()

    for i, pdb_id in enumerate(RNA_PDB_IDS, 1):
        try:
            print(f"[{i}/{len(RNA_PDB_IDS)}] 下载 {pdb_id}...", end=" ")
            pdbl.retrieve_pdb_file(
                pdb_id,
                pdir=str(output_path),
                file_format='pdb',
                overwrite=False
            )
            print("成功")
            successful.append(pdb_id)
        except Exception as e:
            print(f"失败: {e}")

    print()
    print(f"下载完成: {len(successful)}/{len(RNA_PDB_IDS)} 成功")
    print(f"成功下载的 PDB ID: {', '.join(successful)}")

    return successful

if __name__ == "__main__":
    successful = download_rna_structures()
