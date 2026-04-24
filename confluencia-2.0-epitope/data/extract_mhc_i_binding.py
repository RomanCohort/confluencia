"""
从 IEDB 原始数据提取 MHC-I 绑定训练数据
==========================================
使用 IEDB T-cell epitope 数据中的：
- Epitope.2: 肽段序列
- MHC Restriction: allele 名称
- Assay.5: Positive/Negative (绑定标签)
- MHC Restriction.4: MHC class I/II

只保留 MHC class I 数据 (与 NetMHCpan 对齐)
"""
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT / "data" / "raw" / "iedb_tcell_full_v3.zip"
OUTPUT_PATH = PROJECT / "data" / "iedb_mhc_i_binding.csv"


def extract_binding_data():
    print("Opening zip...")
    z = zipfile.ZipFile(ZIP_PATH)
    csv_name = 'tcell_full_v3.csv'

    print("Reading IEDB T-cell data (chunked)...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(z.open(csv_name), chunksize=200000, low_memory=False)):
        cols = chunk.columns.tolist()

        # 找关键列索引
        seq_idx = cols.index('Epitope.2')      # 序列
        allele_idx = cols.index('MHC Restriction')  # allele
        qual_idx = cols.index('Assay.5')         # Positive/Negative
        mhc_class_idx = cols.index('MHC Restriction.4')  # I/II

        sub = chunk.iloc[:, [seq_idx, allele_idx, qual_idx, mhc_class_idx]].copy()
        sub.columns = ['epitope_seq', 'mhc_allele', 'qualitative', 'mhc_class']
        chunks.append(sub)
        print(f"  Chunk {i+1}: {len(sub)} rows")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Total raw: {len(df)}")

    # 过滤
    # 1. 有效序列
    df = df[df['epitope_seq'].notna() & ~df['epitope_seq'].isin(['Name', ''])]
    # 2. 有效 allele
    df = df[df['mhc_allele'].notna() & ~df['mhc_allele'].isin(['Name', ''])]
    # 3. MHC class I only
    df = df[df['mhc_class'] == 'I']
    # 4. 有效 qualitative label
    df = df[df['qualitative'].isin(['Positive', 'Negative'])]

    print(f"After filtering (MHC-I + valid labels): {len(df)}")

    # 创建二值标签
    df['is_binder'] = (df['qualitative'] == 'Positive').astype(int)
    print(f"  Positive: {df['is_binder'].sum()} ({df['is_binder'].mean():.1%})")
    print(f"  Negative: {(~df['is_binder'].astype(bool)).sum()}")

    # 清理 allele 名称
    def clean_allele(s):
        if pd.isna(s):
            return None
        s = str(s).strip().replace(' ', '')
        if ';' in s:
            s = s.split(';')[0]
        if '*' in s and not s.startswith('HLA-') and s.startswith(('A', 'B', 'C')):
            s = 'HLA-' + s
        return s

    df['mhc_allele'] = df['mhc_allele'].apply(clean_allele)

    # 去重：同 peptide+allele 多条记录时取多数投票
    print("Deduplicating by (peptide, allele)...")
    dedup = df.groupby(['epitope_seq', 'mhc_allele']).agg(
        is_binder=('is_binder', 'mean'),
        count=('is_binder', 'count'),
    ).reset_index()
    dedup['is_binder'] = (dedup['is_binder'] >= 0.5).astype(int)
    dedup = dedup.drop(columns=['count'])

    print(f"Deduplicated: {len(dedup)} unique (peptide, allele) pairs")
    print(f"  Binders: {dedup['is_binder'].sum()} ({dedup['is_binder'].mean():.1%})")

    # 统计 allele 分布
    print(f"\nTop 15 alleles:")
    for a, cnt in dedup['mhc_allele'].value_counts().head(15).items():
        binder_rate = dedup[dedup['mhc_allele'] == a]['is_binder'].mean()
        print(f"  {a:20s}: {cnt:6d} (binder rate: {binder_rate:.1%})")

    # 保存
    dedup.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    return dedup


if __name__ == "__main__":
    extract_binding_data()