"""
从 IEDB 原始数据提取 MHC allele 信息
====================================
从 iedb_tcell_full_v3.zip 提取 epitope_seq 和 mhc_allele 的映射
"""
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parents[2]
ZIP_PATH = PROJECT / "confluencia-2.0-epitope" / "data" / "raw" / "iedb_tcell_full_v3.zip"
OUTPUT_PATH = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_allele_mapping.csv"

def extract_allele_mapping():
    print("Opening zip file...")
    z = zipfile.ZipFile(ZIP_PATH)
    csv_name = 'tcell_full_v3.csv'

    print(f"Reading {csv_name} (this may take a while)...")
    # 只读取需要的列
    usecols = ['Epitope.2', 'MHC Restriction', 'Epitope.6']  # sequence, allele, source organism

    # 分块读取以节省内存
    chunks = []
    for chunk in pd.read_csv(z.open(csv_name), chunksize=100000, low_memory=False):
        # 找到实际列名（可能有重复）
        cols = chunk.columns.tolist()

        # Epitope.2 = sequence (column index after header processing)
        # MHC Restriction = allele
        # 找到正确的列索引
        seq_col_idx = None
        allele_col_idx = None

        for i, c in enumerate(cols):
            if c == 'Epitope.2' and seq_col_idx is None:
                seq_col_idx = i
            if c == 'MHC Restriction' and allele_col_idx is None:
                allele_col_idx = i

        if seq_col_idx is not None and allele_col_idx is not None:
            chunk_data = chunk.iloc[:, [seq_col_idx, allele_col_idx]].copy()
            chunk_data.columns = ['epitope_seq', 'mhc_allele']
            # 过滤有效行
            chunk_data = chunk_data[
                chunk_data['epitope_seq'].notna() &
                ~chunk_data['epitope_seq'].isin(['Name', '']) &
                chunk_data['mhc_allele'].notna() &
                ~chunk_data['mhc_allele'].isin(['Name', ''])
            ]
            chunks.append(chunk_data)

    print(f"Read {len(chunks)} chunks, concatenating...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # 清理 allele 名称
    def clean_allele(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        # 处理多 allele 的情况（取第一个）
        if ';' in s:
            s = s.split(';')[0]
        # 规范化格式
        s = s.replace(' ', '')
        # 常见格式转换
        if s.startswith('HLA-') or s.startswith('H2-') or s.startswith('HLA'):
            return s
        # 尝试添加 HLA- 前缀
        if s.startswith('A*') or s.startswith('B*') or s.startswith('C*'):
            return 'HLA-' + s
        return s

    df['mhc_allele'] = df['mhc_allele'].apply(clean_allele)

    # 去重：每个序列取最常见的 allele
    print("Deduplicating by epitope_seq...")
    allele_counts = df.groupby(['epitope_seq', 'mhc_allele']).size().reset_index(name='count')
    allele_counts = allele_counts.sort_values('count', ascending=False)
    dedup = allele_counts.drop_duplicates(subset=['epitope_seq'], keep='first')

    print(f"Unique epitopes with allele: {len(dedup)}")
    print(f"Top alleles:\n{dedup['mhc_allele'].value_counts().head(10)}")

    # 保存
    dedup[['epitope_seq', 'mhc_allele']].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    return dedup

if __name__ == "__main__":
    extract_allele_mapping()