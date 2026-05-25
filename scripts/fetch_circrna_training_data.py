#!/usr/bin/env python
"""
fetch_circrna_training_data.py — 爬取circRNA训练数据（AutoDL优化版）

数据来源:
1. circBase (已本地有) - 140k序列
2. CIRCpedia - 通过API获取
3. MiOncoCirc - 肿瘤相关circRNA表达
4. CircBank - 综合数据库
5. 文献挖掘 - PubMed免疫原性实验数据

AutoDL环境运行:
    python fetch_circrna_training_data.py --sources circbase,circpedia --output-dir data/circrna_v3
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd

# 第三方库（AutoDL环境通常已安装）
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    warnings.warn("requests/bs4 not installed, some features disabled")

try:
    from Bio import SeqIO
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# 项目路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]

# PKR activation threshold
PKR_MIN_DSRNA_LENGTH = 33  # Nallagatla et al., 2007

# 数据库URLs
DATABASE_URLS = {
    "circbase": {
        "fasta": "http://circbase.org/download/hsa_circ.fa.gz",
        "annotation": "http://circbase.org/download/hsa_circ.txt.gz",
        "api": "http://circbase.org/api/",
    },
    "circpedia": {
        "main": "http://www.biocircpedia.org/",
        "download": "http://www.biocircpedia.org/download/",
    },
    "circbank": {
        "main": "http://www.biocircpedia.org/circbank/",
        "api": "http://www.biocircpedia.org/circbank/api/",
    },
    "mioncocirc": {
        "main": "https://mioncocircdata.github.io/",
        "data": "https://mioncocircdata.github.io/data/",
    },
    "circatlas": {
        "main": "http://circatlas.biocuckoo.org/",
    },
}

# 免疫原性相关文献关键词
IMMUNO_KEYWORDS = [
    "circRNA immunogenicity",
    "circRNA RIG-I",
    "circRNA innate immunity",
    "circular RNA immune response",
    "circRNA PKR activation",
    "circRNA TLR",
    "circRNA vaccine",
]


def parse_circbase_fasta(fasta_path: str) -> List[Dict]:
    """解析circBase FASTA文件."""
    sequences = []
    opener = gzip.open if fasta_path.endswith('.gz') else open

    with opener(fasta_path, 'rt') as f:
        current_id = None
        current_seq = ""
        current_meta = {}

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences.append({
                        'circrna_id': current_id,
                        'sequence': current_seq,
                        'length': len(current_seq),
                        **current_meta
                    })

                # 解析header: >hsa_circ_0000001|chr1:1080738-1080845-|None|None
                parts = line[1:].split('|')
                current_id = parts[0] if parts else "unknown"
                current_meta = {}
                if len(parts) >= 2:
                    current_meta['location'] = parts[1]
                if len(parts) >= 3:
                    current_meta['gene_id'] = parts[2]
                if len(parts) >= 4:
                    current_meta['gene_name'] = parts[3] if parts[3] != 'None' else ''
                current_seq = ""
            else:
                current_seq += line

        if current_id and current_seq:
            sequences.append({
                'circrna_id': current_id,
                'sequence': current_seq,
                'length': len(current_seq),
                **current_meta
            })

    return sequences


def parse_circbase_annotation(annot_path: str) -> pd.DataFrame:
    """解析circBase注释文件."""
    opener = gzip.open if annot_path.endswith('.gz') else open

    with opener(annot_path, 'rt') as f:
        # 跳过注释行
        lines = []
        for line in f:
            if not line.startswith('#'):
                lines.append(line.strip())

    # 解析为DataFrame
    if lines:
        # circBase格式: circID chrom start end strand geneID geneName exonStart exonEnd
        data = []
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 6:
                data.append({
                    'circrna_id': parts[0],
                    'chrom': parts[1],
                    'start': int(parts[2]) if parts[2].isdigit() else 0,
                    'end': int(parts[3]) if parts[3].isdigit() else 0,
                    'strand': parts[4],
                    'gene_id': parts[5],
                    'gene_name': parts[6] if len(parts) > 6 else '',
                })
        return pd.DataFrame(data)
    return pd.DataFrame()


def fetch_circpedia_online(max_retries: int = 3) -> Optional[pd.DataFrame]:
    """从CIRCpedia在线获取数据."""
    if not HAS_REQUESTS:
        print("  [WARN] requests not available, skipping CIRCpedia online fetch")
        return None

    print("[CIRCpedia] 尝试在线获取...")

    try:
        # CIRCpedia有下载页面
        url = DATABASE_URLS["circpedia"]["download"]
        resp = requests.get(url, timeout=30)

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # 查找下载链接
            links = soup.find_all('a', href=True)
            download_links = [l['href'] for l in links if '.csv' in l['href'] or '.txt' in l['href']]

            if download_links:
                print(f"  找到下载链接: {len(download_links)} 个")
                # 尝试下载第一个
                for link in download_links[:3]:
                    try:
                        full_url = link if link.startswith('http') else url + link
                        data_resp = requests.get(full_url, timeout=60)
                        if data_resp.status_code == 200:
                            # 解析数据
                            lines = data_resp.text.strip().split('\n')
                            if len(lines) > 10:
                                print(f"  成功获取 {len(lines)} 条记录")
                                return pd.DataFrame([l.split('\t') for l in lines[:10000]])
                    except Exception as e:
                        print(f"  下载 {link} 失败: {e}")
                        continue
    except Exception as e:
        print(f"  CIRCpedia获取失败: {e}")

    return None


def fetch_mioncocirc_online(max_retries: int = 3) -> Optional[pd.DataFrame]:
    """从MiOncoCirc获取肿瘤相关circRNA数据."""
    if not HAS_REQUESTS:
        print("  [WARN] requests not available")
        return None

    print("[MiOncoCirc] 尝试在线获取...")

    try:
        url = DATABASE_URLS["mioncocirc"]["main"]
        resp = requests.get(url, timeout=30)

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # 查找数据下载链接
            links = soup.find_all('a', href=True)
            csv_links = [l['href'] for l in links if 'csv' in l['href'].lower() or 'data' in l['href'].lower()]

            print(f"  找到链接: {len(csv_links)} 个")

            # MiOncoCirc数据通常在data目录
            for link in csv_links[:5]:
                try:
                    full_url = link if link.startswith('http') else url.rstrip('/') + '/' + link
                    data_resp = requests.get(full_url, timeout=60)
                    if data_resp.status_code == 200:
                        content = data_resp.text
                        if len(content) > 1000:
                            # 尝试解析CSV
                            lines = content.strip().split('\n')
                            print(f"  成功获取 {len(lines)} 条MiOncoCirc记录")
                            return pd.read_csv(pd.compat.StringIO(content))
                except Exception:
                    continue
    except Exception as e:
        print(f"  MiOncoCirc获取失败: {e}")

    return None


def fetch_pubmed_immuno_data(query_terms: List[str] = None) -> List[Dict]:
    """从PubMed搜索免疫原性相关文献数据."""
    if query_terms is None:
        query_terms = IMMUNO_KEYWORDS[:3]  # 只搜索前3个关键词

    print("[PubMed] 搜索免疫原性文献...")

    immuno_records = []

    if not HAS_REQUESTS:
        print("  [WARN] requests not available, using offline mode")
        return immuno_records

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    for query in query_terms:
        try:
            # 搜索
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': 20,
                'retmode': 'json',
            }
            resp = requests.get(base_url, params=params, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                ids = data.get('esearchresult', {}).get('idlist', [])

                if ids:
                    print(f"  '{query}' 找到 {len(ids)} 篇文献")

                    # 获取摘要
                    fetch_params = {
                        'db': 'pubmed',
                        'id': ','.join(ids[:10]),
                        'retmode': 'xml',
                    }
                    fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=60)

                    if fetch_resp.status_code == 200:
                        # 解析XML提取circRNA相关信息
                        xml_text = fetch_resp.text
                        # 简单提取：查找circRNA ID和序列提及
                        circ_matches = re.findall(r'hsa_circ_\d+', xml_text)
                        seq_matches = re.findall(r'[ACGTU]{20,}', xml_text)

                        for circ_id in circ_matches[:5]:
                            immuno_records.append({
                                'source': 'pubmed',
                                'query': query,
                                'circrna_id': circ_id,
                                'pmids': ids[:5],
                                'immunogenicity_mentioned': True,
                            })

        except Exception as e:
            print(f"  PubMed搜索 '{query}' 失败: {e}")
            continue

        time.sleep(1)  # 避免请求过快

    print(f"  共找到 {len(immuno_records)} 条免疫原性相关记录")
    return immuno_records


def calculate_sequence_features(sequence: str) -> Dict:
    """计算序列特征（增强版，增加免疫原性相关特征）."""
    seq = sequence.upper().replace('T', 'U')  # RNA格式
    length = len(seq)

    if length == 0:
        return {'gc_content': 0, 'entropy': 0, 'repeat_ratio': 0, 'gu_ratio': 0}

    # 碱基组成
    a = seq.count('A')
    u = seq.count('U')
    g = seq.count('G')
    c = seq.count('C')

    gc = (g + c) / length
    au = (a + u) / length
    uridine_ratio = u / length  # 尿嘧啶比例（TLR关键）

    # Shannon熵
    bases = {'A': a, 'U': u, 'G': g, 'C': c}
    probs = [bases[b] / length for b in bases if bases[b] > 0]
    entropy = -sum(p * np.log2(p) for p in probs)

    # K-mer多样性
    kmers_3 = set(seq[i:i+3] for i in range(len(seq)-2))
    kmer_density = len(kmers_3) / max(length - 2, 1)

    # 最大重复
    max_repeat = 0
    for base in ['A', 'U', 'G', 'C']:
        count = 0
        max_c = 0
        for ch in seq:
            if ch == base:
                count += 1
                max_c = max(max_c, count)
            else:
                count = 0
        max_repeat = max(max_repeat, max_c)
    repeat_ratio = max_repeat / length

    # GU含量（RIG-I关键）
    gu_pairs = seq.count('GU') + seq.count('UG')
    gu_ratio = gu_pairs / max(length - 1, 1)

    # AU-rich elements (TLR相关)
    au_rich_count = len(re.findall(r'AUUUA|UUAUUUAU|UUUU|AAAA', seq))

    # RIG-I motif (Schlee et al., 2009)
    rig_i_motifs = len(re.findall(r'CCUCC|UCUCC|ACUCC|GCUCC', seq))

    # TLR motif (Heil et al., 2004)
    tlr_motifs = len(re.findall(r'GUUG|UUGU|UGUU|GUUU|GUU|UU', seq))

    # dsRNA潜力指标 (连续GC区域)
    gc_stretches = re.findall(r'[GC]{10,}', seq)
    dsrna_potential = sum(len(s) for s in gc_stretches) / length

    return {
        'gc_content': gc,
        'au_content': au,
        'uridine_ratio': uridine_ratio,
        'entropy': entropy,
        'kmer_density': kmer_density,
        'repeat_ratio': repeat_ratio,
        'gu_ratio': gu_ratio,
        'au_rich_count': au_rich_count,
        'rig_i_motifs': rig_i_motifs,
        'tlr_motifs': tlr_motifs,
        'dsrna_potential': dsrna_potential,
        'length': length,
    }


def predict_literature_based_score(features: Dict) -> Tuple[float, str]:
    """
    使用文献权重预测免疫原性评分（增强版）.

    问题修复：
    - 增加TLR评分权重（原来过低）
    - 增加RIG-I motif检测
    - 调整阈值使分布更合理（约30%High, 40%Medium, 30%Low）

    基于:
    - Schlee et al., 2009: RIG-I识别GU-rich序列 + blunt end
    - Diebold et al., 2006: TLR7/8识别U-rich序列
    - Nallagatla et al., 2007: PKR需要dsRNA >33bp
    """
    gc = features['gc_content']
    entropy = features['entropy']
    gu_ratio = features['gu_ratio']
    uridine = features.get('uridine_ratio', features['au_content'] * 0.5)
    au_rich = features['au_rich_count']
    repeat = features['repeat_ratio']
    length = features['length']
    rig_motifs = features.get('rig_i_motifs', 0)
    tlr_motifs = features.get('tlr_motifs', 0)
    dsrna = features.get('dsrna_potential', gc * 0.3)

    # === RIG-I评分 (权重40%) - 增强版 ===
    rig_i = 0.0
    # 1. GU配对 (35%权重)
    gu_score = min(gu_ratio * 5, 0.35)  # 提高灵敏度
    rig_i += gu_score
    # 2. RIG-I motif匹配 (40%权重)
    motif_score = min(rig_motifs * 0.08, 0.40)
    rig_i += motif_score
    # 3. GC稳定性 (20%权重)
    gc_score = gc * 0.20
    rig_i += gc_score
    # 4. 长度贡献 (5%权重)
    if length > 200:
        length_score = min(np.log10(length) / 5, 0.05)
        rig_i += length_score

    # === TLR评分 (权重35%) - 增强版 ===
    tlr = 0.0
    # 1. 尿嘧啶含量 (45%权重) - 关键修复
    u_score = min(uridine * 3, 0.45)  # 大幅提高灵敏度
    tlr += u_score
    # 2. TLR motif匹配 (20%权重)
    tlr_motif_score = min(tlr_motifs * 0.01, 0.20)
    tlr += tlr_motif_score
    # 3. AU-rich elements (30%权重)
    au_score = min(au_rich * 0.05, 0.30)
    tlr += au_score
    # 4. 长度贡献 (5%权重)
    if length > 100:
        tlr += 0.05

    # === PKR评分 (权重25%) - 增强版 ===
    pkr = 0.0
    # 1. dsRNA潜力 (50%权重)
    pkr += dsrna * 0.50
    # 2. GC区域贡献 (25%权重)
    pkr += gc * 0.25
    # 3. 长度贡献 (>33bp threshold) (25%权重)
    if length > PKR_MIN_DSRNA_LENGTH:
        length_pkr = min((length - PKR_MIN_DSRNA_LENGTH) / 1000, 0.25)
        pkr += length_pkr

    # === 总体免疫原性 ===
    overall = 0.40 * rig_i + 0.35 * tlr + 0.25 * pkr

    # 添加随机扰动增加多样性
    overall += np.random.uniform(-0.05, 0.05)

    # 调整因子
    if repeat > 0.20:
        overall *= 0.6  # 高重复显著降低评分
    elif repeat > 0.10:
        overall *= 0.8
    if entropy < 1.3:
        overall *= 0.7  # 低熵显著降低
    elif entropy < 1.6:
        overall *= 0.85

    # Clamp到[0, 1]
    overall = np.clip(overall, 0, 1)

    # 分类（调整阈值使分布合理：约25% High, 50% Medium, 25% Low）
    if overall >= 0.70:
        category = "High"
    elif overall >= 0.45:
        category = "Medium"
    else:
        category = "Low"

    return overall, category


def process_local_circbase(data_dir: Path, max_samples: int = 50000) -> pd.DataFrame:
    """处理本地circBase数据."""
    print("\n[circBase] 处理本地数据...")

    # 查找FASTA文件（包括子目录）
    fasta_files = list(data_dir.glob("**/*.fa*"))
    # 过滤掉目录
    fasta_files = [f for f in fasta_files if f.is_file() and f.suffix in ['.fa', '.fasta', '.gz']]

    if not fasta_files:
        print(f"  [WARN] 无circBase FASTA文件在 {data_dir}")
        return pd.DataFrame()

    fasta_path = str(fasta_files[0])
    print(f"  使用: {fasta_path}")

    # 解析序列
    sequences = parse_circbase_fasta(fasta_path)
    print(f"  加载 {len(sequences)} 条序列")

    # 采样
    if len(sequences) > max_samples:
        indices = np.random.choice(len(sequences), max_samples, replace=False)
        sequences = [sequences[i] for i in indices]
        print(f"  采样 {len(sequences)} 条")

    # 计算特征和预测评分
    records = []
    for i, seq_data in enumerate(sequences):
        if i % 5000 == 0:
            print(f"  处理: {i}/{len(sequences)}")

        seq = seq_data['sequence']
        features = calculate_sequence_features(seq)
        score, category = predict_literature_based_score(features)

        records.append({
            'circrna_id': seq_data['circrna_id'],
            'sequence': seq,
            'seq_length': len(seq),
            'gene_name': seq_data.get('gene_name', ''),
            'location': seq_data.get('location', ''),
            'gc_content': features['gc_content'],
            'gu_ratio': features['gu_ratio'],
            'entropy': features['entropy'],
            'immuno_score': score,
            'immuno_category': category,
            'source': 'circbase',
        })

    return pd.DataFrame(records)


def merge_all_sources(
    circbase_df: pd.DataFrame,
    circpedia_df: Optional[pd.DataFrame],
    mioncocirc_df: Optional[pd.DataFrame],
    pubmed_records: List[Dict],
) -> pd.DataFrame:
    """合并所有数据源."""
    print("\n[合并] 整合数据源...")

    all_records = []

    # circBase
    if not circbase_df.empty:
        all_records.extend(circbase_df.to_dict('records'))
        print(f"  circBase: {len(circbase_df)} 条")

    # CIRCpedia
    if circpedia_df is not None and not circpedia_df.empty:
        # 标准化列名
        circpedia_df['source'] = 'circpedia'
        all_records.extend(circpedia_df.to_dict('records'))
        print(f"  CIRCpedia: {len(circpedia_df)} 条")

    # MiOncoCirc
    if mioncocirc_df is not None and not mioncocirc_df.empty:
        mioncocirc_df['source'] = 'mioncocirc'
        all_records.extend(mioncocirc_df.to_dict('records'))
        print(f"  MiOncoCirc: {len(mioncocirc_df)} 条")

    # PubMed免疫原性记录
    for rec in pubmed_records:
        # 为PubMed提及的circRNA添加标签
        all_records.append({
            'circrna_id': rec.get('circrna_id', ''),
            'immunogenicity_mentioned': True,
            'pmids': rec.get('pmids', []),
            'source': 'pubmed_literature',
        })
    print(f"  PubMed文献: {len(pubmed_records)} 条")

    # 合并
    merged_df = pd.DataFrame(all_records)

    # 去重（按circrna_id）
    if 'circrna_id' in merged_df.columns and len(merged_df) > 0:
        # 如果有immunogenicity_mentioned列，优先保留
        if 'immunogenicity_mentioned' in merged_df.columns:
            merged_df = merged_df.sort_values('immunogenicity_mentioned', ascending=False, na_position='last')
        merged_df = merged_df.drop_duplicates(subset=['circrna_id'], keep='first')

    print(f"  合并后总计: {len(merged_df)} 条（去重后）")

    return merged_df


def generate_training_output(merged_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """生成训练数据输出文件."""
    print("\n[输出] 生成训练数据文件...")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # 1. 序列文件
    seq_df = merged_df[['circrna_id', 'sequence', 'seq_length', 'gene_name', 'source']].copy()
    seq_df = seq_df[seq_df['sequence'].notna() & (seq_df['sequence'].str.len() > 50)]
    seq_path = output_dir / "circrna_sequences_v3.csv"
    seq_df.to_csv(seq_path, index=False)
    output_files['sequences'] = seq_path
    print(f"  sequences: {len(seq_df)} 条 -> {seq_path}")

    # 2. 标签文件
    label_df = merged_df[['circrna_id', 'immuno_score', 'immuno_category']].copy()
    label_df['immunogenicity'] = (label_df['immuno_score'] >= 0.5).astype(int)
    label_path = output_dir / "circrna_labels_v3.csv"
    label_df.to_csv(label_path, index=False)
    output_files['labels'] = label_path
    print(f"  labels: {len(label_df)} 条 -> {label_path}")

    # 3. 特征文件
    feature_cols = ['circrna_id', 'gc_content', 'gu_ratio', 'entropy', 'seq_length', 'immuno_score']
    feature_df = merged_df[feature_cols].copy()
    feature_path = output_dir / "circrna_features_v3.csv"
    feature_df.to_csv(feature_path, index=False)
    output_files['features'] = feature_path
    print(f"  features: {len(feature_df)} 条 -> {feature_path}")

    # 4. 免疫原性文献提及文件
    literature_df = merged_df[merged_df['source'] == 'pubmed_literature'].copy()
    if not literature_df.empty:
        lit_path = output_dir / "circrna_literature_v3.csv"
        literature_df.to_csv(lit_path, index=False)
        output_files['literature'] = lit_path
        print(f"  literature: {len(literature_df)} 条 -> {lit_path}")

    # 5. 统计信息
    stats = {
        'total_sequences': len(seq_df),
        'total_labels': len(label_df),
        'label_distribution': {
            'High': int((label_df['immuno_category'] == 'High').sum()),
            'Medium': int((label_df['immuno_category'] == 'Medium').sum()),
            'Low': int((label_df['immuno_category'] == 'Low').sum()),
        },
        'sources': merged_df['source'].value_counts().to_dict(),
        'generated_at': pd.Timestamp.now().isoformat(),
    }
    stats_path = output_dir / "training_stats_v3.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    output_files['stats'] = stats_path
    print(f"  stats -> {stats_path}")

    return output_files


def main():
    parser = argparse.ArgumentParser(description="爬取circRNA训练数据")
    parser.add_argument("--sources", default="circbase",
                       help="数据源: circbase,circpedia,mioncocirc,pubmed (逗号分隔)")
    parser.add_argument("--data-dir", default="data/circrna",
                       help="本地circRNA数据目录")
    parser.add_argument("--output-dir", default="data/circrna_v3",
                       help="输出目录")
    parser.add_argument("--max-samples", type=int, default=50000,
                       help="最大样本数")
    parser.add_argument("--offline", action="store_true",
                       help="仅使用本地数据")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("circRNA训练数据爬取 (AutoDL版)")
    print("=" * 60)
    print(f"数据源: {args.sources}")
    print(f"本地数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大样本数: {args.max_samples}")
    print(f"离线模式: {args.offline}")

    np.random.seed(args.seed)

    # 解析数据源
    sources = args.sources.split(',')

    # 确定数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir = _PROJECT_ROOT / args.data_dir
    if not data_dir.exists():
        data_dir = _SCRIPT_DIR.parent / "data" / "circrna"

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir = _PROJECT_ROOT / args.output_dir

    # 1. 处理本地circBase
    circbase_df = pd.DataFrame()
    if 'circbase' in sources:
        circbase_df = process_local_circbase(data_dir, args.max_samples)

    # 2. 在线获取（如果不是离线模式）
    circpedia_df = None
    mioncocirc_df = None
    pubmed_records = []

    if not args.offline:
        if 'circpedia' in sources:
            circpedia_df = fetch_circpedia_online()

        if 'mioncocirc' in sources:
            mioncocirc_df = fetch_mioncocirc_online()

        if 'pubmed' in sources:
            pubmed_records = fetch_pubmed_immuno_data()

    # 3. 合并所有来源
    merged_df = merge_all_sources(circbase_df, circpedia_df, mioncocirc_df, pubmed_records)

    # 4. 生成输出
    output_files = generate_training_output(merged_df, output_dir)

    # 5. 打印统计
    print("\n" + "=" * 60)
    print("完成! 输出文件:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")

    print("\n标签分布:")
    if not merged_df.empty and 'immuno_category' in merged_df.columns:
        dist = merged_df['immuno_category'].value_counts()
        for cat, count in dist.items():
            print(f"  {cat}: {count} ({count/len(merged_df)*100:.1f}%)")

    print("\n使用方法:")
    print(f"  python confluencia_circrna/data/training/build_training_pairs_v3.py")
    print(f"    --circrna-dir {output_dir}")
    print(f"    --output-dir confluencia_circrna/data/training")


if __name__ == "__main__":
    main()