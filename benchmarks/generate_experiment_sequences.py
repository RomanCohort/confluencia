"""
生成实验验证用的circRNA序列

根据免疫评分预测生成：
1. 高免疫评分序列 (>0.7) - 用于疫苗
2. 低免疫评分序列 (<0.4) - 用于治疗性载体
3. m6A修饰版对比
4. 高IRES评分序列

AutoDL运行命令：
cd /root/autodl-tmp/confluencia && git pull && python benchmarks/generate_experiment_sequences.py
"""

import os
import sys

# 自动检测路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# 添加正确的路径
sys.path.insert(0, root_dir)

from confluencia_circrna.core import (
    predict_circrna_immunogenicity,
    ImmuneSensingConfig,
    predict_modifications,
    compute_cirrna_objectives,
)
import numpy as np
import pandas as pd

print("=" * 60)
print("实验验证序列生成")
print("Confluencia circRNA v2.5")
print("=" * 60)

# ============================================================================
# 序列生成函数
# ============================================================================

def generate_gc_rich_sequence(length=500, gc_content=0.85):
    """生成高GC含量序列（高免疫评分）"""
    np.random.seed(42)
    gc_nt = ['G', 'C']
    au_nt = ['A', 'U']

    gc_count = int(length * gc_content)
    au_count = length - gc_count

    seq = []
    seq.extend(np.random.choice(gc_nt, gc_count).tolist())
    seq.extend(np.random.choice(au_nt, au_count).tolist())
    np.random.shuffle(seq)

    return ''.join(seq)


def generate_au_rich_sequence(length=500, gc_content=0.35):
    """生成AU富集序列（低免疫评分）"""
    np.random.seed(43)
    gc_nt = ['G', 'C']
    au_nt = ['A', 'U']

    gc_count = int(length * gc_content)
    au_count = length - gc_count

    seq = []
    seq.extend(np.random.choice(gc_nt, gc_count).tolist())
    seq.extend(np.random.choice(au_nt, au_count).tolist())
    np.random.shuffle(seq)

    return ''.join(seq)


def add_m6a_motifs(sequence, n_motifs=10):
    """添加DRACH m6A motif (D=A/G/U, R=A/G, H=A/C/U)"""
    seq = list(sequence)
    motifs = ['GGACU', 'AGACU', 'UGACU', 'GGACA', 'AGACA']

    np.random.seed(44)
    positions = np.random.choice(range(20, len(seq)-10), n_motifs, replace=False)

    for pos in sorted(positions):
        motif = np.random.choice(motifs)
        for i, nt in enumerate(motif):
            if pos + i < len(seq):
                seq[pos + i] = nt

    return ''.join(seq)


def add_ires_motifs(sequence, n_motifs=8):
    """添加IRES增强motif"""
    seq = list(sequence)
    ires_motifs = ['GCGCC', 'CCUG', 'GGGG', 'UUGU', 'AUGG', 'GGAAGG']

    np.random.seed(45)
    positions = np.random.choice(range(30, len(seq)-20), n_motifs, replace=False)

    for pos in sorted(positions):
        motif = np.random.choice(ires_motifs)
        for i, nt in enumerate(motif):
            if pos + i < len(seq):
                seq[pos + i] = nt

    return ''.join(seq)


def protect_backsplice_junction(sequence, junction_size=15):
    """保护backsplice junction区域"""
    seq = list(sequence)
    # 保持起始和结尾区域稳定
    junction_seq = 'GCGCGCGCGCGCGCG'  # 稳定junction序列

    # 替换起始junction
    for i in range(min(junction_size, len(seq))):
        seq[i] = junction_seq[i % len(junction_seq)]

    # 替换结尾junction
    for i in range(min(junction_size, len(seq))):
        seq[len(seq) - junction_size + i] = junction_seq[i % len(junction_seq)]

    return ''.join(seq)


# ============================================================================
# 生成实验序列
# ============================================================================

sequences = {}
config = ImmuneSensingConfig()

print("\n生成序列...")

# 1. 高免疫评分序列 (GC-rich, 无m6A)
print("\n[1] 高免疫评分序列 (GC-rich)...")
high_gc_seq = generate_gc_rich_sequence(500, gc_content=0.85)
high_gc_seq = protect_backsplice_junction(high_gc_seq)
immune_high = predict_circrna_immunogenicity(high_gc_seq, config)
obj_high = compute_cirrna_objectives(high_gc_seq, "none")

sequences['high_immune_1'] = {
    'sequence': high_gc_seq,
    'predicted_immune': immune_high['overall_immunogenicity'],
    'predicted_rig_i': immune_high['rig_i_score'],
    'predicted_pkr': immune_high['pkr_score'],
    'predicted_tlr': immune_high['tlr_score'],
    'modification': 'none',
    'gc_content': sum(1 for c in high_gc_seq if c in 'GC') / len(high_gc_seq),
    'purpose': 'PBMC验证 - 高免疫激活预期',
    'expected_IFN_alpha': 'high (3-5 fold vs control)',
}

# 2. 另一条高免疫序列
print("[2] 高免疫评分序列2 (dsRNA富集)...")
high_gc_seq2 = generate_gc_rich_sequence(500, gc_content=0.90)
high_gc_seq2 = protect_backsplice_junction(high_gc_seq2)
immune_high2 = predict_circrna_immunogenicity(high_gc_seq2, config)

sequences['high_immune_2'] = {
    'sequence': high_gc_seq2,
    'predicted_immune': immune_high2['overall_immunogenicity'],
    'predicted_rig_i': immune_high2['rig_i_score'],
    'predicted_pkr': immune_high2['pkr_score'],
    'predicted_tlr': immune_high2['tlr_score'],
    'modification': 'none',
    'gc_content': sum(1 for c in high_gc_seq2 if c in 'GC') / len(high_gc_seq2),
    'purpose': 'RIG-I/PKR验证 - 高激活预期',
}

# 3. 低免疫评分序列 (AU-rich, 有m6A)
print("[3] 低免疫评分序列 (AU-rich + m6A)...")
low_gc_seq = generate_au_rich_sequence(500, gc_content=0.35)
low_gc_seq = add_m6a_motifs(low_gc_seq, n_motifs=12)
low_gc_seq = protect_backsplice_junction(low_gc_seq)
immune_low = predict_circrna_immunogenicity(low_gc_seq, config)
obj_low = compute_cirrna_objectives(low_gc_seq, "m6A")

sequences['low_immune_1'] = {
    'sequence': low_gc_seq,
    'predicted_immune': immune_low['overall_immunogenicity'],
    'predicted_rig_i': immune_low['rig_i_score'],
    'predicted_pkr': immune_low['pkr_score'],
    'predicted_tlr': immune_low['tlr_score'],
    'modification': 'm6A',
    'gc_content': sum(1 for c in low_gc_seq if c in 'GC') / len(low_gc_seq),
    'purpose': 'PBMC验证 - 低免疫激活预期',
    'expected_IFN_alpha': 'low (1-1.5 fold vs control)',
}

# 4. 另一条低免疫序列
print("[4] 低免疫评分序列2...")
low_gc_seq2 = generate_au_rich_sequence(500, gc_content=0.30)
low_gc_seq2 = add_m6a_motifs(low_gc_seq2, n_motifs=15)
low_gc_seq2 = protect_backsplice_junction(low_gc_seq2)
immune_low2 = predict_circrna_immunogenicity(low_gc_seq2, config)

sequences['low_immune_2'] = {
    'sequence': low_gc_seq2,
    'predicted_immune': immune_low2['overall_immunogenicity'],
    'predicted_rig_i': immune_low2['rig_i_score'],
    'predicted_pkr': immune_low2['pkr_score'],
    'predicted_tlr': immune_low2['tlr_score'],
    'modification': 'm6A',
    'gc_content': sum(1 for c in low_gc_seq2 if c in 'GC') / len(low_gc_seq2),
    'purpose': '对照 - 极低免疫激活预期',
}

# 5. m6A修饰对比（同序列不同修饰）
print("[5] m6A修饰对比序列...")
base_seq = generate_gc_rich_sequence(500, gc_content=0.55)
base_seq = protect_backsplice_junction(base_seq)
immune_none = predict_circrna_immunogenicity(base_seq, config)

# 同序列加m6A
seq_with_m6a = add_m6a_motifs(base_seq, n_motifs=10)
immune_m6a = predict_circrna_immunogenicity(seq_with_m6a, config)

sequences['m6a_comparison_none'] = {
    'sequence': base_seq,
    'predicted_immune': immune_none['overall_immunogenicity'],
    'modification': 'none',
    'purpose': 'm6A效果验证 - 无修饰对照',
}

sequences['m6a_comparison_m6a'] = {
    'sequence': seq_with_m6a,
    'predicted_immune': immune_m6a['overall_immunogenicity'],
    'modification': 'm6A',
    'purpose': 'm6A效果验证 - 有修饰',
    'expected': '免疫激活低于无修饰版',
}

# 6. 高IRES序列
print("[6] 高IRES评分序列...")
ires_seq = generate_gc_rich_sequence(500, gc_content=0.50)
ires_seq = add_ires_motifs(ires_seq, n_motifs=10)
ires_seq = protect_backsplice_junction(ires_seq)
mod_result = predict_modifications(ires_seq)

sequences['high_ires'] = {
    'sequence': ires_seq,
    'ires_sites': len(mod_result.ires_sites),
    'translation_potential': mod_result.translation_potential,
    'purpose': 'IRES验证 - 高翻译效率预期',
}

# ============================================================================
# 输出结果
# ============================================================================

print("\n" + "=" * 60)
print("生成的实验序列")
print("=" * 60)

results = []
for name, data in sequences.items():
    seq = data['sequence']
    gc = sum(1 for c in seq if c in 'GC') / len(seq)

    results.append({
        'Name': name,
        'Length': len(seq),
        'GC_content': gc,
        'Predicted_Immune': data.get('predicted_immune', '-'),
        'Modification': data.get('modification', '-'),
        'Purpose': data['purpose'],
    })

df = pd.DataFrame(results)
print(df.to_string())

print("\n" + "=" * 60)
print("详细序列信息")
print("=" * 60)

for name, data in sequences.items():
    print(f"\n【{name}】")
    print(f"用途: {data['purpose']}")
    print(f"长度: {len(data['sequence'])} nt")
    print(f"GC含量: {sum(1 for c in data['sequence'] if c in 'GC')/len(data['sequence']):.2f}")
    print(f"修饰: {data.get('modification', 'none')}")

    if 'predicted_immune' in data:
        print(f"预测免疫评分: {data['predicted_immune']:.3f}")
        rig_i = data.get('predicted_rig_i')
        pkr = data.get('predicted_pkr')
        tlr = data.get('predicted_tlr')
        if rig_i is not None:
            print(f"  - RIG-I: {rig_i:.3f}")
        if pkr is not None:
            print(f"  - PKR: {pkr:.3f}")
        if tlr is not None:
            print(f"  - TLR: {tlr:.3f}")

    if 'expected_IFN_alpha' in data:
        print(f"预期IFN-α: {data['expected_IFN_alpha']}")

    print(f"\n序列 (前100nt):")
    print(data['sequence'][:100] + "...")
    print("-" * 40)

# ============================================================================
# 保存文件
# ============================================================================

# 保存序列文件
print("\n保存序列文件...")

with open('experiment_sequences.fasta', 'w') as f:
    for name, data in sequences.items():
        f.write(f">{name}\n")
        f.write(f"# Purpose: {data['purpose']}\n")
        f.write(f"# Modification: {data.get('modification', 'none')}\n")
        if 'predicted_immune' in data:
            f.write(f"# Predicted_Immune: {data['predicted_immune']:.3f}\n")
        f.write(data['sequence'] + "\n\n")

print("✓ experiment_sequences.fasta")

# 保存CSV
df.to_csv('experiment_sequences_summary.csv', index=False)
print("✓ experiment_sequences_summary.csv")

# 保存详细JSON
import json
with open('experiment_sequences_details.json', 'w') as f:
    json.dump(sequences, f, indent=2)
print("✓ experiment_sequences_details.json")

print("\n" + "=" * 60)
print("完成！文件已生成")
print("=" * 60)
print("\n提供给医学院的序列文件:")
print("  - experiment_sequences.fasta (合成用)")
print("  - experiment_sequences_summary.csv (汇总)")
print("  - experiment_sequences_details.json (详细信息)")
print("\n实验设计:")
print("  高免疫评分序列 → 预期高IFN-α激活")
print("  低免疫评分序列 → 预期低IFN-α激活")
print("  m6A对比 → 验证修饰降低免疫激活")