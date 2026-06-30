#!/usr/bin/env python3
"""只运行Stage 1-2 (ViennaRNA + trRosettaRNA2) - GPU加速"""

import sys
sys.path.append('circrna_3d_pipeline')

from stage1_vienna import ViennaRNAPredictor
from stage2_trrosetta import trRosettaRNA2Predictor
import os
import yaml
import re

def run_stage1_stage2_only(fasta_path, output_dir):
    """只运行Stage 1-2，跳过OpenMM"""

    # 加载配置
    with open('config_quality.yaml') as f:
        config = yaml.safe_load(f)

    # 初始化Stage 1和Stage 2
    stage1 = ViennaRNAPredictor(config['vienna'])
    stage2 = trRosettaRNA2Predictor(config['rosetta'])

    # 读取FASTA
    sequences = []
    bsj_positions = []
    seq_ids = []

    with open(fasta_path) as f:
        current_id = None
        current_seq = ""
        bsj_start = 0
        bsj_end = 0

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存上一条序列
                if current_seq:
                    sequences.append(current_seq)
                    bsj_positions.append((bsj_start, bsj_end))
                    seq_ids.append(current_id)

                # 解析header: >circ_000000 bsj_start=0 bsj_end=107
                header = line[1:]  # 去掉 >
                parts = header.split()

                current_id = parts[0]  # circ_000000

                # 提取bsj_start和bsj_end
                for part in parts:
                    if part.startswith('bsj_start='):
                        bsj_start = int(part.split('=')[1])
                    elif part.startswith('bsj_end='):
                        bsj_end = int(part.split('=')[1])

                current_seq = ""
            else:
                current_seq += line

        # 保存最后一条序列
        if current_seq:
            sequences.append(current_seq)
            bsj_positions.append((bsj_start, bsj_end))
            seq_ids.append(current_id)

    print(f"\n{'='*70}")
    print(f"Total sequences: {len(sequences)}")
    print(f"{'='*70}")

    # 处理每条序列
    for i, (seq_id, seq, (bsj_start, bsj_end)) in enumerate(zip(seq_ids, sequences, bsj_positions)):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(sequences)}] Processing {seq_id}")
        print(f"{'='*70}")
        print(f"  Length: {len(seq)} nt")
        print(f"  BSJ: {bsj_start}-{bsj_end}")

        seq_dir = os.path.join(output_dir, seq_id)
        os.makedirs(seq_dir, exist_ok=True)

        # Stage 1: ViennaRNA
        print(f"\n  [Stage 1] ViennaRNA secondary structure...")
        ss_result = stage1.predict(seq, bsj_start, bsj_end)

        # Stage 2: trRosettaRNA2
        print(f"\n  [Stage 2] trRosettaRNA2 3D prediction...")
        linear_dir = os.path.join(seq_dir, 'linear')
        linear_results = stage2.predict(
            sequence=seq,
            dot_bracket=ss_result['dot_bracket'],
            bp_probs=ss_result['bp_probs'],
            output_dir=linear_dir
        )

        print(f"\n  ✓ Completed {seq_id}")
        print(f"    Generated {len(linear_results)} structures")
        print(f"    Best confidence: {max(r['confidence'] for r in linear_results):.2f}")

        # 每10条序列打印总进度
        if (i + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"Progress: {i+1}/{len(sequences)} ({(i+1)/len(sequences)*100:.1f}%)")
            print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"ALL COMPLETED: {len(sequences)} sequences")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default='circbase_filtered_5000.fa')
    parser.add_argument('--output', default='stage1_stage2_only_output')
    args = parser.parse_args()

    run_stage1_stage2_only(args.fasta, args.output)