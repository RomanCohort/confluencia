#!/usr/bin/env python3
"""Stage 1 only: ViennaRNA secondary structure prediction"""

import sys
sys.path.append('circrna_3d_pipeline')

from stage1_vienna import ViennaRNAPredictor
import os
import yaml
import json

def run_stage1_only(fasta_path, output_dir):
    """只运行Stage 1: ViennaRNA"""

    # 加载配置
    with open('config_quality.yaml') as f:
        config = yaml.safe_load(f)

    # 初始化Stage 1
    stage1 = ViennaRNAPredictor(config['vienna'])

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
                if current_seq:
                    sequences.append(current_seq)
                    bsj_positions.append((bsj_start, bsj_end))
                    seq_ids.append(current_id)

                header = line[1:]
                parts = header.split()
                current_id = parts[0]

                for part in parts:
                    if part.startswith('bsj_start='):
                        bsj_start = int(part.split('=')[1])
                    elif part.startswith('bsj_end='):
                        bsj_end = int(part.split('=')[1])

                current_seq = ""
            else:
                current_seq += line

        if current_seq:
            sequences.append(current_seq)
            bsj_positions.append((bsj_start, bsj_end))
            seq_ids.append(current_id)

    print(f"\n{'='*70}")
    print(f"Stage 1: ViennaRNA Secondary Structure Prediction")
    print(f"Total sequences: {len(sequences)}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)

    # 处理每条序列
    for i, (seq_id, seq, (bsj_start, bsj_end)) in enumerate(zip(seq_ids, sequences, bsj_positions)):
        print(f"\n[{i+1}/{len(sequences)}] {seq_id} ({len(seq)} nt)")

        seq_dir = os.path.join(output_dir, seq_id)
        os.makedirs(seq_dir, exist_ok=True)

        # Stage 1: ViennaRNA
        ss_result = stage1.predict(seq, bsj_start, bsj_end)

        # 保存结果
        result_path = os.path.join(seq_dir, 'stage1_result.json')
        with open(result_path, 'w') as f:
            json.dump({
                'seq_id': seq_id,
                'sequence': seq,
                'bsj_start': bsj_start,
                'bsj_end': bsj_end,
                'dot_bracket': ss_result['dot_bracket'],
                'mfe': ss_result['mfe']
            }, f, indent=2)

        print(f"  ✓ Saved to {result_path}")

        if (i + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"Progress: {i+1}/{len(sequences)} ({(i+1)/len(sequences)*100:.1f}%)")
            print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"Stage 1 COMPLETED: {len(sequences)} sequences")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default='circbase_filtered_5000.fa')
    parser.add_argument('--output', default='stage1_output')
    args = parser.parse_args()

    run_stage1_only(args.fasta, args.output)