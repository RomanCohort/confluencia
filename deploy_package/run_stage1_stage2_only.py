#!/usr/bin/env python3
"""只运行Stage 1-2 (ViennaRNA + trRosettaRNA2) - GPU加速"""

import sys
sys.path.append('circrna_3d_pipeline')

from stage1_vienna import ViennaRNAPredictor
from stage2_trrosetta import trRosettaRNA2Predictor
import os
import yaml

def run_stage1_stage2_only(fasta_path, output_dir):
    # 加载配置
    with open('config_quality.yaml') as f:
        config = yaml.safe_load(f)

    # 初始化Stage 1和Stage 2
    stage1 = ViennaRNAPredictor(config['vienna'])
    stage2 = trRosettaRNA2Predictor(config['rosetta'])

    # 读取FASTA
    sequences = []
    bsj_positions = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                # 解析BSJ位置（假设在header中）
                parts = line.strip().split('_')
                bsj_start = int(parts[-2])
                bsj_end = int(parts[-1])
            else:
                sequences.append(line.strip())
                bsj_positions.append((bsj_start, bsj_end))

    # 处理每条序列
    for i, (seq, (bsj_start, bsj_end)) in enumerate(zip(sequences, bsj_positions)):
        print(f"\n[Seq {i}/{len(sequences)}] Processing...")

        seq_dir = os.path.join(output_dir, f'seq_{i}')
        os.makedirs(seq_dir, exist_ok=True)

        # Stage 1
        ss_result = stage1.predict(seq, bsj_start, bsj_end)

        # Stage 2
        linear_dir = os.path.join(seq_dir, 'linear')
        linear_results = stage2.predict(
            sequence=seq,
            dot_bracket=ss_result['dot_bracket'],
            bp_probs=ss_result['bp_probs'],
            output_dir=linear_dir
        )

        print(f"  ✓ Completed {i+1}/{len(sequences)}")

if __name__ == '__main__':
    run_stage1_stage2_only('circbase_filtered_5000.fa', 'stage1_stage2_only_output')