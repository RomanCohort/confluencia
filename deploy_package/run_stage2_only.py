#!/usr/bin/env python3
"""Stage 2 only: trRosettaRNA2 3D prediction (requires Stage 1 output)"""

import sys
sys.path.append('circrna_3d_pipeline')

from stage2_trrosetta import trRosettaRNA2Predictor
import os
import yaml
import json

def run_stage2_only(stage1_output_dir, output_dir):
    """只运行Stage 2: trRosettaRNA2 (读取Stage 1结果)"""

    # 加载配置
    with open('config_quality.yaml') as f:
        config = yaml.safe_load(f)

    # 初始化Stage 2
    stage2 = trRosettaRNA2Predictor(config['rosetta'])

    # 查找所有Stage 1结果文件
    seq_dirs = sorted([d for d in os.listdir(stage1_output_dir) if d.startswith('circ_')])

    print(f"\n{'='*70}")
    print(f"Stage 2: trRosettaRNA2 3D Prediction")
    print(f"Input: {stage1_output_dir}")
    print(f"Total sequences: {len(seq_dirs)}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)

    # 处理每条序列
    for i, seq_id in enumerate(seq_dirs):
        print(f"\n[{i+1}/{len(seq_dirs)}] {seq_id}")

        # 读取Stage 1结果
        stage1_result_path = os.path.join(stage1_output_dir, seq_id, 'stage1_result.json')
        with open(stage1_result_path) as f:
            stage1_result = json.load(f)

        # 跳过长序列（避免 OOM）— 默认 >2000nt 跳过，等大内存机器跑
        seq_len = len(stage1_result['sequence'])
        max_len = int(os.environ.get('STAGE2_MAX_SEQ_LEN', '2000'))
        if seq_len > max_len:
            print(f"  ⚠ 跳过长序列 ({seq_len}nt > {max_len}nt)，等大内存机器处理")
            # 写入跳过标记
            skip_path = os.path.join(output_dir, seq_id, 'SKIPPED_TOO_LONG.txt')
            os.makedirs(os.path.dirname(skip_path), exist_ok=True)
            with open(skip_path, 'w') as f:
                f.write(f"seq_len={seq_len}, max={max_len}\n")
            continue

        # 创建输出目录
        seq_dir = os.path.join(output_dir, seq_id)
        linear_dir = os.path.join(seq_dir, 'linear')
        os.makedirs(linear_dir, exist_ok=True)

        # Stage 2: trRosettaRNA2
        linear_results = stage2.predict(
            sequence=stage1_result['sequence'],
            dot_bracket=stage1_result['dot_bracket'],
            bp_probs=stage1_result.get('bp_probs', None),
            output_dir=linear_dir
        )

        # 保存Stage 2结果摘要
        stage2_summary_path = os.path.join(seq_dir, 'stage2_summary.json')
        with open(stage2_summary_path, 'w') as f:
            json.dump({
                'seq_id': seq_id,
                'num_structures': len(linear_results),
                'best_confidence': max(r['confidence'] for r in linear_results),
                'structures': [{'sample_id': r['sample_id'], 'confidence': r['confidence']} for r in linear_results]
            }, f, indent=2)

        print(f"  ✓ Generated {len(linear_results)} structures")
        print(f"  ✓ Best confidence: {max(r['confidence'] for r in linear_results):.2f}")

        if (i + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"Progress: {i+1}/{len(seq_dirs)} ({(i+1)/len(seq_dirs)*100:.1f}%)")
            print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"Stage 2 COMPLETED: {len(seq_dirs)} sequences")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='stage1_output')
    parser.add_argument('--output', default='stage2_output')
    args = parser.parse_args()

    run_stage2_only(args.input, args.output)