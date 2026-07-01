#!/usr/bin/env python3
"""
data_chunking.py - 数据分块与权重分配（Phase 0 of training strategy）

根据training_strategy_v2.md实现：
- 数据来源权重映射
- 训练/验证/测试集划分
- 高质量数据优先分配
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 数据来源权重映射（来自training_strategy_v2.md）
WEIGHT_MAP = {
    'pdb_circularized': 0.0,  # 仅验证集
    'pdb3d': 0.0,             # 仅验证集
    'shape_experimental': 1.5,
    'shape_expanded': 1.2,
    'rfam_consensus': 1.0,
    'trrosetta_predicted': 1.0,
    'synthetic': 0.5,
    'vienna_fallback': 0.3,
}

def chunk_and_weight_data(filtered_data: List[Dict], weight_map: Dict[str, float]):
    """根据数据来源分配训练权重并划分数据集"""
    train_data = []
    val_data = []
    test_data = []

    for item in filtered_data:
        source = item['source']
        weight = weight_map.get(source, 1.0)

        if source in ['pdb_circularized', 'pdb3d']:
            val_data.append(item)
        elif weight == 0.0:
            test_data.append(item)
        else:
            train_data.append({**item, 'loss_weight': weight})

    return train_data, val_data, test_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    train, val, test = chunk_and_weight_data(data, WEIGHT_MAP)

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    with open(output_dir/'train.json', 'w') as f:
        json.dump(train, f)
    with open(output_dir/'val.json', 'w') as f:
        json.dump(val, f)
    with open(output_dir/'test.json', 'w') as f:
        json.dump(test, f)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")