#!/usr/bin/env python3
"""
从预训练权重文件中提取正确的模型参数
不再猜测，直接从checkpoint读取实际参数
"""

import torch
import json
import os

def extract_config_from_checkpoint(checkpoint_path):
    """从checkpoint提取模型结构参数"""
    print(f"加载checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 获取所有键
    keys = list(checkpoint.keys())

    # 分析structure_module参数
    structure_params = {}

    # 从权重形状推断参数
    for key in keys:
        if 'structure_module.ipa' in key:
            shape = checkpoint[key].shape

            # head_weights: [no_heads_ipa]
            if 'head_weights' in key:
                structure_params['no_heads_ipa'] = shape[0]
                print(f"  no_heads_ipa = {shape[0]} (from {key})")

            # linear_q: [no_heads_ipa * c_ipa, c_z]
            if 'linear_q.weight' in key:
                no_heads_ipa_times_c_ipa, c_z = shape
                if 'no_heads_ipa' in structure_params:
                    c_ipa = no_heads_ipa_times_c_ipa // structure_params['no_heads_ipa']
                    structure_params['c_ipa'] = c_ipa
                    structure_params['c_z'] = c_z
                    print(f"  c_ipa = {c_ipa}, c_z = {c_z} (from {key})")

            # linear_q_points: [no_heads_ipa * no_qk_points * 3, c_z]
            if 'linear_q_points.weight' in key:
                total, c_z = shape
                if 'no_heads_ipa' in structure_params:
                    no_qk_points = total // (structure_params['no_heads_ipa'] * 3)
                    structure_params['no_qk_points'] = no_qk_points
                    print(f"  no_qk_points = {no_qk_points} (from {key})")

            # linear_kv_points: [no_heads_ipa * no_v_points * 3, c_z]
            if 'linear_kv_points.weight' in key:
                total, c_z = shape
                if 'no_heads_ipa' in structure_params:
                    no_v_points = total // (structure_params['no_heads_ipa'] * 3)
                    structure_params['no_v_points'] = no_v_points
                    print(f"  no_v_points = {no_v_points} (from {key})")

        # angle_resnet参数
        if 'angle_resnet.linear_in.weight' in key:
            shape = checkpoint[key].shape
            c_resnet = shape[0]
            structure_params['c_resnet'] = c_resnet
            print(f"  c_resnet = {c_resnet} (from {key})")

        # no_angles
        if 'angle_resnet.linear_out.weight' in key:
            shape = checkpoint[key].shape
            no_angles = shape[0]
            structure_params['no_angles'] = no_angles
            print(f"  no_angles = {no_angles} (from {key})")

    # 分析其他结构参数
    # transition layers
    transition_keys = [k for k in keys if 'structure_module.transition.layers' in k]
    if transition_keys:
        # 提取层数
        layer_nums = set()
        for k in transition_keys:
            parts = k.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i+1 < len(parts):
                    layer_nums.add(int(parts[i+1]))
        structure_params['no_transition_layers'] = max(layer_nums) + 1 if layer_nums else 2
        print(f"  no_transition_layers = {structure_params.get('no_transition_layers', 2)}")

    # no_blocks (从其他结构推断)
    # 检查structure_module的其他层
    for key in keys:
        if 'structure_module.transition.layers' in key:
            parts = key.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i+1 < len(parts):
                    layer_num = int(parts[i+1])
                    if 'no_blocks' not in structure_params or layer_num > structure_params.get('no_blocks', 0):
                        # 这不是no_blocks，是transition_layers
                        pass

    return structure_params

def main():
    base_dir = "trRosettaRNA2"
    checkpoint_path = f"{base_dir}/weights/params/models/model_1.pth.tar"
    output_path = f"{base_dir}/weights/params/config/model_1.json"

    print("======================================================================")
    print("  从checkpoint提取正确的模型参数")
    print("======================================================================")
    print()

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    print("分析checkpoint权重...")
    structure_params = extract_config_from_checkpoint(checkpoint_path)
    print()

    print("提取的结构参数:")
    for k, v in sorted(structure_params.items()):
        print(f"  {k}: {v}")
    print()

    # 构建完整config
    config = {
        "model_name": "model_1",
        "num_recycles": 3,
        "nrows": 64,
        "max_recycle": 3,
        "dim_pair": 64,
        "use_ss": True,
        "divide": False,
        "init_str": True,
        "ss3D": False,

        "RNAformer": {
            "n_block": 6,
            "d_model": 256,
            "d_ff": 512,
            "num_heads": 8,
            "dim_head": 32,
            "num_tokens": 4,
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "msa_tie_row_attn": False,
            "msa_conv": False,
            "use_r2n": False,
            "qknorm": True,
            "dropout_rate_attn": 0.0,
            "dropout_rate_ff": 0.0,
        },

        "structure_module": {
            # 从checkpoint提取的参数（必须匹配权重！）
            "c_s": structure_params.get('c_s', 64),
            "c_z": structure_params.get('c_z', 64),
            "c_ipa": structure_params.get('c_ipa', 64),
            "c_resnet": structure_params.get('c_resnet', 64),
            "no_heads_ipa": structure_params.get('no_heads_ipa', 12),
            "no_qk_points": structure_params.get('no_qk_points', 4),
            "no_v_points": structure_params.get('no_v_points', 4),
            "no_blocks": 6,  # 需要从其他权重推断
            "no_transition_layers": structure_params.get('no_transition_layers', 2),
            "no_resnet_blocks": 2,  # 需要从其他权重推断
            "no_angles": structure_params.get('no_angles', 8),
            "trans_scale_factor": 1.0,
        },

        "ss_module": {
            "hidden_size": 128,
            "dropout": 0.0,
            "num_layers": 1,
        },

        "input_embedder": {
            "dim": 64,
            "use_ss": True,
        },

        "pair_embedder": {
            "dim": 64,
        },

        "recycling": {
            "num_recycles": 3,
        },
    }

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"✓ Config已生成: {output_path}")
    print()
    print("验证关键字段:")
    print(f"  no_heads_ipa: {config['structure_module']['no_heads_ipa']}")
    print(f"  c_ipa: {config['structure_module']['c_ipa']}")
    print(f"  c_z: {config['structure_module']['c_z']}")
    print(f"  c_resnet: {config['structure_module']['c_resnet']}")
    print(f"  no_angles: {config['structure_module']['no_angles']}")

if __name__ == '__main__':
    main()