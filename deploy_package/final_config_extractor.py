#!/usr/bin/env python3
"""
完整分析并修复trRosettaRNA2 config与checkpoint的匹配问题
"""
import torch
import json
import os
import re

def analyze_and_fix():
    checkpoint_path = "trRosettaRNA2/weights/params/models/model_1.pth.tar"
    output_path = "trRosettaRNA2/weights/params/config/model_1.json"

    print("============================================")
    print("  分析checkpoint权重结构")
    print("============================================\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    keys = list(checkpoint.keys())

    # 提取所有structure_module相关参数
    params = {}

    for key in keys:
        if 'structure_module.ipa' in key:
            shape = checkpoint[key].shape
            name = key.split('.')[-1]

            if 'head_weights' in key:
                params['no_heads_ipa'] = shape[0]
                print(f"✓ {name}: {shape[0]}")

            elif 'linear_q' in key:
                if 'weight' in key:  # 只处理weight，跳过bias
                    no_heads_ipa_times_c_ipa, c_z = shape
                    if 'no_heads_ipa' in params:
                        c_ipa = no_heads_ipa_times_c_ipa // params['no_heads_ipa']
                        params['c_ipa'] = c_ipa
                        params['c_z'] = c_z
                        print(f"✓ {name}: c_ipa={c_ipa}, c_z={c_z}")

            elif 'linear_q_points' in key:
                if 'weight' in key:
                    total, c_z = shape
                    if 'no_heads_ipa' in params:
                        no_qk_points = total // (params['no_heads_ipa'] * 3)
                        params['no_qk_points'] = no_qk_points
                        print(f"✓ {name}: no_qk_points={no_qk_points}")

            elif 'linear_kv_points' in key:
                if 'weight' in key:
                    total, c_z = shape
                    if 'no_heads_ipa' in params:
                        no_v_points = total // (params['no_heads_ipa'] * 3)
                        params['no_v_points'] = no_v_points
                        print(f"✓ {name}: no_v_points={no_v_points}")

        elif 'structure_module.angle_resnet' in key:
            shape = checkpoint[key].shape
            name = key.split('.')[-1]

            if 'linear_in' in key:
                c_resnet = shape[0]
                params['c_resnet'] = c_resnet
                print(f"✓ {name}: c_resnet={c_resnet}")

            elif 'linear_out' in key:
                no_angles = shape[0]
                params['no_angles'] = no_angles
                print(f"✓ {name}: no_angles={no_angles}")

    # 构建最终config
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
            "n_block": 6, "d_model": 256, "d_ff": 512, "num_heads": 8,
            "dim_head": 32, "num_tokens": 4, "attn_dropout": 0.0,
            "ff_dropout": 0.0, "msa_tie_row_attn": False, "msa_conv": False,
            "use_r2n": False, "qknorm": True, "dropout_rate_attn": 0.0, "dropout_rate_ff": 0.0,
        },
        "structure_module": {
            "c_s": params.get('c_s', 64),
            "c_z": params.get('c_z', 64),
            "c_ipa": params.get('c_ipa', 64),
            "c_resnet": params.get('c_resnet', 64),
            "no_heads_ipa": params.get('no_heads_ipa', 12),
            "no_qk_points": params.get('no_qk_points', 4),
            "no_v_points": params.get('no_v_points', 12),
            "no_blocks": 6,
            "no_transition_layers": 1,
            "no_resnet_blocks": 2,
            "no_angles": params.get('no_angles', 8),
            "trans_scale_factor": 1.0,
        },
        "ss_module": {"hidden_size": 128, "dropout": 0.0, "num_layers": 1},
        "input_embedder": {"dim": 64, "use_ss": True},
        "pair_embedder": {"dim": 64},
        "recycling": {"num_recycles": 3},
    }

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print("\n" + "="*50)
    print("生成的config:")
    print("="*50)
    print(json.dumps(config["structure_module"], indent=2))
    print("\n已保存到:", output_path)

if __name__ == '__main__':
    analyze_and_fix()