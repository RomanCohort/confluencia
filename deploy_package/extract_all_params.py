#!/usr/bin/env python3
"""
从checkpoint权重提取StructureModule的所有必需参数
基于实际的__init__签名生成正确的config
"""
import torch
import json
import os

def extract_structure_params(checkpoint_path):
    """从checkpoint提取所有StructureModule参数"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    params = {}

    for k, v in ckpt.items():
        # no_heads_ipa
        if 'structure_module.ipa.head_weights' in k:
            params['no_heads_ipa'] = int(v.shape[0])

        # c_ipa, c_z (从linear_q.weight)
        elif 'structure_module.ipa.linear_q.weight' in k:
            total, c_z = v.shape
            if 'no_heads_ipa' in params:
                params['c_ipa'] = int(total // params['no_heads_ipa'])
                params['c_z'] = int(c_z)

        # no_qk_points
        elif 'structure_module.ipa.linear_q_points.weight' in k:
            total, _ = v.shape
            if 'no_heads_ipa' in params:
                params['no_qk_points'] = int(total // (params['no_heads_ipa'] * 3))

        # no_v_points
        elif 'structure_module.ipa.linear_kv_points.weight' in k:
            total, _ = v.shape
            if 'no_heads_ipa' in params:
                params['no_v_points'] = int(total // (params['no_heads_ipa'] * 3))

        # c_resnet, c_s
        elif 'structure_module.angle_resnet.linear_in.weight' in k:
            params['c_resnet'] = int(v.shape[0])
            params['c_s'] = int(v.shape[1])

        # no_angles
        elif 'structure_module.angle_resnet.linear_out.weight' in k:
            params['no_angles'] = int(v.shape[0])

        # no_transition_layers
        elif 'structure_module.transition.layers' in k and '.linear_1.weight' in k:
            nums = [int(x) for x in k.split('.') if x.isdigit()]
            if nums:
                max_layer = max(nums)
                if 'no_transition_layers' not in params or max_layer + 1 > params['no_transition_layers']:
                    params['no_transition_layers'] = max_layer + 1

        # no_resnet_blocks (从angle_resnet.layers)
        elif 'structure_module.angle_resnet.layers' in k and '.linear_1.weight' in k:
            nums = [int(x) for x in k.split('.') if x.isdigit()]
            if nums:
                max_layer = max(nums)
                if 'no_resnet_blocks' not in params or max_layer + 1 > params['no_resnet_blocks']:
                    params['no_resnet_blocks'] = max_layer + 1

        # no_blocks (从IPA blocks)
        elif 'structure_module.ipa_blocks' in k:
            nums = [int(x) for x in k.split('.') if x.isdigit()]
            if nums:
                max_block = max(nums)
                if 'no_blocks' not in params or max_block + 1 > params['no_blocks']:
                    params['no_blocks'] = max_block + 1

    # 设置默认值（如果未从权重中提取到）
    params.setdefault('no_blocks', 6)
    params.setdefault('no_transition_layers', 1)
    params.setdefault('no_resnet_blocks', 2)
    params.setdefault('trans_scale_factor', 1.0)

    return params

def main():
    checkpoint_path = "trRosettaRNA2/weights/params/models/model_1.pth.tar"
    output_path = "trRosettaRNA2/weights/params/config/model_1.json"

    print("="*60)
    print("  从checkpoint提取StructureModule参数")
    print("="*60)

    params = extract_structure_params(checkpoint_path)

    print("\n提取的参数:")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")

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
            "c_s": params['c_s'],
            "c_z": params['c_z'],
            "c_ipa": params['c_ipa'],
            "c_resnet": params['c_resnet'],
            "no_heads_ipa": params['no_heads_ipa'],
            "no_qk_points": params['no_qk_points'],
            "no_v_points": params['no_v_points'],
            "no_blocks": params['no_blocks'],
            "no_transition_layers": params['no_transition_layers'],
            "no_resnet_blocks": params['no_resnet_blocks'],
            "no_angles": params['no_angles'],
            "trans_scale_factor": params['trans_scale_factor'],
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

    print(f"\n✓ Config已生成: {output_path}")
    print("\nstructure_module配置:")
    print(json.dumps(config['structure_module'], indent=2))

if __name__ == '__main__':
    main()