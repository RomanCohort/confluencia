#!/usr/bin/env python3
"""
分析StructureModule的实际参数需求，生成完全正确的config
"""

import os
import re
import json
import inspect
import importlib.util

def get_class_signature(filepath, class_name):
    """获取类的完整签名"""
    try:
        spec = importlib.util.spec_from_file_location("module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            sig = inspect.signature(cls.__init__)
            return list(sig.parameters.keys())
    except Exception as e:
        print(f"Warning: {e}")
        return []

def main():
    base_dir = "trRosettaRNA2/trRNA2"

    print("======================================================================")
    print("  分析StructureModule实际参数需求")
    print("======================================================================")
    print()

    # 获取StructureModule的__init__参数
    structure_params = get_class_signature(f"{base_dir}/structure_module.py", "StructureModule")

    print("StructureModule.__init__ 参数:")
    for p in structure_params:
        print(f"  - {p}")
    print()

    # 获取Folding的__init__参数
    folding_params = get_class_signature(f"{base_dir}/model_3d.py", "Folding")
    print("Folding.__init__ 参数:")
    for p in folding_params:
        print(f"  - {p}")
    print()

    # 基于实际参数生成config
    # 从错误信息看，StructureModule需要这些位置参数：
    # c_s, c_ipa, c_resnet, no_heads_ipa, no_qk_points, no_v_points,
    # no_blocks, no_transition_layers, no_resnet_blocks, no_angles

    config = {
        "model_name": "model_1",
        "num_recycles": 3,
        "nrows": 64,
        "max_recycle": 3,

        # 基础字段
        "dim_pair": 64,
        "use_ss": True,
        "divide": False,
        "init_str": True,
        "ss3D": False,

        # RNAformer block
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
        },

        # StructureModule block - 包含所有必需的位置参数
        "structure_module": {
            "c_s": 64,      # 必需的位置参数
            "c_ipa": 64,    # 必需的位置参数
            "c_resnet": 64, # 必需的位置参数
            "no_heads_ipa": 1,  # 必需的位置参数
            "no_qk_points": 1,  # 必需的位置参数
            "no_v_points": 1,   # 必需的位置参数
            "no_blocks": 6,     # 必需的位置参数
            "no_transition_layers": 2,  # 必需的位置参数
            "no_resnet_blocks": 2,   # 必需的位置参数
            "no_angles": 36,     # 必需的位置参数
        },

        # ss_module
        "ss_module": {
            "hidden_size": 128,
            "dropout": 0.0,
            "num_layers": 1,
        },

        # input_embedder
        "input_embedder": {
            "dim": 64,
            "use_ss": True,
        },

        # pair_embedder
        "pair_embedder": {
            "dim": 64,
        },

        # recycling
        "recycling": {
            "num_recycles": 3,
        },
    }

    # 保存
    output_path = "trRosettaRNA2/weights/params/config/model_1.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"✓ Config已生成: {output_path}")
    print(f"  包含 {len(config)} 个顶层字段")
    print()

    # 验证关键字段
    print("关键字段验证:")
    print(f"  ✓ dim_pair: {config['dim_pair']}")
    print(f"  ✓ RNAformer.n_block: {config['RNAformer']['n_block']}")
    print(f"  ✓ structure_module.c_s: {config['structure_module']['c_s']}")
    print(f"  ✓ structure_module.no_blocks: {config['structure_module']['no_blocks']}")
    print()

    print("下一步:")
    print("  bash fix_pipeline.sh")

if __name__ == '__main__':
    main()