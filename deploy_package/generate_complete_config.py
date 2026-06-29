#!/usr/bin/env python3
"""
自动分析trRosettaRNA2源代码，提取所有需要的config字段
生成完整的model_1.json配置文件
"""

import os
import re
import json

def extract_config_fields(source_dir):
    """从trRosettaRNA2源代码中提取所有config字段"""
    fields = {}
    nested_fields = {}

    # 遍历所有Python文件
    for root, dirs, files in os.walk(source_dir):
        # 跳过__pycache__
        if '__pycache__' in root:
            continue

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 提取 config['field'] 模式
                        simple_matches = re.findall(r"config\['(\w+)'\](?!\[)", content)
                        for field in simple_matches:
                            if field not in fields:
                                fields[field] = None  # None表示我们不知道默认值

                        # 提取 config['block']['field'] 模式
                        nested_matches = re.findall(r"config\['(\w+)'\]\['(\w+)'\]", content)
                        for block, field in nested_matches:
                            if block not in nested_fields:
                                nested_fields[block] = set()
                            nested_fields[field] = None
                            nested_fields[block].add(field)
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")

    return fields, nested_fields

def generate_default_config(fields, nested_fields):
    """生成完整的config字典"""
    # 基于已知错误和RNA模型经验设置默认值
    config = {
        # 基础模型参数
        "model_name": "model_1",
        "num_recycles": 3,
        "nrows": 64,
        "dropout": 0.0,
        "d_model": 256,
        "d_ff": 512,
        "num_heads": 8,
        "num_layers": 6,
        "max_len": 500,

        # Embedding维度
        "dim_pair": 64,
        "dim_single": 64,
        "use_ss": True,

        # 新发现的顶层字段
        "c_z": 64,
        "divide": False,
        "init_str": True,
        "max_recycle": 3,
        "ss3D": False,
    }

    # RNAformer block
    if 'RNAformer' in nested_fields:
        config["RNAformer"] = {
            "n_block": 6,
            "d_model": 256,
            "d_ff": 512,
            "num_heads": 8,
            "dropout": 0.0,
            "max_len": 500,
            "msa_tie_row_attn": False,
            "dropout_rate_attn": 0.0,  # 新发现
            "dropout_rate_ff": 0.0,    # 新发现
            "qknorm": True,             # 新发现
            "use_r2n": False,           # 新发现
        }

    # structure_module
    if 'structure_module' in nested_fields or 'structure_module' in fields:
        config["structure_module"] = {
            "hidden_size": 256,
            "dropout": 0.0,
            "num_layers": 2,
            "use_bias": True,
            "trans_scale_factor": 1.0,  # 新发现
        }

    # ss_module
    if 'ss_module' in nested_fields or 'ss_module' in fields:
        config["ss_module"] = {
            "hidden_size": 128,
            "dropout": 0.0,
            "num_layers": 1,
            "use_bias": True
        }

    # input_embedder
    if 'input_embedder' in nested_fields or 'input_embedder' in fields:
        config["input_embedder"] = {
            "dim": 64,
            "use_ss": True
        }

    # pair_embedder
    if 'pair_embedder' in nested_fields or 'pair_embedder' in fields:
        config["pair_embedder"] = {
            "dim": 64
        }

    # recycling
    if 'recycling' in fields:
        config["recycling"] = {
            "num_recycles": 3
        }

    # 添加所有其他字段（使用None或合理默认值）
    for field in fields:
        if field not in config and field not in ['RNAformer', 'structure_module', 'ss_module', 'input_embedder', 'pair_embedder', 'recycling']:
            # 使用合理的默认值
            config[field] = None

    return config

def main():
    source_dir = "trRosettaRNA2/trRNA2"

    print("======================================================================")
    print("  分析trRosettaRNA2源代码，提取config字段")
    print("======================================================================")
    print()

    if not os.path.exists(source_dir):
        print(f"ERROR: {source_dir} 不存在")
        return

    # 提取字段
    fields, nested_fields = extract_config_fields(source_dir)

    print(f"发现的顶层config字段 ({len(fields)}个):")
    for field in sorted(fields.keys()):
        print(f"  - {field}")

    print(f"\n发现的嵌套config字段 ({len(nested_fields)}个block):")
    for block, field_set in sorted(nested_fields.items()):
        if isinstance(field_set, set):
            print(f"  {block}:")
            for field in sorted(field_set):
                print(f"    - {field}")

    # 生成完整config
    config = generate_default_config(fields, nested_fields)

    # 写入文件
    output_path = "trRosettaRNA2/weights/params/config/model_1.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n✓ 完整config已生成: {output_path}")
    print(f"  包含 {len(config)} 个顶层字段")

    print("\n配置文件内容预览:")
    print(json.dumps(config, indent=2)[:500] + "...")

if __name__ == '__main__':
    main()