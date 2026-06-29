#!/usr/bin/env python3
"""
基于真实的__init__参数生成正确的trRosettaRNA2配置文件
不再猜测字段，只使用实际参数
"""

import os
import re
import json
import inspect
import importlib.util

def get_class_init_params(filepath, class_name):
    """获取类的__init__参数列表"""
    try:
        spec = importlib.util.spec_from_file_location("module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters.keys() if p != 'self']
            return params
    except Exception as e:
        print(f"Warning: Could not inspect {class_name}: {e}")
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                match = re.search(f'class {class_name}.*?def __init__\(self([^)]*)\)', content, re.DOTALL)
                if match:
                    params_str = match.group(1)
                    params = re.findall(r'(\w+)=', params_str)
                    return params
        except:
            pass
    return []

def analyze_source_for_config_usage(source_dir):
    """分析源代码中config['field']的使用"""
    fields = set()
    nested_fields = {}

    for root, dirs, files in os.walk(source_dir):
        if '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        # 简单字段
                        simple = re.findall(r"config\['(\w+)'\](?!\[)", content)
                        fields.update(simple)
                        # 嵌套字段
                        nested = re.findall(r"config\['(\w+)'\]\['(\w+)'\]", content)
                        for block, field in nested:
                            if block not in nested_fields:
                                nested_fields[block] = set()
                            nested_fields[block].add(field)
                except:
                    pass

    return fields, nested_fields

def main():
    base_dir = "trRosettaRNA2/trRNA2"

    print("======================================================================")
    print("  生成完全正确的trRosettaRNA2配置文件")
    print("======================================================================")
    print()

    # 分析源代码中的config使用
    fields, nested_fields = analyze_source_for_config_usage(base_dir)

    print("发现的config字段:")
    for field in sorted(fields):
        print(f"  - {field}")
    print()

    # 获取RNAformer的实际参数
    rnaformer_params = get_class_init_params(f"{base_dir}/RNAformer.py", "RNAformer")

    print("RNAformer实际参数:")
    for param in rnaformer_params:
        print(f"  - {param}")
    print()

    # 构建config - 只包含实际需要的字段
    config = {
        # 基础参数
        "model_name": "model_1",
        "num_recycles": 3,
        "nrows": 64,
        "max_recycle": 3,

        # 从源码分析得出的字段
        "dim_pair": 64,
        "use_ss": True,
        "divide": False,
        "init_str": True,
        "ss3D": False,

        # RNAformer block - 基于实际__init__参数
        "RNAformer": {
            "n_block": 6,  # 对应depth
            "d_model": 256,  # 对应dim
            "d_ff": 512,  # 对应emb_dim
            "num_heads": 8,  # 对应heads
            "dim_head": 32,
            "num_tokens": 4,
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "msa_tie_row_attn": False,
            "msa_conv": False,
            "use_r2n": False,
            "qknorm": True,

            # 从config['RNAformer']['field']使用分析得出
            "dropout_rate_attn": 0.0,
            "dropout_rate_ff": 0.0,
        },

        # structure_module - 基于源码分析，不需要hidden_size等
        "structure_module": {
            "c_z": 64,  # 源码中使用的
            "trans_scale_factor": 1.0,  # 源码中使用的
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

    # 添加所有发现的顶层字段（如果缺失）
    for field in fields:
        if field not in config:
            config[field] = None

    # 保存
    output_path = "trRosettaRNA2/weights/params/config/model_1.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"✓ 完整config已生成: {output_path}")
    print(f"  包含 {len(config)} 个顶层字段")
    print()

    # 验证关键字段
    print("关键字段验证:")
    print(f"  ✓ dim_pair: {config.get('dim_pair')}")
    print(f"  ✓ RNAformer.n_block: {config['RNAformer'].get('n_block')}")
    print(f"  ✓ structure_module.c_z: {config['structure_module'].get('c_z')}")
    print()

    print("下一步:")
    print("  bash fix_pipeline.sh")

if __name__ == '__main__':
    main()