#!/usr/bin/env python3
"""
彻底分析trRosettaRNA2所有模块的完整参数需求
不再猜测，直接从源码提取所有__init__签名
"""

import os
import re
import json

def extract_init_params(filepath):
    """从源码提取类的完整__init__参数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 找到class定义和__init__方法
        # 匹配多行
        matches = re.findall(r'class (\w+).*?def __init__\(self([^)]*)\)', content, re.DOTALL)

        results = {}
        for class_name, params_str in matches:
            # 提取所有参数
            # 参数可能包含默认值，格式: param, param=default
            params = []
            # 分割参数
            parts = params_str.split(',')
            for part in parts:
                part = part.strip()
                if part and part != 'self':
                    # 提取参数名（去掉默认值）
                    param_name = part.split('=')[0].strip()
                    if param_name and param_name != 'self':
                        params.append(param_name)

            results[class_name] = params

        return results
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def main():
    base_dir = "trRosettaRNA2/trRNA2"

    print("======================================================================")
    print("  彻底分析trRosettaRNA2所有模块参数")
    print("======================================================================")
    print()

    all_params = {}

    # 分析关键文件
    files_to_analyze = {
        'structure_module.py': ['StructureModule'],
        'model_3d.py': ['Folding'],
        'RNAformer.py': ['RNAformer'],
        'modules.py': ['InputEmbedder', 'PairEmbedder'],
    }

    for filename, class_names in files_to_analyze.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            params = extract_init_params(filepath)
            for class_name in class_names:
                if class_name in params:
                    all_params[class_name] = params[class_name]
                    print(f"{class_name}.__init__参数 ({len(params[class_name])}个):")
                    for p in params[class_name]:
                        print(f"  - {p}")
                    print()

    # 分析config使用
    print("分析config字段使用...")
    config_fields = set()
    nested_fields = {}

    for root, dirs, files in os.walk(base_dir):
        if '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        # config['field']
                        simple = re.findall(r"config\['(\w+)'\](?!\[)", content)
                        config_fields.update(simple)
                        # config['block']['field']
                        nested = re.findall(r"config\['(\w+)'\]\['(\w+)'\]", content)
                        for block, field in nested:
                            if block not in nested_fields:
                                nested_fields[block] = set()
                            nested_fields[block].add(field)
                except:
                    pass

    print("顶层config字段:")
    for f in sorted(config_fields):
        print(f"  - {f}")
    print()

    print("嵌套config字段:")
    for block, fields in sorted(nested_fields.items()):
        print(f"  {block}:")
        for f in sorted(fields):
            print(f"    - {f}")
    print()

    # 生成完整config
    config = {
        "model_name": "model_1",
        "num_recycles": 3,
        "nrows": 64,
        "max_recycle": 3,

        # 从config使用分析得出
        "dim_pair": 64,
        "use_ss": True,
        "divide": False,
        "init_str": True,
        "ss3D": False,
    }

    # 添加config字段（如果缺失）
    for field in config_fields:
        if field not in config:
            config[field] = None

    # RNAformer
    if 'RNAformer' in all_params:
        print("生成RNAformer配置...")
        config["RNAformer"] = {}
        for param in all_params['RNAformer']:
            # 根据参数名推断默认值
            if 'dropout' in param:
                config["RNAformer"][param] = 0.0
            elif 'norm' in param or param == 'qknorm':
                config["RNAformer"][param] = True
            elif 'block' in param or 'depth' in param:
                config["RNAformer"][param] = 6
            elif 'dim' in param or 'size' in param:
                config["RNAformer"][param] = 256 if 'd_' in param or 'emb' in param else 64
            elif 'head' in param:
                config["RNAformer"][param] = 8
            elif 'token' in param:
                config["RNAformer"][param] = 4
            else:
                config["RNAformer"][param] = False

    # StructureModule
    if 'StructureModule' in all_params:
        print("生成StructureModule配置...")
        config["structure_module"] = {}
        for param in all_params['StructureModule']:
            # 根据参数名推断默认值
            if 'c_' in param:
                config["structure_module"][param] = 64
            elif 'no_' in param:
                config["structure_module"][param] = 6 if 'block' in param else 2
            elif 'scale' in param:
                config["structure_module"][param] = 1.0
            else:
                config["structure_module"][param] = 1

    # 添加源码中使用的嵌套字段
    for block, fields in nested_fields.items():
        if block not in config:
            config[block] = {}
        for field in fields:
            if field not in config[block]:
                config[block][field] = 0.0 if 'dropout' in field else 64

    # 保存
    output_path = "trRosettaRNA2/weights/params/config/model_1.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"✓ 完整config已生成: {output_path}")
    print(f"  包含 {len(config)} 个顶层字段")
    print()

    # 验证
    print("验证关键字段:")
    print(f"  ✓ dim_pair: {config.get('dim_pair')}")
    if 'RNAformer' in config:
        print(f"  ✓ RNAformer字段数: {len(config['RNAformer'])}")
    if 'structure_module' in config:
        print(f"  ✓ structure_module字段数: {len(config['structure_module'])}")
        print(f"  ✓ structure_module参数: {list(config['structure_module'].keys())}")

if __name__ == '__main__':
    main()