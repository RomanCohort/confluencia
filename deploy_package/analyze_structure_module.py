#!/usr/bin/env python3
"""
分析trRosettaRNA2源代码，提取所有模块的正确参数
通过检查__init__方法签名来确定正确的config字段
"""

import os
import re
import json
import inspect
import importlib.util

def analyze_class_init(filepath, class_name):
    """分析类的__init__方法参数"""
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the class
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            # Get __init__ signature
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.keys())
            # Remove 'self'
            if 'self' in params:
                params.remove('self')
            return params
    except Exception as e:
        # Fallback: parse source code
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                # Find __init__ method
                match = re.search(f'class {class_name}.*?def __init__\(self([^)]*)\)', content, re.DOTALL)
                if match:
                    params_str = match.group(1)
                    # Extract parameter names
                    params = re.findall(r'(\w+)=', params_str)
                    return params
        except:
            pass
    return []

def main():
    base_dir = "trRosettaRNA2/trRNA2"

    print("======================================================================")
    print("  分析trRosettaRNA2模块参数需求")
    print("======================================================================")
    print()

    # 分析关键类
    classes_to_analyze = {
        'StructureModule': f'{base_dir}/structure_module.py',
        'Folding': f'{base_dir}/model_3d.py',
        'RNAformer': f'{base_dir}/RNAformer.py',
        'InputEmbedder': f'{base_dir}/modules.py',
    }

    all_params = {}

    for class_name, filepath in classes_to_analyze.items():
        if os.path.exists(filepath):
            params = analyze_class_init(filepath, class_name)
            all_params[class_name] = params
            print(f"{class_name}.__init__参数:")
            for p in params:
                print(f"  - {p}")
            print()

    # 生成正确config
    config = {
        "model_name": "model_1",
        "num_recycles": 3,
        "nrows": 64,

        # 根据StructureModule参数生成
        "structure_module": {}
    }

    # 为每个参数生成默认值
    for class_name, params in all_params.items():
        if class_name == 'StructureModule':
            for param in params:
                if param not in ['self', 'config']:
                    # 根据参数名推断默认值
                    if 'dim' in param or 'size' in param:
                        config["structure_module"][param] = 64
                    elif 'dropout' in param:
                        config["structure_module"][param] = 0.0
                    elif 'scale' in param:
                        config["structure_module"][param] = 1.0
                    else:
                        config["structure_module"][param] = None

    # 保存
    output_path = "trRosettaRNA2/weights/params/config/model_1.json"
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"✓ Config已生成: {output_path}")
    print(json.dumps(config, indent=2))

if __name__ == '__main__':
    main()