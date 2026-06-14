#!/usr/bin/env python3
"""
debug_rinalmo.py — 测试 RiNALMo 是否能正常输出

在 AutoDL 上运行:
    cd /root/autodl-tmp/confluencia/confluencia-circrna-encoder
    python scripts/debug_rinalmo.py
"""

import sys
import torch

# RiNALMo 路径
sys.path.insert(0, "/root/autodl-tmp/RiNALMo")

print("=== RiNALMo 调试 ===\n")

# 1. 测试导入
try:
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo
    from rinalmo.data.alphabet import Alphabet
    print("1. 导入成功 ✓")
except Exception as e:
    print(f"1. 导入失败: {e}")
    sys.exit(1)

# 2. 加载模型
try:
    config = model_config('giga')
    model = RiNALMo(config)
    checkpoint = torch.load(
        '/root/autodl-tmp/RiNALMo/weights/rinalmo_giga_pretrained.pt',
        map_location='cpu'
    )
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"2. 模型加载 ✓ (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"   Missing keys (前5个): {missing[:5]}")
    if unexpected:
        print(f"   Unexpected keys (前5个): {unexpected[:5]}")
except Exception as e:
    print(f"2. 模型加载失败: {e}")
    sys.exit(1)

# 3. 测试 Alphabet
try:
    alphabet = Alphabet(**config['alphabet'])
    seqs = ['ACGUACGU', 'GGCCUUAACC']
    tokens = alphabet.batch_tokenize(seqs)
    print(f"3. Alphabet ✓")
    print(f"   序列: {seqs}")
    print(f"   Tokens: {tokens}")
    print(f"   tkn_to_idx keys: {list(alphabet.tkn_to_idx.keys())[:10]}")
except Exception as e:
    print(f"3. Alphabet 失败: {e}")
    sys.exit(1)

# 4. 前向传播
try:
    model.eval()
    tokens_t = torch.tensor(tokens, dtype=torch.int64)
    with torch.no_grad():
        out = model(tokens_t)
    print(f"4. 前向传播 ✓")
    print(f"   Output keys: {list(out.keys())}")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"   {k}: shape={v.shape}, has_nan={torch.isnan(v).any()}, mean={v.mean():.4f}")
except Exception as e:
    print(f"4. 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. GPU 测试
if torch.cuda.is_available():
    try:
        model = model.cuda()
        tokens_t = tokens_t.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(tokens_t)
        repr = out.get('representation', out.get('embeddings'))
        has_nan = torch.isnan(repr).any()
        print(f"5. GPU 前向传播 ✓")
        print(f"   has_nan={has_nan}, mean={repr.mean():.4f}")
    except Exception as e:
        print(f"5. GPU 失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("5. GPU 不可用，跳过")

# 6. Pooling 测试
try:
    repr = out.get('representation', out.get('embeddings'))
    # Mean pool
    pooled = repr[:, 1:-1, :].mean(dim=1)
    print(f"6. Pooling ✓: shape={pooled.shape}, has_nan={torch.isnan(pooled).any()}, mean={pooled.mean():.4f}")
except Exception as e:
    print(f"6. Pooling 失败: {e}")

print("\n=== 调试完成 ===")
