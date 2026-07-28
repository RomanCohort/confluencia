"""
demo_structure_viewer.py — 端到端验证：circRNA 序列 → 3D 结构 → 可交互 HTML。

数据源逻辑在 visualization.structure_sources，studio (app.py) 和本脚本共用。
合成/真实两条路径，真实失败自动 fallback 合成并透明标注。

用法：
    cd confluencia-circrna-encoder
    # 合成（无需权重）
    python scripts/demo_structure_viewer.py --fallback
    # 真实推理（需 v2 权重；v1 残留会自动 fallback）
    python scripts/demo_structure_viewer.py --weights path/to/v2.pt
    python scripts/demo_structure_viewer.py --sequence GUCCCCCCUCCAAUUGG

输出：
    output/circrna_viewer_demo.html  （双击浏览器打开）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本能从项目根目录跑
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.structure_sources import (
    get_structure_for_sequence,
    DEFAULT_WEIGHTS,
)
from visualization.structure_export import export_circrna_structure
from visualization.html_renderer import render_from_export

# 默认演示序列（短，CPU 推理快）
DEFAULT_SEQUENCE = "GUCCCCCCUCCAAUUGGAACGUU"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TorusFold circRNA 3D 可交互 viewer 生成器"
    )
    p.add_argument(
        "--sequence", default=DEFAULT_SEQUENCE,
        help=f"circRNA 序列 (ACGU)，默认 {DEFAULT_SEQUENCE}",
    )
    p.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help=f"TorusFold 权重路径，默认 {DEFAULT_WEIGHTS}",
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="推理设备，默认 cpu",
    )
    p.add_argument(
        "--fallback", action="store_true",
        help="强制用合成数据（不加载 TorusFold）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model = "synthetic" if args.fallback else "torusfold"
    weights = Path(args.weights) if model == "torusfold" else None

    if model == "torusfold":
        print(f"[1/3] 真实 TorusFold 推理: seq={args.sequence[:20]}... (len={len(args.sequence)})")
        print(f"      权重: {weights}")
    else:
        print(f"[1/3] 合成数据: L=48, stem+loop 折叠结构")

    result = get_structure_for_sequence(
        args.sequence, model=model, weights=weights, device=args.device,
    )

    coords = result["coords"]
    seq = result["sequence"]
    immune = result["immune_fingerprints"]
    conf = result["confidence"]
    source = result["source"]

    # 透明标注：合成/合成fallback 的 HTML title 加 (SYNTHETIC) 标记
    if source == "torusfold":
        title_suffix = ""
        print(f"      推理完成 [OK]  source=torusfold")
    elif source == "synthetic-fallback":
        title_suffix = " (SYNTHETIC-FALLBACK — 权重不可用，回退合成)"
        print(f"[FALLBACK] 真实推理失败: {result['fallback_reason']}")
        print(f"           回退合成数据。修好权重后重试。")
    else:
        title_suffix = " (SYNTHETIC — 非 TorusFold 真实预测)"
        print(f"      source=synthetic")

    print(f"[2/3] export: pdb + 指纹 JSON")
    export = export_circrna_structure(
        coords=coords, sequence=seq, immune_fingerprints=immune,
        confidence=conf, circular=True,
    )
    print(f"      pdb={len(export['pdb'])} chars, "
          f"fp_json={len(export['fingerprint_json'])} chars")

    html = render_from_export(
        export, title=f"TorusFold circRNA 3D Viewer{title_suffix}"
    )
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "circrna_viewer_demo.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[3/3] HTML 写出: {out_path}  ({len(html)} bytes)")
    print()
    if source == "torusfold":
        print("双击打开即可看到 TorusFold 真实预测的可交互 3D circRNA 结构。")
    else:
        print("[!] 当前是合成数据，不是 TorusFold 真实预测。")
        print("    双击打开能看到可交互 3D circRNA 折叠（stem+loop 合成结构），")
        print("    坐标是合成的，但 stem/loop 免疫指纹分布是结构性的，切 coloring 能看出差异。")
    print("    浏览器里切 Coloring Scheme 可换 confidence / PKR / m6A / TLR7 等上色。")


if __name__ == "__main__":
    main()
