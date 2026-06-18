"""Confluencia 3.0 CLI 入口。

用法:
    python -m confluencia_3_0 --config config.yaml
    python -m confluencia_3_0 --steps 365 --subtype BLIS
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="confluencia-3.0",
        description="Confluencia 3.0 — circRNA + TNBC Simulacrum 统一计算平台",
    )
    parser.add_argument("--steps", type=int, default=365, help="模拟步数（天）")
    parser.add_argument("--subtype", type=str, default="BLIS",
                        choices=["BLIS", "IM", "M", "LAR"], help="分子亚型")
    parser.add_argument("--circrna-backend", type=str, default="heuristic",
                        choices=["heuristic", "vienna", "esm2"], help="circRNA 免疫原性后端")
    parser.add_argument("--structure-mode", type=str, default="heuristic",
                        choices=["heuristic", "simple", "diffusion", "physics_b", "physics_ba"],
                        help="circRNA 结构预测模式: "
                             "heuristic=Backend降级, simple=MDS快速, "
                             "diffusion=AF3扩散, physics_b=几何约束, "
                             "physics_ba=几何+OpenMM")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-circrna", action="store_true", help="禁用 circRNA 子系统")
    args = parser.parse_args()

    from confluencia_3_0.core.config import Confluencia3Config, CircRNAConfig
    config = Confluencia3Config()
    config.molecular_subtype = args.subtype
    config.experiment.max_steps = args.steps
    config.experiment.seed = args.seed
    config.circrna.immunogenicity_backend = args.circrna_backend
    config.circrna.enabled = not args.no_circrna
    config.circrna.structure_mode = args.structure_mode

    print(f"Confluencia 3.0 启动")
    print(f"  亚型: {args.subtype}")
    print(f"  步数: {args.steps}")
    print(f"  circRNA 后端: {args.circrna_backend}")
    print(f"  circRNA 结构模式: {args.structure_mode}")
    print(f"  circRNA 启用: {config.circrna.enabled}")
    if args.structure_mode != "heuristic":
        print(f"  TorusFold 已启用 (mode={args.structure_mode})")

    # TODO: 初始化 Confluencia3Agent 并运行模拟
    print("模拟引擎初始化中...")


if __name__ == "__main__":
    main()
