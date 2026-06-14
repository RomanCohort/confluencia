"""亚型比较实验 (Subtype Comparison Experiment)

4种TNBC亚型并行模拟，比较治疗敏感性。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


SUBTYPES = ["BLIS", "IM", "M", "LAR"]


def run_subtype(subtype: str, n_days: int = 180) -> dict:
    """运行一个亚型的模拟"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = subtype
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    # 基线30天
    for _ in range(30):
        agent.step()

    # 化疗150天
    agent.administer_drug("doxorubicin", 60.0)
    for _ in range(150):
        agent.step()

    s = agent.state
    return {
        "subtype": subtype,
        "final_volume": s["tum_volume"],
        "recist": s["cli_recist_response"],
        "change_pct": s["cli_tumor_change_pct"],
        "toxicity": s["cli_toxicity_grade"],
        "ied_phase": s["ied_phase"],
        "resistance": s["drg_resistance_level"],
    }


def run_experiment():
    """运行亚型比较实验"""
    print("亚型比较实验 - TNBC Simulacrum")
    print("=" * 50)

    results = {}
    for subtype in SUBTYPES:
        print(f"\n--- {subtype} 亚型 ---")
        result = run_subtype(subtype)
        results[subtype] = result
        print(f"  最终体积: {result['final_volume']:.1f} mm³")
        print(f"  RECIST: {result['recist']}")
        print(f"  变化: {result['change_pct']:.1f}%")
        print(f"  免疫编辑: {result['ied_phase']}")
        print(f"  耐药: {result['resistance']:.3f}")

    # 比较总结
    print("\n" + "=" * 50)
    print("亚型比较总结")
    print("-" * 50)
    for subtype, r in results.items():
        print(f"  {subtype}: V={r['final_volume']:.1f}, "
              f"RECIST={r['recist']}, Change={r['change_pct']:.1f}%")

    return results


if __name__ == "__main__":
    run_experiment()