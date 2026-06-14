"""AR拮抗剂 + LAR亚型实验 (AR Antagonist + LAR Subtype Experiment)

LAR亚型(雄激素受体阳性)对AR拮抗剂(enzalutamide)的敏感性。
与其他亚型对比验证亚型特异性治疗。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_ar_arm(subtype: str, n_days: int = 180) -> dict:
    """运行一个亚型的AR拮抗剂模拟"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = subtype
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    history = []

    # 基线30天
    for _ in range(30):
        agent.step()

    # AR拮抗剂150天
    agent.administer_drug("enzalutamide", 160.0)
    for i in range(150):
        agent.step()
        if i % 15 == 0:
            s = agent.state
            history.append({
                "day": agent.day,
                "volume": s["tum_volume"],
                "recist": s["cli_recist_response"],
                "ar_expression": s["sub_ar_expression"],
            })

    s = agent.state
    return {
        "subtype": subtype,
        "final_volume": s["tum_volume"],
        "recist": s["cli_recist_response"],
        "change_pct": s["cli_tumor_change_pct"],
        "ar_expression": s["sub_ar_expression"],
        "history": history,
    }


def run_experiment():
    """运行AR+LAR实验"""
    print("AR拮抗剂 + LAR亚型实验 - TNBC Simulacrum")
    print("=" * 50)

    results = {}
    for subtype in ["LAR", "BLIS", "IM", "M"]:
        print(f"\n--- {subtype} + Enzalutamide ---")
        result = run_ar_arm(subtype)
        results[subtype] = result
        print(f"  最终体积: {result['final_volume']:.1f} mm3")
        print(f"  RECIST: {result['recist']}")
        print(f"  AR表达: {result['ar_expression']:.2f}")

    # LAR特异性验证
    print("\n" + "=" * 50)
    lar_change = results["LAR"]["change_pct"]
    other_avg = sum(results[s]["change_pct"] for s in ["BLIS", "IM", "M"]) / 3
    print(f"LAR变化: {lar_change:.1f}%, 其他亚型平均: {other_avg:.1f}%")
    if lar_change < other_avg:
        print("  -> LAR亚型对AR拮抗剂更敏感 (亚型特异性验证)")
    else:
        print("  -> 未观察到LAR亚型特异性优势")

    return results


if __name__ == "__main__":
    run_experiment()
