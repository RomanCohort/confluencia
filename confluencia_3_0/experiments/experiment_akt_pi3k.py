"""AKT抑制剂 + PIK3CA突变实验 (AKT Inhibitor + PIK3CA Experiment)

比较 PIK3CA突变 vs 野生型 对 AKT抑制剂(ipatasertib)的敏感性。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_akt_arm(pi3k_mutated: bool, n_days: int = 180) -> dict:
    """运行一个PIK3CA状态的模拟臂"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)
    if pi3k_mutated:
        agent._internal_state["bio_pi3k_mutation"] = 1

    history = []

    # 基线30天
    for _ in range(30):
        agent.step()

    # AKT抑制剂150天
    agent.administer_drug("ipatasertib", 400.0)
    for i in range(150):
        agent.step()
        if i % 15 == 0:
            s = agent.state
            history.append({
                "day": agent.day,
                "volume": s["tum_volume"],
                "recist": s["cli_recist_response"],
                "change_pct": s["cli_tumor_change_pct"],
                "pi3k_mutation": s["bio_pi3k_mutation"],
            })

    s = agent.state
    return {
        "pi3k_mutated": pi3k_mutated,
        "final_volume": s["tum_volume"],
        "recist": s["cli_recist_response"],
        "change_pct": s["cli_tumor_change_pct"],
        "history": history,
    }


def run_experiment():
    """运行AKT+PIK3CA实验"""
    print("AKT抑制剂 + PIK3CA突变实验 - TNBC Simulacrum")
    print("=" * 50)

    # PIK3CA突变臂
    print("\n--- PIK3CA突变 + Ipatasertib ---")
    result_mut = run_akt_arm(pi3k_mutated=True)
    print(f"  最终体积: {result_mut['final_volume']:.1f} mm3")
    print(f"  RECIST: {result_mut['recist']}")
    print(f"  变化: {result_mut['change_pct']:.1f}%")

    # PIK3CA野生型臂
    print("\n--- PIK3CA野生型 + Ipatasertib ---")
    result_wt = run_akt_arm(pi3k_mutated=False)
    print(f"  最终体积: {result_wt['final_volume']:.1f} mm3")
    print(f"  RECIST: {result_wt['recist']}")
    print(f"  变化: {result_wt['change_pct']:.1f}%")

    # 比较
    print("\n" + "=" * 50)
    delta = result_wt['final_volume'] - result_mut['final_volume']
    print(f"PIK3CA突变效应: 突变肿瘤比野生型小 {delta:.1f} mm3")

    return {"pi3k_mutated": result_mut, "pi3k_wildtype": result_wt}


if __name__ == "__main__":
    run_experiment()
