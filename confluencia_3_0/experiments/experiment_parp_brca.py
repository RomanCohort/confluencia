"""PARP抑制剂 + BRCA突变实验 (PARP Inhibitor + BRCA Experiment)

比较 BRCA突变 vs 野生型 对 PARP抑制剂(olaparib)的敏感性。
验证合成致死效应。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_brca_arm(brca_mutated: bool, n_days: int = 180) -> dict:
    """运行一个BRCA状态的模拟臂"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.brca_mutation = brca_mutated
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    history = []

    # 基线30天
    for _ in range(30):
        agent.step()

    # PARP抑制剂150天
    agent.administer_drug("olaparib", 300.0)
    for i in range(150):
        agent.step()
        if i % 15 == 0:
            s = agent.state
            history.append({
                "day": agent.day,
                "volume": s["tum_volume"],
                "recist": s["cli_recist_response"],
                "change_pct": s["cli_tumor_change_pct"],
                "resistance": s["drg_resistance_level"],
                "brca_status": s["bio_brca_status"],
            })

    s = agent.state
    return {
        "brca_mutated": brca_mutated,
        "final_volume": s["tum_volume"],
        "recist": s["cli_recist_response"],
        "change_pct": s["cli_tumor_change_pct"],
        "resistance": s["drg_resistance_level"],
        "history": history,
    }


def run_experiment():
    """运行PARP+BRCA实验"""
    print("PARP抑制剂 + BRCA突变实验 - TNBC Simulacrum")
    print("=" * 50)

    # BRCA突变臂
    print("\n--- BRCA1突变 + Olaparib ---")
    result_mut = run_brca_arm(brca_mutated=True)
    print(f"  最终体积: {result_mut['final_volume']:.1f} mm3")
    print(f"  RECIST: {result_mut['recist']}")
    print(f"  变化: {result_mut['change_pct']:.1f}%")
    print(f"  耐药: {result_mut['resistance']:.3f}")

    # BRCA野生型臂
    print("\n--- BRCA野生型 + Olaparib ---")
    result_wt = run_brca_arm(brca_mutated=False)
    print(f"  最终体积: {result_wt['final_volume']:.1f} mm3")
    print(f"  RECIST: {result_wt['recist']}")
    print(f"  变化: {result_wt['change_pct']:.1f}%")
    print(f"  耐药: {result_wt['resistance']:.3f}")

    # 合成致死效应
    print("\n" + "=" * 50)
    delta = result_wt['final_volume'] - result_mut['final_volume']
    print(f"合成致死效应: BRCA突变肿瘤比野生型小 {delta:.1f} mm3")
    if result_mut['change_pct'] < result_wt['change_pct']:
        print("  -> PARP抑制剂对BRCA突变肿瘤更有效 (合成致死验证)")
    else:
        print("  -> 未观察到显著合成致死效应")

    return {"brca_mutated": result_mut, "brca_wildtype": result_wt}


if __name__ == "__main__":
    run_experiment()
