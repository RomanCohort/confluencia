"""联合化疗+免疫治疗实验 (Combination Chemo+Immunotherapy Experiment)

比较:
  1. 单药化疗
  2. 单药免疫治疗
  3. 化疗+免疫联合 (协同效应)
  4. 序贯治疗 (先化疗后免疫)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_arm(arm_name: str, n_days: int = 180) -> dict:
    """运行一个治疗臂"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "IM"  # IM亚型适合免疫治疗
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    history = []

    # 基线30天
    for _ in range(30):
        agent.step()

    # 根据臂名给药
    if arm_name == "chemo_only":
        agent.administer_drug("paclitaxel", 175.0)
    elif arm_name == "immuno_only":
        agent.administer_drug("atezolizumab", 1200.0)
    elif arm_name == "concurrent":
        agent.administer_drug("paclitaxel", 175.0)
        agent.administer_drug("atezolizumab", 1200.0)
    elif arm_name == "sequential":
        # 先化疗60天
        agent.administer_drug("paclitaxel", 175.0)
        for i in range(60):
            agent.step()
            if i % 15 == 0:
                s = agent.state
                history.append({
                    "day": agent.day,
                    "volume": s["tum_volume"],
                    "recist": s["cli_recist_response"],
                    "exhaustion": s["imm_t_cell_exhaustion"],
                    "cd8": s["imm_cd8_count"],
                })
        # 后免疫治疗90天
        agent.administer_drug("atezolizumab", 1200.0)

    # 继续治疗
    remaining = n_days - 30 - (60 if arm_name == "sequential" else 0)
    for i in range(remaining):
        agent.step()
        if i % 15 == 0:
            s = agent.state
            history.append({
                "day": agent.day,
                "volume": s["tum_volume"],
                "recist": s["cli_recist_response"],
                "exhaustion": s["imm_t_cell_exhaustion"],
                "cd8": s["imm_cd8_count"],
            })

    s = agent.state
    return {
        "arm": arm_name,
        "final_volume": s["tum_volume"],
        "recist": s["cli_recist_response"],
        "change_pct": s["cli_tumor_change_pct"],
        "exhaustion": s["imm_t_cell_exhaustion"],
        "cd8": s["imm_cd8_count"],
        "history": history,
    }


def run_experiment():
    """运行联合治疗实验"""
    print("联合化疗+免疫治疗实验 - TNBC Simulacrum")
    print("=" * 50)

    arms = ["chemo_only", "immuno_only", "concurrent", "sequential"]
    arm_names = {
        "chemo_only": "单药化疗(Paclitaxel)",
        "immuno_only": "单药免疫(Atezolizumab)",
        "concurrent": "联合(化疗+免疫)",
        "sequential": "序贯(先化疗后免疫)",
    }

    results = {}
    for arm in arms:
        print(f"\n--- {arm_names[arm]} ---")
        result = run_arm(arm)
        results[arm] = result
        print(f"  最终体积: {result['final_volume']:.1f} mm3")
        print(f"  RECIST: {result['recist']}")
        print(f"  变化: {result['change_pct']:.1f}%")
        print(f"  CD8+: {result['cd8']:.0f}")

    # 协同效应分析
    print("\n" + "=" * 50)
    chemo_v = results["chemo_only"]["final_volume"]
    immuno_v = results["immuno_only"]["final_volume"]
    concur_v = results["concurrent"]["final_volume"]
    seq_v = results["sequential"]["final_volume"]

    # Bliss独立性预期
    baseline_v = 50.0  # 初始体积
    chemo_effect = (baseline_v - chemo_v) / baseline_v
    immuno_effect = (baseline_v - immuno_v) / baseline_v
    bliss_expected = chemo_effect + immuno_effect - chemo_effect * immuno_effect
    concur_effect = (baseline_v - concur_v) / baseline_v
    synergy = concur_effect - bliss_expected

    print(f"化疗单药效应: {chemo_effect:.3f}")
    print(f"免疫单药效应: {immuno_effect:.3f}")
    print(f"Bliss预期联合效应: {bliss_expected:.3f}")
    print(f"实际联合效应: {concur_effect:.3f}")
    print(f"协同指数: {synergy:.3f} ({'协同' if synergy > 0.05 else '相加' if synergy > -0.05 else '拮抗'})")

    return results


if __name__ == "__main__":
    run_experiment()
