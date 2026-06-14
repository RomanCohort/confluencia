"""联合治疗筛选实验 (Combination Screening Experiment)

系统筛选双药联合方案，使用Bliss独立性模型评估协同效应。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_single_drug(drug: str, dose: float, n_days: int = 120) -> dict:
    """运行单药模拟"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    # 基线30天
    for _ in range(30):
        agent.step()

    # 治疗90天
    agent.administer_drug(drug, dose)
    for _ in range(90):
        agent.step()

    s = agent.state
    baseline_v = s["cli_baseline_volume"]
    effect = (baseline_v - s["tum_volume"]) / max(baseline_v, 1)
    return {
        "drug": drug,
        "dose": dose,
        "final_volume": s["tum_volume"],
        "effect": effect,
        "recist": s["cli_recist_response"],
    }


def run_combination(drug1: str, dose1: float, drug2: str, dose2: float,
                    n_days: int = 120) -> dict:
    """运行联合用药模拟"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    # 基线30天
    for _ in range(30):
        agent.step()

    # 联合治疗90天
    agent.administer_drug(drug1, dose1)
    agent.administer_drug(drug2, dose2)
    for _ in range(90):
        agent.step()

    s = agent.state
    baseline_v = s["cli_baseline_volume"]
    effect = (baseline_v - s["tum_volume"]) / max(baseline_v, 1)
    return {
        "drug1": drug1,
        "drug2": drug2,
        "final_volume": s["tum_volume"],
        "effect": effect,
        "recist": s["cli_recist_response"],
        "toxicity": s["cli_toxicity_grade"],
    }


def run_experiment():
    """运行联合治疗筛选实验"""
    print("联合治疗筛选实验 - TNBC Simulacrum")
    print("=" * 50)

    # 候选药物
    drugs = [
        ("doxorubicin", 60.0),
        ("paclitaxel", 175.0),
        ("carboplatin", 400.0),
        ("atezolizumab", 1200.0),
        ("bevacizumab", 15.0),
    ]

    # 单药基线
    print("\n单药基线:")
    single_results = {}
    for drug, dose in drugs:
        result = run_single_drug(drug, dose)
        single_results[drug] = result
        print(f"  {drug}: Effect={result['effect']:.3f}, RECIST={result['recist']}")

    # 联合筛选
    print("\n联合筛选:")
    combo_results = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            d1, dose1 = drugs[i]
            d2, dose2 = drugs[j]
            result = run_combination(d1, dose1, d2, dose2)
            combo_results.append(result)

            # Bliss独立性
            e1 = single_results[d1]["effect"]
            e2 = single_results[d2]["effect"]
            bliss_expected = e1 + e2 - e1 * e2
            synergy = result["effect"] - bliss_expected

            result["bliss_expected"] = bliss_expected
            result["synergy"] = synergy

            label = "协同" if synergy > 0.05 else "拮抗" if synergy < -0.05 else "相加"
            print(f"  {d1}+{d2}: Effect={result['effect']:.3f}, "
                  f"Bliss={bliss_expected:.3f}, Synergy={synergy:.3f} ({label})")

    # 排序
    print("\n" + "=" * 50)
    print("联合方案排序 (按协同指数):")
    sorted_combos = sorted(combo_results, key=lambda x: x["synergy"], reverse=True)
    for i, r in enumerate(sorted_combos[:5]):
        label = "协同" if r["synergy"] > 0.05 else "拮抗" if r["synergy"] < -0.05 else "相加"
        print(f"  {i+1}. {r['drug1']}+{r['drug2']}: "
              f"Synergy={r['synergy']:.3f} ({label}), "
              f"RECIST={r['recist']}")

    return {"single": single_results, "combinations": combo_results}


if __name__ == "__main__":
    run_experiment()
