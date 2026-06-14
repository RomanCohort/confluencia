"""生物标志物分层治疗实验 (Biomarker-Stratified Treatment Experiment)

根据PD-L1 CPS、BRCA状态、TIL密度分层选择最优治疗方案。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_stratified_arm(subtype: str, brca_mut: bool, drug: str, dose: float,
                       n_days: int = 180) -> dict:
    """运行一个分层治疗臂"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = subtype
    config.brca_mutation = brca_mut
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    # 基线30天
    for _ in range(30):
        agent.step()

    # 治疗150天
    agent.administer_drug(drug, dose)
    for _ in range(150):
        agent.step()

    s = agent.state
    return {
        "subtype": subtype,
        "brca_mutated": brca_mut,
        "drug": drug,
        "final_volume": s["tum_volume"],
        "recist": s["cli_recist_response"],
        "change_pct": s["cli_tumor_change_pct"],
        "pd_l1_cps": s["bio_pd_l1_cps"],
        "til_density": s["bio_til_density"],
        "toxicity": s["cli_toxicity_grade"],
    }


def run_experiment():
    """运行生物标志物分层治疗实验"""
    print("生物标志物分层治疗实验 - TNBC Simulacrum")
    print("=" * 50)

    # 分层策略
    strategies = [
        # PD-L1高 + IM亚型 -> 免疫治疗
        ("IM", False, "atezolizumab", 1200.0, "PD-L1高/IM -> Atezolizumab"),
        # BRCA突变 -> PARP抑制剂
        ("BLIS", True, "olaparib", 300.0, "BRCA+ -> Olaparib"),
        # LAR亚型 -> AR拮抗剂
        ("LAR", False, "enzalutamide", 160.0, "LAR -> Enzalutamide"),
        # M亚型 -> 化疗
        ("M", False, "paclitaxel", 175.0, "M -> Paclitaxel"),
        # BLIS无BRCA -> 标准化疗
        ("BLIS", False, "doxorubicin", 60.0, "BLIS/BRCA- -> Doxorubicin"),
        # 对照: IM亚型用化疗(非最优)
        ("IM", False, "doxorubicin", 60.0, "IM -> Doxorubicin(对照)"),
    ]

    results = []
    for subtype, brca, drug, dose, label in strategies:
        print(f"\n--- {label} ---")
        result = run_stratified_arm(subtype, brca, drug, dose)
        result["label"] = label
        results.append(result)
        print(f"  最终体积: {result['final_volume']:.1f} mm3")
        print(f"  RECIST: {result['recist']}")
        print(f"  变化: {result['change_pct']:.1f}%")
        print(f"  PD-L1 CPS: {result['pd_l1_cps']:.1f}")

    # 分层效果分析
    print("\n" + "=" * 50)
    print("分层治疗 vs 非分层对照:")
    # IM亚型: 免疫治疗 vs 化疗
    im_immuno = [r for r in results if "Atezolizumab" in r["label"]][0]
    im_chemo = [r for r in results if "IM" in r["subtype"] and "对照" in r["label"]][0]
    print(f"  IM+免疫: {im_immuno['change_pct']:.1f}% vs IM+化疗: {im_chemo['change_pct']:.1f}%")
    delta = im_chemo['change_pct'] - im_immuno['change_pct']
    print(f"  分层获益: {delta:.1f}%")

    return {"strategies": results}


if __name__ == "__main__":
    run_experiment()
