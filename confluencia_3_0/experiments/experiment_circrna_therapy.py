"""circRNA治疗实验 (CircRNA Therapy Experiment)

测试三种circRNA治疗机制:
  1. miRNA海绵 (抑制致癌miRNA)
  2. 蛋白编码 (直接杀伤)
  3. 免疫刺激 (IFN-γ增强)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_circrna_arm(mechanism: str, n_days: int = 180) -> dict:
    """运行一种circRNA机制的模拟"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    history = []

    # 基线30天
    for _ in range(30):
        agent.step()

    # circRNA治疗150天
    from core.events import CIRCRNA_THERAPY_UPDATE
    agent.bus.publish(CIRCRNA_THERAPY_UPDATE, {
        "mechanism": mechanism,
        "dose": 1.0,
        "target": "miR-21" if mechanism == "mirna_sponge" else "p53" if mechanism == "protein_coding" else "RIG-I",
        "day": agent.day,
    }, source="experiment")

    for i in range(150):
        agent.step()
        if i % 15 == 0:
            s = agent.state
            history.append({
                "day": agent.day,
                "volume": s["tum_volume"],
                "recist": s["cli_recist_response"],
                "ifn_gamma": s["imm_ifn_gamma"],
                "cd8": s["imm_cd8_count"],
            })

    s = agent.state
    return {
        "mechanism": mechanism,
        "final_volume": s["tum_volume"],
        "recist": s["cli_recist_response"],
        "change_pct": s["cli_tumor_change_pct"],
        "ifn_gamma": s["imm_ifn_gamma"],
        "history": history,
    }


def run_experiment():
    """运行circRNA治疗实验"""
    print("circRNA治疗实验 - TNBC Simulacrum")
    print("=" * 50)

    mechanisms = ["mirna_sponge", "protein_coding", "immune_stimulation"]
    mechanism_names = {
        "mirna_sponge": "miRNA海绵",
        "protein_coding": "蛋白编码",
        "immune_stimulation": "免疫刺激",
    }

    results = {}
    for mech in mechanisms:
        print(f"\n--- {mechanism_names[mech]} ---")
        result = run_circrna_arm(mech)
        results[mech] = result
        print(f"  最终体积: {result['final_volume']:.1f} mm3")
        print(f"  RECIST: {result['recist']}")
        print(f"  变化: {result['change_pct']:.1f}%")
        print(f"  IFN-g: {result['ifn_gamma']:.3f}")

    # 比较
    print("\n" + "=" * 50)
    print("circRNA机制比较:")
    for mech, r in results.items():
        print(f"  {mechanism_names[mech]}: V={r['final_volume']:.1f}, "
              f"Change={r['change_pct']:.1f}%")

    best = min(results.items(), key=lambda x: x[1]['final_volume'])
    print(f"\n最有效机制: {mechanism_names[best[0]]}")

    return results


if __name__ == "__main__":
    run_experiment()
