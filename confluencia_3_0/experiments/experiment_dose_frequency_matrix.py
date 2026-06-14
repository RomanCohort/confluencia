"""剂量-频率矩阵实验 (Dose-Frequency Matrix Experiment)

探索不同剂量和给药频率组合的治疗效果和毒性。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_dose_freq_arm(dose: float, interval: int, n_days: int = 180) -> dict:
    """运行一个剂量-频率组合"""
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    # 基线30天
    for _ in range(30):
        agent.step()

    # 按间隔给药
    for day in range(150):
        if day % interval == 0:
            agent.administer_drug("paclitaxel", dose)
        agent.step()

    s = agent.state
    return {
        "dose": dose,
        "interval": interval,
        "final_volume": s["tum_volume"],
        "recist": s["cli_recist_response"],
        "change_pct": s["cli_tumor_change_pct"],
        "toxicity": s["cli_toxicity_grade"],
        "resistance": s["drg_resistance_level"],
    }


def run_experiment():
    """运行剂量-频率矩阵实验"""
    print("剂量-频率矩阵实验 - TNBC Simulacrum")
    print("=" * 50)

    doses = [80.0, 135.0, 175.0, 220.0]
    intervals = [7, 14, 21, 28]

    results = []
    for dose in doses:
        for interval in intervals:
            result = run_dose_freq_arm(dose, interval)
            results.append(result)
            print(f"  Dose={dose:.0f}, Interval={interval}d: "
                  f"V={result['final_volume']:.1f}, "
                  f"RECIST={result['recist']}, "
                  f"Tox={result['toxicity']}, "
                  f"Resist={result['resistance']:.3f}")

    # 最优组合
    print("\n" + "=" * 50)
    # 筛选有效且低毒性
    effective = [r for r in results if r['recist'] in ('CR', 'PR')]
    if effective:
        best = min(effective, key=lambda x: (x['toxicity'], x['final_volume']))
        print(f"最优组合: Dose={best['dose']:.0f}, Interval={best['interval']}d")
        print(f"  体积={best['final_volume']:.1f}, RECIST={best['recist']}, "
              f"毒性={best['toxicity']}")
    else:
        best = min(results, key=lambda x: x['final_volume'])
        print(f"最佳体积组合: Dose={best['dose']:.0f}, Interval={best['interval']}d")
        print(f"  体积={best['final_volume']:.1f}, RECIST={best['recist']}")

    return {"matrix": results, "best": best}


if __name__ == "__main__":
    run_experiment()
