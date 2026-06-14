"""化疗响应实验 (Chemotherapy Response Experiment)

5阶段模拟:
  1. 基线生长 (100步)
  2. 化疗给药 (200步)
  3. 耐药出现 (200步)
  4. 换药 (200步)
  5. 观察 (100步)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def read_metrics(agent) -> Dict[str, Any]:
    """提取关键指标"""
    s = agent.state
    return {
        "day": agent.day,
        "volume": s.get("tum_volume", 0),
        "growth_rate": s.get("tum_growth_rate", 0),
        "kill_fraction": s.get("drg_kill_fraction", 0),
        "resistance": s.get("drg_resistance_level", 0),
        "recist": s.get("cli_recist_response", "SD"),
        "toxicity": s.get("cli_toxicity_grade", 0),
        "cd8_count": s.get("imm_cd8_count", 0),
        "exhaustion": s.get("imm_t_cell_exhaustion", 0),
        "csc_fraction": s.get("csc_fraction", 0),
        "pd_l1": s.get("evs_pd_l1_expression", 0),
        "ied_phase": s.get("ied_phase", "elimination"),
        "n_subclones": s.get("het_n_subclones", 0),
        "diversity": s.get("het_diversity_index", 0),
    }


def run_experiment():
    """运行化疗响应实验"""
    print("化疗响应实验 - TNBC Simulacrum")
    print("=" * 50)

    # 配置
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)
    metrics_history = []

    # Phase 1: 基线生长
    print("\nPhase 1: 基线生长 (100天)")
    for i in range(100):
        agent.step()
        if i % 20 == 0:
            m = read_metrics(agent)
            metrics_history.append(m)
            print(f"  Day {m['day']}: V={m['volume']:.1f}, IED={m['ied_phase']}")

    # Phase 2: 化疗
    print("\nPhase 2: 化疗 (doxorubicin, 200天)")
    agent.administer_drug("doxorubicin", 60.0)
    for i in range(200):
        agent.step()
        if i % 40 == 0:
            m = read_metrics(agent)
            metrics_history.append(m)
            print(f"  Day {m['day']}: V={m['volume']:.1f}, RECIST={m['recist']}, "
                  f"Kill={m['kill_fraction']:.3f}, Resist={m['resistance']:.3f}")

    # Phase 3: 耐药出现
    print("\nPhase 3: 继续化疗观察耐药 (200天)")
    for i in range(200):
        agent.step()
        if i % 40 == 0:
            m = read_metrics(agent)
            metrics_history.append(m)
            print(f"  Day {m['day']}: V={m['volume']:.1f}, RECIST={m['recist']}, "
                  f"Resist={m['resistance']:.3f}, Subclones={m['n_subclones']}")

    # Phase 4: 换药（铂类）
    print("\nPhase 4: 换药 carboplatin (200天)")
    agent.administer_drug("carboplatin", 400.0)
    for i in range(200):
        agent.step()
        if i % 40 == 0:
            m = read_metrics(agent)
            metrics_history.append(m)
            print(f"  Day {m['day']}: V={m['volume']:.1f}, RECIST={m['recist']}")

    # Phase 5: 观察期
    print("\nPhase 5: 观察期 (100天)")
    for i in range(100):
        agent.step()
        if i % 20 == 0:
            m = read_metrics(agent)
            metrics_history.append(m)
            print(f"  Day {m['day']}: V={m['volume']:.1f}, RECIST={m['recist']}")

    # 最终统计
    print("\n" + "=" * 50)
    print("实验完成")
    final = read_metrics(agent)
    print(f"  最终体积: {final['volume']:.1f} mm³")
    print(f"  RECIST: {final['recist']}")
    print(f"  耐药水平: {final['resistance']:.3f}")
    print(f"  亚克隆数: {final['n_subclones']}")
    print(f"  CSC比例: {final['csc_fraction']:.4f}")

    return metrics_history


from typing import Dict, Any

if __name__ == "__main__":
    run_experiment()