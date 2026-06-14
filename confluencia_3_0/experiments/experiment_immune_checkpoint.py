"""免疫检查点实验 (Immune Checkpoint Experiment)

测试 anti-PD-1/PD-L1 疗效:
  1. 基线免疫动态
  2. anti-PD-1 单药
  3. 化疗 + anti-PD-1 联合
  4. 洗脱期
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_experiment():
    """运行免疫检查点实验"""
    print("免疫检查点实验 - TNBC Simulacrum")
    print("=" * 50)

    # IM亚型（高PD-L1，适合免疫治疗）
    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "IM"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    # Phase 1: 基线
    print("\nPhase 1: 基线 (60天)")
    for i in range(60):
        agent.step()
    s = agent.state
    print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
          f"PD-L1={s['evs_pd_l1_expression']:.2f}, "
          f"Exhaustion={s['imm_t_cell_exhaustion']:.2f}, "
          f"IED={s['ied_phase']}")

    # Phase 2: anti-PD-1
    print("\nPhase 2: anti-PD-1 (atezolizumab, 120天)")
    agent.administer_drug("atezolizumab", 1200.0)
    for i in range(120):
        agent.step()
    s = agent.state
    print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
          f"Exhaustion={s['imm_t_cell_exhaustion']:.2f}, "
          f"Activation={s['imm_t_cell_activation']:.2f}, "
          f"RECIST={s['cli_recist_response']}")

    # Phase 3: 联合（化疗 + anti-PD-1）
    print("\nPhase 3: 化疗 + anti-PD-1 联合 (120天)")
    agent.administer_drug("paclitaxel", 175.0)
    for i in range(120):
        agent.step()
    s = agent.state
    print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
          f"RECIST={s['cli_recist_response']}, "
          f"IED={s['ied_phase']}")

    # Phase 4: 洗脱
    print("\nPhase 4: 洗脱期 (60天)")
    for i in range(60):
        agent.step()
    s = agent.state
    print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
          f"RECIST={s['cli_recist_response']}, "
          f"Exhaustion={s['imm_t_cell_exhaustion']:.2f}")

    print("\n实验完成")


if __name__ == "__main__":
    run_experiment()