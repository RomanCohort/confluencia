"""耐药演化实验 (Resistance Evolution Experiment)

追踪亚克隆演化、耐药突变出现和交叉耐药模式。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_experiment():
    """运行耐药演化实验"""
    print("耐药演化实验 - TNBC Simulacrum")
    print("=" * 50)

    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    # 基线
    print("\n基线 (30天)")
    for _ in range(30):
        agent.step()
    print(f"  Day {agent.day}: V={agent.state['tum_volume']:.1f}, "
          f"Subclones={agent.state['het_n_subclones']}, "
          f"Resistance={agent.state['drg_resistance_level']:.3f}")

    # 连续化疗
    print("\n连续化疗 (300天)")
    agent.administer_drug("paclitaxel", 175.0)
    resistance_emerged_day = None

    for i in range(300):
        agent.step()
        if i % 30 == 0:
            s = agent.state
            print(f"  Day {s.get('day', 0) if 'day' in s else agent.day}: "
                  f"V={s['tum_volume']:.1f}, "
                  f"Subclones={s['het_n_subclones']}, "
                  f"Resistance={s['drg_resistance_level']:.3f}, "
                  f"Diversity={s['het_diversity_index']:.3f}")

            if s['drg_resistance_level'] > 0.3 and resistance_emerged_day is None:
                resistance_emerged_day = agent.day
                print(f"  *** 耐药检测: Day {resistance_emerged_day} ***")

    # 换药
    print("\n换药 (carboplatin, 100天)")
    agent.administer_drug("carboplatin", 400.0)
    for i in range(100):
        agent.step()
    s = agent.state
    print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
          f"RECIST={s['cli_recist_response']}")

    print("\n实验完成")
    if resistance_emerged_day:
        print(f"  耐药出现时间: Day {resistance_emerged_day}")
    else:
        print("  未检测到耐药")


if __name__ == "__main__":
    run_experiment()