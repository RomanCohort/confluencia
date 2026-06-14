"""Confluencia PK/PD 集成实验"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_experiment():
    """运行Confluencia集成实验"""
    print("Confluencia PK/PD 集成实验 - TNBC Simulacrum")
    print("=" * 50)

    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.confluencia.enabled = True
    config.confluencia.confluencia_path = "D:/IGEM集成方案"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    # 基线
    print("\n基线 (30天)")
    for _ in range(30):
        agent.step()

    # 化疗
    print("\n化疗 (doxorubicin, 150天)")
    agent.administer_drug("doxorubicin", 60.0)
    for i in range(150):
        agent.step()
        if i % 30 == 0:
            s = agent.state
            print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
                  f"RECIST={s['cli_recist_response']}, "
                  f"Confluencia={s['cfl_drug_prediction_score']:.3f}")

    print("\n实验完成")


if __name__ == "__main__":
    run_experiment()