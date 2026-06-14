"""放疗 + 远隔效应实验 (Radiotherapy + Abscopal Effect Experiment)

测试分次放疗的局部控制效果和远隔效应(免疫激活导致远处病灶缩小)。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


def run_experiment():
    """运行放疗+远隔效应实验"""
    print("放疗 + 远隔效应实验 - TNBC Simulacrum")
    print("=" * 50)

    config = TNBCSimulacrumConfig()
    config.molecular_subtype = "BLIS"
    config.experiment.seed = 42

    agent = TNBCSimulacrum(config)

    history = []

    # Phase 1: 基线 (30天)
    print("\nPhase 1: 基线 (30天)")
    for _ in range(30):
        agent.step()
    s = agent.state
    print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
          f"CD8={s['imm_cd8_count']:.0f}, "
          f"IFN-g={s['imm_ifn_gamma']:.3f}")

    # Phase 2: 分次放疗 (2Gy x 25次 = 50Gy, 5周)
    print("\nPhase 2: 分次放疗 (2Gy x 25次, 35天)")
    from core.events import RADIOTHERAPY_UPDATE
    for i in range(35):
        # 每个工作日(5/7)给2Gy
        if i % 7 < 5 and i < 25:
            agent.bus.publish(RADIOTHERAPY_UPDATE, {
                "fraction_dose": 2.0,
                "total_dose": (i // 7 * 5 + i % 7 + 1) * 2.0,
                "day": agent.day,
            }, source="experiment")
        agent.step()
        if i % 7 == 0:
            s = agent.state
            history.append({
                "day": agent.day,
                "volume": s["tum_volume"],
                "cd8": s["imm_cd8_count"],
                "ifn_gamma": s["imm_ifn_gamma"],
                "nk": s["imm_nk_cytotoxicity"],
                "met_burden": s["met_metastatic_burden"],
            })
            print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
                  f"CD8={s['imm_cd8_count']:.0f}, "
                  f"IFN-g={s['imm_ifn_gamma']:.3f}")

    # Phase 3: 观察远隔效应 (60天)
    print("\nPhase 3: 观察远隔效应 (60天)")
    for i in range(60):
        agent.step()
        if i % 15 == 0:
            s = agent.state
            history.append({
                "day": agent.day,
                "volume": s["tum_volume"],
                "cd8": s["imm_cd8_count"],
                "ifn_gamma": s["imm_ifn_gamma"],
                "met_burden": s["met_metastatic_burden"],
            })
            print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
                  f"CD8={s['imm_cd8_count']:.0f}, "
                  f"转移负荷={s['met_metastatic_burden']:.2f}")

    # Phase 4: 放疗 + anti-PD-1 联合 (验证远隔效应增强)
    print("\nPhase 4: 放疗 + anti-PD-1 联合 (60天)")
    agent.administer_drug("pembrolizumab", 200.0)
    for i in range(60):
        agent.step()
        if i % 15 == 0:
            s = agent.state
            print(f"  Day {agent.day}: V={s['tum_volume']:.1f}, "
                  f"CD8={s['imm_cd8_count']:.0f}, "
                  f"Exhaustion={s['imm_t_cell_exhaustion']:.3f}")

    print("\n实验完成")
    s = agent.state
    print(f"  最终体积: {s['tum_volume']:.1f} mm3")
    print(f"  RECIST: {s['cli_recist_response']}")
    print(f"  CD8+ T细胞: {s['imm_cd8_count']:.0f}")

    return {"history": history, "final_state": dict(s)}


if __name__ == "__main__":
    run_experiment()
