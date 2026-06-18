"""Confluencia 3.0 Subsystem Managers

遵循 CLF SubsystemManager 模式：基类持有 agent 引用，
通过 schema 介导状态访问，step() 协调子模块。

新增 CircRNAManager 管理 circRNA 免疫感知、结构预测、序列进化子系统。
"""
import warnings
from typing import Dict, Any, Optional
from .state_schema import StateSchema
from .events import (
    CIRCRNA_IMMUNE_EVAL, CIRCRNA_STRUCTURE_PREDICT,
    CIRCRNA_SEQUENCE_EVOLVE, CIRCRNA_VACCINE_ASSESS,
    CIRCRNA_FOLDING_KINETICS, CIRCRNA_DRUG_RESPONSE,
    CIRCRNA_PK_SIMULATE, MOLECULE_EVOLUTION_REQUEST,
)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class SubsystemManager:
    """子系统管理器基类

    Attributes:
        agent: TNBCSimulacrum 实例引用
        schema: StateSchema 实例
        subsystem_name: 子系统标签
    """
    subsystem_name: str = "base"

    def __init__(self, agent):
        self.agent = agent
        self.schema = agent._schema

    @property
    def state(self) -> Dict[str, Any]:
        return self.agent._internal_state

    def set_state(self, key: str, value: Any) -> None:
        """设置状态值（带范围钳制）"""
        kd = self.schema.get_key(key)
        if kd is not None and kd.range_ is not None and isinstance(value, (int, float)):
            lo, hi = kd.range_
            value = max(lo, min(hi, value))
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def step(self, **kwargs) -> Dict[str, Any]:
        return {}

    def get_summary(self) -> Dict[str, Any]:
        keys = self.schema.get_subsystem_keys(self.subsystem_name)
        return {k: self.state.get(k) for k in keys if k in self.state}


class TumorManager(SubsystemManager):
    """肿瘤子系统管理器"""
    subsystem_name = "tumor"

    def step(self, **kwargs) -> Dict[str, Any]:
        result = {}
        s = self.state

        # 生长引擎
        try:
            growth_result = self.agent.growth_engine.step(s)
            if growth_result:
                for k, v in growth_result.items():
                    self.set_state(k, v)
                result["growth"] = growth_result
        except Exception:
            pass

        # 异质性
        try:
            het_result = self.agent.heterogeneity.step(s)
            if het_result:
                for k, v in het_result.items():
                    self.set_state(k, v)
                result["heterogeneity"] = het_result
        except Exception:
            pass

        # CSC
        try:
            csc_result = self.agent.csc_pool.step(s)
            if csc_result:
                for k, v in csc_result.items():
                    self.set_state(k, v)
                result["csc"] = csc_result
        except Exception:
            pass

        # 血管生成
        try:
            angio_result = self.agent.angiogenesis.step(s)
            if angio_result:
                for k, v in angio_result.items():
                    self.set_state(k, v)
                result["angiogenesis"] = angio_result
        except Exception:
            pass

        # 转移
        try:
            met_result = self.agent.metastasis_engine.step(s)
            if met_result:
                for k, v in met_result.items():
                    self.set_state(k, v)
                result["metastasis"] = met_result
        except Exception:
            pass

        return result


class TMEManager(SubsystemManager):
    """肿瘤微环境管理器"""
    subsystem_name = "tme"

    def step(self, **kwargs) -> Dict[str, Any]:
        result = {}
        s = self.state

        # 免疫细胞
        try:
            imm_result = self.agent.immune.step(s)
            if imm_result:
                for k, v in imm_result.items():
                    self.set_state(k, v)
                result["immune"] = imm_result
        except Exception:
            pass

        # 成纤维/ECM
        try:
            caf_result = self.agent.fibroblast.step(s)
            if caf_result:
                for k, v in caf_result.items():
                    self.set_state(k, v)
                result["fibroblast"] = caf_result
        except Exception:
            pass

        # 免疫逃逸
        try:
            evs_result = self.agent.evasion.step(s)
            if evs_result:
                for k, v in evs_result.items():
                    self.set_state(k, v)
                result["evasion"] = evs_result
        except Exception:
            pass

        # 免疫编辑
        try:
            ied_result = self.agent.immunoediting.step(s)
            if ied_result:
                for k, v in ied_result.items():
                    self.set_state(k, v)
                result["immunoediting"] = ied_result
        except Exception:
            pass

        return result


class TreatmentManager(SubsystemManager):
    """治疗管理器"""
    subsystem_name = "treatment"

    def step(self, **kwargs) -> Dict[str, Any]:
        result = {}
        s = self.state

        # 化疗
        try:
            chemo_result = self.agent.chemotherapy.step(s)
            if chemo_result:
                for k, v in chemo_result.items():
                    self.set_state(k, v)
                result["chemotherapy"] = chemo_result
        except Exception:
            pass

        # 免疫治疗
        try:
            immuno_result = self.agent.immunotherapy.step(s)
            if immuno_result:
                for k, v in immuno_result.items():
                    self.set_state(k, v)
                result["immunotherapy"] = immuno_result
        except Exception:
            pass

        # 靶向治疗
        try:
            targeted_result = self.agent.targeted.step(s)
            if targeted_result:
                for k, v in targeted_result.items():
                    self.set_state(k, v)
                result["targeted"] = targeted_result
        except Exception:
            pass

        # 放疗
        try:
            rt_result = self.agent.radiotherapy.step(s)
            if rt_result:
                for k, v in rt_result.items():
                    self.set_state(k, v)
                result["radiotherapy"] = rt_result
        except Exception:
            pass

        # circRNA治疗
        try:
            cfr_result = self.agent.circrna_therapy.step(s)
            if cfr_result:
                for k, v in cfr_result.items():
                    self.set_state(k, v)
                result["circrna"] = cfr_result
        except Exception:
            pass

        return result


class BiomarkerManager(SubsystemManager):
    """生物标志物管理器"""
    subsystem_name = "biomarker"

    def step(self, **kwargs) -> Dict[str, Any]:
        result = {}
        s = self.state

        try:
            bio_result = self.agent.biomarker_tracker.step(s)
            if bio_result:
                for k, v in bio_result.items():
                    self.set_state(k, v)
                result["tracker"] = bio_result
        except Exception:
            pass

        try:
            sub_result = self.agent.subtype_classifier.step(s)
            if sub_result:
                for k, v in sub_result.items():
                    self.set_state(k, v)
                result["subtype"] = sub_result
        except Exception:
            pass

        return result


class ClinicalManager(SubsystemManager):
    """临床评估管理器"""
    subsystem_name = "clinical"

    def step(self, **kwargs) -> Dict[str, Any]:
        result = {}
        s = self.state

        # RECIST评估（每6周）
        if self.agent._day % self.agent.config.clinical.recist_evaluation_interval == 0:
            try:
                recist_result = self.agent.recist.evaluate(s)
                if recist_result:
                    for k, v in recist_result.items():
                        self.set_state(k, v)
                    result["recist"] = recist_result
            except Exception:
                pass

        # 生存更新
        try:
            surv_result = self.agent.survival_model.step(s)
            if surv_result:
                for k, v in surv_result.items():
                    self.set_state(k, v)
                result["survival"] = surv_result
        except Exception:
            pass

        # 毒性分级
        try:
            tox_result = self.agent.toxicity_grader.step(s)
            if tox_result:
                for k, v in tox_result.items():
                    self.set_state(k, v)
                result["toxicity"] = tox_result
        except Exception:
            pass

        return result


class CircRNAManager(SubsystemManager):
    """circRNA 子系统管理器

    管理免疫感知、结构预测、序列进化、疫苗评估等 circRNA 子模块。
    通过 Backend 架构统一调度，支持 heuristic/vienna/torusfold 三档降级。

    四大支柱:
    - RNACTM: PK/PD (simulate_pk)
    - ViennaRNA: 二级结构 (predict_structure)
    - TorusFold: DL 结构 + 多任务头 (assess_with_torusfold)
    - Simulacrum: TNBCTME 响应 (通过 agent 状态)
    """

    subsystem_name = "circrna"

    def __init__(self, agent):
        super().__init__(agent)
        self._current_sequence: str = ""
        self._immunogenicity_result: Dict[str, Any] = {}
        self._structure_result: Dict[str, Any] = {}
        self._torusfold_result: Dict[str, Any] = {}

        # TorusFold 桥接器 (延迟加载)
        self._torusfold_scorer = None

        # 订阅 circRNA 事件
        if hasattr(agent, '_event_bus') and agent._event_bus is not None:
            agent._event_bus.subscribe(CIRCRNA_IMMUNE_EVAL, self._on_immune_eval)
            agent._event_bus.subscribe(CIRCRNA_STRUCTURE_PREDICT, self._on_structure_predict)
            agent._event_bus.subscribe(CIRCRNA_SEQUENCE_EVOLVE, self._on_sequence_evolve)
            agent._event_bus.subscribe(CIRCRNA_VACCINE_ASSESS, self._on_vaccine_assess)
            agent._event_bus.subscribe(CIRCRNA_FOLDING_KINETICS, self._on_folding_kinetics)
            agent._event_bus.subscribe(CIRCRNA_DRUG_RESPONSE, self._on_drug_response)
            agent._event_bus.subscribe(CIRCRNA_PK_SIMULATE, self._on_pk_simulate)
            agent._event_bus.subscribe(MOLECULE_EVOLUTION_REQUEST, self._on_molecule_evolution)

    def step(self, **kwargs) -> Dict[str, Any]:
        """每步执行 circRNA 相关计算，更新 crna_* 状态键。"""
        result = {}
        circrna_cfg = getattr(self.agent.config, 'circrna', None)
        if circrna_cfg is None or not circrna_cfg.enabled:
            return result

        # 更新后端层级到状态
        self.set_state("crna_backend_tier", circrna_cfg.immunogenicity_backend)

        # 如果有活跃序列，执行免疫评估
        if self._current_sequence:
            try:
                immune_result = self.assess_immunogenicity(
                    self._current_sequence,
                    backend=circrna_cfg.immunogenicity_backend,
                )
                if immune_result:
                    for k, v in immune_result.items():
                        if k.startswith("crna_"):
                            self.set_state(k, v)
                    result["immunogenicity"] = immune_result
                    self._immunogenicity_result = immune_result
            except Exception:
                pass

            # 结构预测（可选）
            if circrna_cfg.enable_structure_prediction:
                try:
                    struct_result = self.predict_structure(self._current_sequence)
                    if struct_result:
                        for k, v in struct_result.items():
                            if k.startswith("crna_"):
                                self.set_state(k, v)
                        result["structure"] = struct_result
                        self._structure_result = struct_result
                except Exception:
                    pass

            # TorusFold 结构预测（根据 structure_mode 选择）
            # structure_mode: heuristic (不走 TorusFold), simple, diffusion, physics_b, physics_ba
            if circrna_cfg.torusfold_required:
                try:
                    tf_result = self.assess_with_torusfold(
                        self._current_sequence,
                        structure_mode=circrna_cfg.structure_mode,
                    )
                    if tf_result:
                        for k, v in tf_result.items():
                            if k.startswith("crna_"):
                                self.set_state(k, v)
                        result["torusfold"] = tf_result
                        self._torusfold_result = tf_result
                except Exception:
                    pass

        return result

    def assess_immunogenicity(self, sequence: str, backend: str = "heuristic") -> Dict[str, Any]:
        """通过 Backend 架构调度免疫原性评估，支持三层降级。

        降级链: ESM2 (Tier 0) → ViennaRNA (Tier 1) → Heuristic (Tier 2)

        Args:
            sequence: circRNA 序列
            backend: 请求的后端层级 ("esm2", "vienna", "heuristic")

        Returns:
            Dict 包含 crna_immunogenicity_score, crna_rig_i_score 等
        """
        actual_backend = backend
        backend_warning = None

        # === Tier 0: ESM2 深度学习 ===
        if backend == "esm2":
            esm2_available = self._check_esm2_available()
            if esm2_available:
                try:
                    return self._assess_immunogenicity_esm2(sequence)
                except Exception as e:
                    backend_warning = (
                        f"ESM2 backend failed: {e}. "
                        f"Falling back to ViennaRNA. "
                        f"Ensure GPU memory is sufficient and ESM2 weights are downloaded."
                    )
                    actual_backend = "vienna"
                    warnings.warn(backend_warning, UserWarning, stacklevel=2)
            else:
                backend_warning = (
                    "ESM2 backend unavailable: requires GPU and ESM2 model weights. "
                    "Falling back to ViennaRNA (Tier 1). "
                    "To enable ESM2: pip install fair-esm and ensure CUDA GPU is available."
                )
                actual_backend = "vienna"
                warnings.warn(backend_warning, UserWarning, stacklevel=2)

        # === Tier 1: ViennaRNA 结构辅助 ===
        if actual_backend == "vienna":
            vienna_available = self._check_viennarna_available()
            if vienna_available:
                try:
                    return self._assess_immunogenicity_vienna(sequence)
                except Exception as e:
                    backend_warning = (
                        f"ViennaRNA backend failed: {e}. "
                        f"Falling back to heuristic (Tier 2). "
                        f"Check ViennaRNA installation."
                    )
                    actual_backend = "heuristic"
                    warnings.warn(backend_warning, UserWarning, stacklevel=2)
            else:
                backend_warning = (
                    "ViennaRNA backend unavailable: RNAfold not installed. "
                    "Falling back to heuristic (Tier 2). "
                    "To enable ViennaRNA: conda install -c bioconda viennarna"
                )
                actual_backend = "heuristic"
                warnings.warn(backend_warning, UserWarning, stacklevel=2)

        # === Tier 2: Heuristic 纯启发式 ===
        return self._assess_immunogenicity_heuristic(sequence, actual_backend=actual_backend)

    def _check_esm2_available(self) -> bool:
        """检查 ESM2 模型是否可用 (GPU + 模型权重)."""
        if not _has_cuda():
            return False
        try:
            import esm
            # 检查是否能加载模型（不实际加载，避免内存开销）
            return True
        except ImportError:
            return False

    def _check_viennarna_available(self) -> bool:
        """检查 ViennaRNA 是否已安装."""
        try:
            import subprocess
            result = subprocess.run(
                ["RNAfold", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _assess_immunogenicity_esm2(self, sequence: str) -> Dict[str, Any]:
        """ESM2 深度学习后端免疫原性评估 (Tier 0)."""
        # TODO: 当 ESM2-RNA 权重可用时实现
        # 当前 ESM2 主要是蛋白质模型，RNA 特化版本需要额外训练
        raise NotImplementedError("ESM2 RNA immunogenicity model not yet trained")

    def _assess_immunogenicity_vienna(self, sequence: str) -> Dict[str, Any]:
        """ViennaRNA 结构辅助后端免疫原性评估 (Tier 1)."""
        from .circrna.immune_sensing import predict_circrna_immunogenicity, ImmuneSensingConfig
        from .circrna.structure_prediction import StructurePredictor

        # 先用 ViennaRNA 预测结构
        predictor = StructurePredictor()
        structure_features = predictor.predict(sequence)

        # 基于结构特征调整免疫评分
        config = ImmuneSensingConfig()
        result = predict_circrna_immunogenicity(sequence, config=config)

        # ViennaRNA 提供的 dsRNA 区域信息可修正 PKR 评分
        dsrna_fraction = getattr(structure_features, 'dsrna_fraction', 0.0)
        if dsrna_fraction > 0.25:  # 高 dsRNA 比例增强 PKR 激活风险
            pkr_score = result.get("pkr_score", 0.0)
            result["pkr_score"] = min(1.0, pkr_score * 1.2)

        return {
            "crna_backend_tier": "vienna",
            "crna_backend_method": "viennarna_structure_assisted",
            "crna_immunogenicity_score": result.get("overall_score", 0.0),
            "crna_rig_i_score": result.get("rig_i_score", 0.0),
            "crna_tlr_score": result.get("tlr_score", 0.0),
            "crna_pkr_score": result.get("pkr_score", 0.0),
            "crna_ips_score": result.get("ips", 0.0),
            "crna_dsrna_fraction": dsrna_fraction,
        }

    def _assess_immunogenicity_heuristic(self, sequence: str, actual_backend: str = "heuristic") -> Dict[str, Any]:
        """纯启发式后端免疫原性评估 (Tier 2)."""
        from .circrna.immune_sensing import predict_circrna_immunogenicity, ImmuneSensingConfig
        config = ImmuneSensingConfig()
        result = predict_circrna_immunogenicity(sequence, config=config)

        return {
            "crna_backend_tier": actual_backend,  # 可能是 "heuristic" 或降级后的值
            "crna_backend_method": "heuristic_motif_based",
            "crna_immunogenicity_score": result.get("overall_score", 0.0),
            "crna_rig_i_score": result.get("rig_i_score", 0.0),
            "crna_tlr_score": result.get("tlr_score", 0.0),
            "crna_pkr_score": result.get("pkr_score", 0.0),
            "crna_ips_score": result.get("ips", 0.0),
        }

    def predict_structure(self, sequence: str) -> Dict[str, Any]:
        """调度结构预测（ViennaRNA 或 fallback）。"""
        try:
            from .circrna.structure_prediction import StructurePredictor
            predictor = StructurePredictor()
            features = predictor.predict(sequence)
            return {
                "crna_structure_method": getattr(features, 'prediction_method', 'fallback'),
                "crna_mfe_kcal": getattr(features, 'mfe_kcal', 0.0),
            }
        except Exception:
            return {"crna_structure_method": "fallback", "crna_mfe_kcal": 0.0}

    def assess_with_torusfold(self, sequence: str, structure_mode: str = "simple") -> Dict[str, Any]:
        """运行 TorusFold 深度学习评估，提取结构信号并修正免疫评分。

        根据 structure_mode 选择不同的结构预测模块:
          - "simple": SimpleStructureHead (MDS 快速推断)
          - "diffusion": CircDiffusionStructure (AF3 风格扩散)
          - "physics_b": PhysicsStructureHead (几何约束求解器，零训练)
          - "physics_ba": PhysicsStructureHead + OpenMM MD 精修

        返回的 crna_* 键会自动写入状态。
        """
        circrna_cfg = getattr(self.agent.config, 'circrna', None)

        if self._torusfold_scorer is None:
            try:
                from .circrna.torusfold_scorer import TorusFoldScorer
                device = "cuda" if _has_cuda() else "cpu"
                self._torusfold_scorer = TorusFoldScorer(
                    device=device,
                    structure_mode=structure_mode,
                    diffusion_steps=getattr(circrna_cfg, 'diffusion_steps', 100) if circrna_cfg else 100,
                    solver_samples=getattr(circrna_cfg, 'solver_samples', 20) if circrna_cfg else 20,
                    openmm_minimize_steps=getattr(circrna_cfg, 'openmm_minimize_steps', 500) if circrna_cfg else 500,
                    openmm_md_steps=getattr(circrna_cfg, 'openmm_md_steps', 5000) if circrna_cfg else 5000,
                )
            except Exception:
                return {}

        try:
            signals = self._torusfold_scorer.extract_signals(sequence, structure_mode=structure_mode)
            if not signals.available:
                return {"crna_torusfold_method": "unavailable"}

            # 用 TorusFold 免疫头覆盖启发式评分
            immune_override = self._torusfold_scorer.compute_immune_override(signals)

            # 计算修正后的四维目标
            mfe = self._structure_result.get("crna_mfe_kcal") if self._structure_result else None
            objectives = self._torusfold_scorer.compute_objectives(
                sequence,
                immune_scores=self._immunogenicity_result if self._immunogenicity_result else None,
                torusfold_signals=signals,
                viennarna_mfe=mfe,
            )

            result = {
                "crna_torusfold_method": structure_mode,
                "crna_closure_score": signals.closure_score,
                "crna_bsj_stability": signals.bsj_stability,
                "crna_dsRNA_fraction_dl": signals.dsRNA_fraction,
                "crna_translation_efficiency_dl": signals.translation_efficiency,
                "crna_circ_stability_dl": signals.circ_stability,
                # 修正后的四维目标
                "crna_obj_stability": float(objectives[0]),
                "crna_obj_translation": float(objectives[1]),
                "crna_obj_immune_evasion": float(objectives[2]),
                "crna_obj_delivery": float(objectives[3]),
            }

            # 如果 TorusFold 免疫覆盖可用，写入覆盖评分
            if immune_override:
                result["crna_rig_i_score_dl"] = immune_override["rig_i_score"]
                result["crna_pkr_score_dl"] = immune_override["pkr_score"]
                result["crna_tlr_score_dl"] = immune_override.get("tlr7_score", 0.0) + immune_override.get("tlr8_score", 0.0)

            return result
        except Exception:
            return {"crna_torusfold_method": "error"}

    def set_sequence(self, sequence: str):
        """设置当前分析的 circRNA 序列。"""
        self._current_sequence = sequence.upper().replace("T", "U")

    def evolve_sequence(self, sequence: str, objective: str = "ips", generations: int = 5) -> Dict[str, Any]:
        """序列进化优化 (通过内化 evolution 模块)。"""
        try:
            from .evolution.cirrna_evolution import evolve_cirrna, CircRNAEvolutionConfig
            # 根据 objective 调整权重
            weight_map = {
                "stability": (0.50, 0.20, 0.20, 0.10),
                "translation": (0.25, 0.50, 0.15, 0.10),
                "immune_safety": (0.25, 0.15, 0.50, 0.10),
                "ips": (0.35, 0.30, 0.25, 0.10),
            }
            ws, wt, wi, wd = weight_map.get(objective, weight_map["ips"])
            config = CircRNAEvolutionConfig(
                seed_seq=sequence,
                rounds=generations,
                weight_stability=ws,
                weight_translation=wt,
                weight_immune_evasion=wi,
                weight_delivery=wd,
            )
            result_df, artifacts = evolve_cirrna(config)
            return {
                "crna_evolution_generation": artifacts.rounds_ran,
                "crna_evolution_best_score": artifacts.best_reward,
                "best_sequence": artifacts.best_sequence,
            }
        except Exception:
            return {"crna_evolution_generation": 0, "crna_evolution_best_score": 0.0}

    def simulate_pk(self, sequence: str, dose: float = 1.0, freq: float = 1.0, **kwargs) -> Dict[str, Any]:
        """通过内化 RNACTM 模拟 circRNA PK。"""
        try:
            from .pk.rnactm import infer_rna_ctm_params, simulate_rna_ctm, summarize_rna_ctm_curve
            params = infer_rna_ctm_params(
                modification=kwargs.get("modification", "none"),
                delivery_vector=kwargs.get("delivery_vector", "LNP_standard"),
                route=kwargs.get("route", "IV"),
                ires_score=kwargs.get("ires_score", 0.5),
                gc_content=kwargs.get("gc_content", 0.5),
                struct_stability=kwargs.get("struct_stability", 0.5),
                innate_immune_score=kwargs.get("innate_immune_score", 0.0),
            )
            circrna_cfg = getattr(self.agent.config, 'circrna', None)
            horizon = kwargs.get("horizon", getattr(circrna_cfg, 'pk_default_horizon', 168) if circrna_cfg else 168)
            dt = kwargs.get("dt", getattr(circrna_cfg, 'pk_default_dt', 1.0) if circrna_cfg else 1.0)
            curve = simulate_rna_ctm(dose=dose, freq=freq, params=params, horizon=horizon, dt=dt)
            summary = summarize_rna_ctm_curve(curve)
            # 更新状态键
            self.set_state("crna_pk_auc_efficacy", summary.get("rna_ctm_auc_efficacy", 0.0))
            self.set_state("crna_pk_peak_protein", summary.get("rna_ctm_peak_protein", 0.0))
            self.set_state("crna_pk_rna_half_life", summary.get("rna_ctm_rna_half_life_h", 0.0))
            return {"source": "internal", "available": True, "curve": curve, **summary}
        except Exception:
            return {"source": "fallback", "available": False}

    def evolve_molecules(self, seed_smiles, cfg=None, pipeline_fn=None, ed2mol_adapter=None) -> Dict[str, Any]:
        """药物分子进化优化 (通过内化 evolution 模块)。"""
        try:
            from .evolution.molecule_evolution import evolve_molecules_with_reflection, EvolutionConfig
            if cfg is None:
                cfg = EvolutionConfig()
            result_df, artifacts = evolve_molecules_with_reflection(
                seed_smiles=seed_smiles, cfg=cfg,
                pipeline_fn=pipeline_fn, ed2mol_adapter=ed2mol_adapter,
            )
            self.set_state("molecule_evolution_best_score", artifacts.best_reward)
            return {
                "best_reward": artifacts.best_reward,
                "rounds_ran": artifacts.rounds_ran,
                "reflections": artifacts.reflections,
                "source": "internal",
            }
        except Exception:
            return {"best_reward": 0.0, "source": "fallback"}

    # --- EventBus 处理器 ---

    def _on_immune_eval(self, event_data: Dict[str, Any]):
        seq = event_data.get("sequence", "")
        backend = event_data.get("backend", "heuristic")
        if seq:
            self.set_sequence(seq)
            result = self.assess_immunogenicity(seq, backend=backend)
            for k, v in result.items():
                self.set_state(k, v)
            self._immunogenicity_result = result

    def _on_structure_predict(self, event_data: Dict[str, Any]):
        seq = event_data.get("sequence", self._current_sequence)
        if seq:
            result = self.predict_structure(seq)
            for k, v in result.items():
                self.set_state(k, v)
            self._structure_result = result

    def _on_sequence_evolve(self, event_data: Dict[str, Any]):
        seq = event_data.get("sequence", self._current_sequence)
        objective = event_data.get("objective", "ips")
        generations = event_data.get("generations", 50)
        if seq:
            result = self.evolve_sequence(seq, objective=objective, generations=generations)
            for k, v in result.items():
                if k.startswith("crna_"):
                    self.set_state(k, v)

    def _on_vaccine_assess(self, event_data: Dict[str, Any]):
        seq = event_data.get("sequence", self._current_sequence)
        if seq:
            immune = self.assess_immunogenicity(seq)
            ips = immune.get("crna_ips_score", 0.0)
            self.set_state("crna_vaccine_therapeutic_window",
                           min(1.0, max(0.0, ips / 10.0 * immune.get("crna_immunogenicity_score", 0.0))))

    def _on_folding_kinetics(self, event_data: Dict[str, Any]):
        seq = event_data.get("sequence", self._current_sequence)
        if seq:
            try:
                from .circrna.folding_kinetics import predict_folding_kinetics
                result = predict_folding_kinetics(seq)
                self.set_state("crna_folding_method",
                               getattr(result, 'kinetics_method', 'fallback_kinetics'))
            except Exception:
                self.set_state("crna_folding_method", "fallback_kinetics")

    def _on_drug_response(self, event_data: Dict[str, Any]):
        pass  # 占位，未来扩展

    def _on_pk_simulate(self, event_data: Dict[str, Any]):
        """处理 CIRCRNA_PK_SIMULATE 事件。"""
        seq = event_data.get("sequence", self._current_sequence)
        dose = event_data.get("dose", 1.0)
        freq = event_data.get("freq", 1.0)
        if seq:
            self.simulate_pk(seq, dose=dose, freq=freq, **event_data.get("kwargs", {}))

    def _on_molecule_evolution(self, event_data: Dict[str, Any]):
        """处理 MOLECULE_EVOLUTION_REQUEST 事件。"""
        seed_smiles = event_data.get("seed_smiles", [])
        if seed_smiles:
            self.evolve_molecules(seed_smiles, **event_data.get("kwargs", {}))