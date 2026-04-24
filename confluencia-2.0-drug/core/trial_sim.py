from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Virtual Cohort
# ---------------------------------------------------------------------------


@dataclass
class CohortConfig:
    n_patients: int = 100
    age_range: Tuple[float, float] = (18.0, 75.0)
    age_mean: float = 55.0
    age_std: float = 12.0
    female_frac: float = 0.45
    weight_mean: float = 70.0
    weight_std: float = 15.0
    disease_stages: List[str] = field(default_factory=lambda: ["I", "II", "III"])
    stage_probs: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    biomarker_positive_frac: float = 0.4
    ecog_scores: List[int] = field(default_factory=lambda: [0, 1, 2])
    ecog_probs: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.2])
    seed: int = 42


def generate_cohort(config: CohortConfig | None = None) -> pd.DataFrame:
    """Generate virtual patient cohort.

    Returns a DataFrame with columns:
        patient_id, age, sex, weight_kg, disease_stage, biomarker_positive,
        ecog_score, baseline_risk_score
    """
    if config is None:
        config = CohortConfig()

    rng = np.random.default_rng(config.seed)
    n = config.n_patients

    # Age: truncated normal within range
    ages = _truncated_normal(rng, config.age_mean, config.age_std,
                              config.age_range[0], config.age_range[1], n)

    # Sex
    female = rng.random(n) < config.female_frac
    sex = np.where(female, "F", "M")

    # Weight: normally distributed, slightly correlated with sex
    base_weight = rng.normal(config.weight_mean, config.weight_std, n)
    weight_adj = np.where(female, -4.0, 4.0)
    weight_kg = np.clip(base_weight + weight_adj, 35.0, 150.0)

    # Disease stage
    stage_probs_norm = np.array(config.stage_probs, dtype=np.float64)
    stage_probs_norm = stage_probs_norm / stage_probs_norm.sum()
    disease_stage = rng.choice(config.disease_stages, size=n, p=stage_probs_norm)

    # Biomarker
    biomarker_positive = rng.random(n) < config.biomarker_positive_frac

    # ECOG
    ecog_probs_norm = np.array(config.ecog_probs, dtype=np.float64)
    ecog_probs_norm = ecog_probs_norm / ecog_probs_norm.sum()
    ecog_score = rng.choice(config.ecog_scores, size=n, p=ecog_probs_norm)

    # Baseline risk score: composite of age, stage, ECOG
    stage_risk = {"I": 0.2, "II": 0.5, "III": 0.8}
    risk = np.zeros(n, dtype=np.float64)
    for i in range(n):
        age_factor = (ages[i] - config.age_range[0]) / (config.age_range[1] - config.age_range[0])
        stage_factor = stage_risk.get(str(disease_stage[i]), 0.5)
        ecog_factor = float(ecog_score[i]) / max(config.ecog_scores)
        risk[i] = 0.3 * age_factor + 0.4 * stage_factor + 0.3 * ecog_factor
        risk[i] += rng.normal(0.0, 0.05)
    risk = np.clip(risk, 0.0, 1.0)

    df = pd.DataFrame({
        "patient_id": np.arange(n),
        "age": ages,
        "sex": sex,
        "weight_kg": weight_kg,
        "disease_stage": disease_stage,
        "biomarker_positive": biomarker_positive,
        "ecog_score": ecog_score,
        "baseline_risk_score": risk,
    })
    return df


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    lo: float,
    hi: float,
    size: int,
) -> np.ndarray:
    """Sample from a normal distribution truncated to [lo, hi]."""
    if std <= 0:
        return np.full(size, np.clip(mean, lo, hi))
    samples = rng.normal(mean, std, size)
    return np.clip(samples, lo, hi)


# ---------------------------------------------------------------------------
# 2. Phase I dose escalation
# ---------------------------------------------------------------------------


@dataclass
class PhaseIConfig:
    design: str = "3+3"  # "3+3", "BOIN", "CRM"
    dose_levels: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    start_level: int = 0
    max_level: int = 5
    dlt_threshold_3p3: float = 0.33
    dlt_threshold_boin: float = 0.30
    n_per_cohort: int = 3
    seed: int = 42


@dataclass
class PhaseIResult:
    mtd_estimate: float
    rp2d: float  # recommended phase 2 dose
    dose_toxicity_curve: pd.DataFrame  # dose, dlt_rate columns
    dose_levels_tested: List[float]
    patients_per_level: List[int]
    dlts_per_level: List[int]
    decision_log: List[str]


def _used_patients(level_data: Dict[int, List[Tuple[int, bool]]]) -> List[int]:
    """Return all patient indices already used across dose levels."""
    used: List[int] = []
    for results in level_data.values():
        used.extend(i for i, _ in results)
    return used


def _make_empty_phase_i(
    dose_levels: List[float], decision_log: List[str]
) -> PhaseIResult:
    """Create a minimal PhaseIResult when no dose escalation could proceed."""
    return PhaseIResult(
        mtd_estimate=dose_levels[0],
        rp2d=dose_levels[0],
        dose_toxicity_curve=pd.DataFrame(
            {"dose": [dose_levels[0]], "dlt_rate": [0.0]}
        ),
        dose_levels_tested=[dose_levels[0]],
        patients_per_level=[0],
        dlts_per_level=[0],
        decision_log=decision_log,
    )


def _build_phase_i_result(
    mtd_estimate: float,
    rp2d: float,
    dose_levels: List[float],
    level_data: Dict[int, List[Tuple[int, bool]]],
    decision_log: List[str],
) -> PhaseIResult:
    """Build a PhaseIResult from accumulated per-level data."""
    levels_tested = sorted(level_data.keys())
    doses_tested = [dose_levels[lvl] for lvl in levels_tested]
    patients_per = [len(level_data[lvl]) for lvl in levels_tested]
    dlts_per = [sum(1 for _, d in level_data[lvl] if d) for lvl in levels_tested]
    dlt_rates = [
        dlts_per[i] / max(patients_per[i], 1)
        for i in range(len(levels_tested))
    ]

    dose_tox = pd.DataFrame({
        "dose": doses_tested,
        "dlt_rate": dlt_rates,
    })

    return PhaseIResult(
        mtd_estimate=mtd_estimate,
        rp2d=rp2d,
        dose_toxicity_curve=dose_tox,
        dose_levels_tested=doses_tested,
        patients_per_level=patients_per,
        dlts_per_level=dlts_per,
        decision_log=decision_log,
    )


def _phase_i_3p3(
    dlt_prob_fn: Callable[[float, pd.Series], float],
    cohort: pd.DataFrame,
    config: PhaseIConfig,
    rng: np.random.Generator,
) -> Tuple[PhaseIResult, Dict[int, List[Tuple[int, bool]]]]:
    """Standard 3+3 dose escalation design.

    Algorithm:
    - Enroll 3 patients at current dose level.
    - If 0/3 DLT: escalate to next level.
    - If >=2/3 DLT: de-escalate; MTD is previous level.
    - If 1/3 DLT: expand to 6 patients at same level.
      - If >=3/6 DLT: de-escalate.
      - If <3/6 DLT: escalate.
    """
    dose_levels = config.dose_levels
    max_level = min(config.max_level, len(dose_levels) - 1)
    decision_log: List[str] = []
    level_data: Dict[int, List[Tuple[int, bool]]] = {}

    for level in range(config.start_level, max_level + 1):
        dose = dose_levels[level]
        decision_log.append(f"Testing dose level {level} ({dose} mg)")

        # Select patients not yet used
        available = [i for i in range(len(cohort)) if i not in _used_patients(level_data)]
        if len(available) < config.n_per_cohort:
            decision_log.append("Not enough patients remaining. Stopping.")
            break
        idx = available[: config.n_per_cohort]

        # Observe DLTs
        cohort_results: List[Tuple[int, bool]] = []
        dlt_count = 0
        for i in idx:
            prob = dlt_prob_fn(dose, cohort.iloc[i])
            dlt = rng.random() < prob
            cohort_results.append((i, dlt))
            if dlt:
                dlt_count += 1
        level_data[level] = cohort_results
        decision_log.append(f"  Level {level}: {dlt_count}/{len(idx)} DLTs")

        # Decision rules
        if dlt_count == 0:
            decision_log.append(f"  0/{len(idx)} DLTs -> Escalate to next level")
            continue
        elif dlt_count >= 2:
            decision_log.append(
                f"  >=2/{len(idx)} DLTs -> De-escalate. MTD is previous level."
            )
            break
        else:
            # Exactly 1 DLT out of 3: expand to 6
            extra_available = [
                i for i in range(len(cohort))
                if i not in _used_patients(level_data)
            ]
            if len(extra_available) < 3:
                decision_log.append("Not enough patients to expand. Stopping.")
                break
            extra_idx = extra_available[:3]
            for i in extra_idx:
                prob = dlt_prob_fn(dose, cohort.iloc[i])
                dlt = rng.random() < prob
                cohort_results.append((i, dlt))
                if dlt:
                    dlt_count += 1
            level_data[level] = cohort_results
            decision_log.append(f"  Expanded: {dlt_count}/6 DLTs at level {level}")

            if dlt_count >= 3:
                decision_log.append(
                    f"  >=3/6 DLTs -> De-escalate. MTD is previous level."
                )
                break
            else:
                decision_log.append(f"  <3/6 DLTs -> Escalate to next level")
                continue

    # Determine MTD: highest level with DLT rate < threshold
    levels_tested = sorted(level_data.keys())
    if not levels_tested:
        return _make_empty_phase_i(dose_levels, decision_log), level_data

    mtd_level = levels_tested[0]
    for lvl in levels_tested:
        results = level_data[lvl]
        n_pat = len(results)
        n_dlt = sum(1 for _, d in results if d)
        rate = n_dlt / max(n_pat, 1)
        if rate < config.dlt_threshold_3p3:
            mtd_level = lvl

    mtd_estimate = dose_levels[mtd_level]
    rp2d = dose_levels[mtd_level]

    decision_log.append(f"MTD estimated at level {mtd_level} ({mtd_estimate} mg)")
    decision_log.append(f"RP2D set at {rp2d} mg")

    return _build_phase_i_result(
        mtd_estimate, rp2d, dose_levels, level_data, decision_log
    ), level_data


def _phase_i_boin(
    dlt_prob_fn: Callable[[float, pd.Series], float],
    cohort: pd.DataFrame,
    config: PhaseIConfig,
    rng: np.random.Generator,
) -> Tuple[PhaseIResult, Dict[int, List[Tuple[int, bool]]]]:
    """BOIN (Bayesian Optimal Interval) design.

    Uses boundary-based dose escalation with target toxicity probability.
    Boundaries: lower = target - 0.056, upper = target + 0.056.
    """
    target = config.dlt_threshold_boin
    lower_bound = target - 0.056
    upper_bound = target + 0.056
    dose_levels = config.dose_levels
    max_level = min(config.max_level, len(dose_levels) - 1)
    decision_log: List[str] = []
    level_data: Dict[int, List[Tuple[int, bool]]] = {}

    current_level = config.start_level
    n_cohorts = 0
    max_cohorts = 2 * (max_level - config.start_level + 1) + 2

    while n_cohorts < max_cohorts:
        n_cohorts += 1
        dose = dose_levels[current_level]
        decision_log.append(
            f"BOIN cohort {n_cohorts}: dose level {current_level} ({dose} mg)"
        )

        available = [i for i in range(len(cohort)) if i not in _used_patients(level_data)]
        n_enroll = min(config.n_per_cohort, len(available))
        if n_enroll == 0:
            decision_log.append("No available patients. Stopping.")
            break

        idx = available[:n_enroll]
        dlt_count = 0
        cohort_results: List[Tuple[int, bool]] = []
        for i in idx:
            prob = dlt_prob_fn(dose, cohort.iloc[i])
            dlt = rng.random() < prob
            cohort_results.append((i, dlt))
            if dlt:
                dlt_count += 1
        level_data[current_level] = cohort_results

        obs_rate = dlt_count / len(idx)
        decision_log.append(
            f"  Observed DLT rate: {obs_rate:.3f} ({dlt_count}/{len(idx)})"
        )

        # BOIN decision boundaries
        if obs_rate <= lower_bound:
            if current_level < max_level:
                decision_log.append(f"  Rate <= {lower_bound:.3f} -> Escalate")
                current_level += 1
            else:
                decision_log.append("At max level, cannot escalate. Stopping.")
                break
        elif obs_rate >= upper_bound:
            if current_level > config.start_level:
                decision_log.append(f"  Rate >= {upper_bound:.3f} -> De-escalate")
                current_level -= 1
            else:
                decision_log.append("At start level, cannot de-escalate. Stopping.")
                break
        else:
            decision_log.append(
                f"  Rate in [{lower_bound:.3f}, {upper_bound:.3f}] -> Stay"
            )
            # Convergence check: if we have accumulated enough data at this level
            if current_level in level_data and len(level_data[current_level]) >= 6:
                decision_log.append("Sufficient data at current level. Stopping.")
                break

    levels_tested = sorted(level_data.keys())
    if not levels_tested:
        return _make_empty_phase_i(dose_levels, decision_log), level_data

    mtd_level = levels_tested[0]
    for lvl in levels_tested:
        results = level_data[lvl]
        n_dlt = sum(1 for _, d in results if d)
        rate = n_dlt / max(len(results), 1)
        if rate <= upper_bound:
            mtd_level = lvl

    mtd_estimate = dose_levels[mtd_level]
    rp2d = dose_levels[mtd_level]

    decision_log.append(f"MTD estimated at level {mtd_level} ({mtd_estimate} mg)")
    decision_log.append(f"RP2D set at {rp2d} mg")

    return _build_phase_i_result(
        mtd_estimate, rp2d, dose_levels, level_data, decision_log
    ), level_data


def _phase_i_crm(
    dlt_prob_fn: Callable[[float, pd.Series], float],
    cohort: pd.DataFrame,
    config: PhaseIConfig,
    rng: np.random.Generator,
) -> Tuple[PhaseIResult, Dict[int, List[Tuple[int, bool]]]]:
    """Continual Reassessment Method (CRM) with skeleton probabilities.

    Uses a simple skeleton with increasing prior DLT probabilities.
    Posterior is updated with Beta(alpha_prior + DLTs, beta_prior + non-DLTs).
    The dose level with posterior mean closest to the target toxicity is selected.
    """
    dose_levels = config.dose_levels
    max_level = min(config.max_level, len(dose_levels) - 1)
    target = config.dlt_threshold_boin
    decision_log: List[str] = []
    level_data: Dict[int, List[Tuple[int, bool]]] = {}

    # Skeleton: prior DLT probabilities (monotonically increasing with dose)
    n_levels = max_level + 1
    skeleton = np.linspace(0.05, 0.55, n_levels)

    # Beta(1, 1) prior for each level (uniform)
    alpha_prior = np.ones(n_levels, dtype=np.float64)
    beta_prior = np.ones(n_levels, dtype=np.float64)

    current_level = config.start_level
    n_patients_total = 0
    max_patients = n_levels * config.n_per_cohort * 3

    while n_patients_total < max_patients:
        dose = dose_levels[current_level]
        decision_log.append(
            f"CRM: enrolling at level {current_level} ({dose} mg), "
            f"skeleton prob={skeleton[current_level]:.3f}"
        )

        available = [
            i for i in range(len(cohort))
            if i not in _used_patients(level_data)
        ]
        n_enroll = min(config.n_per_cohort, len(available),
                       max_patients - n_patients_total)
        if n_enroll == 0:
            decision_log.append("No available patients. Stopping.")
            break

        idx = available[:n_enroll]
        dlt_count = 0
        cohort_results: List[Tuple[int, bool]] = []
        for i in idx:
            prob = dlt_prob_fn(dose, cohort.iloc[i])
            dlt = rng.random() < prob
            cohort_results.append((i, dlt))
            if dlt:
                dlt_count += 1
                alpha_prior[current_level] += 1.0
            else:
                beta_prior[current_level] += 1.0

        if current_level in level_data:
            level_data[current_level].extend(cohort_results)
        else:
            level_data[current_level] = list(cohort_results)

        n_patients_total += n_enroll
        decision_log.append(
            f"  {dlt_count}/{n_enroll} DLTs, total at level: "
            f"{len(level_data[current_level])}"
        )

        # Posterior means
        posterior_means = alpha_prior / (alpha_prior + beta_prior)

        # Find level whose posterior mean is closest to target
        distances = np.abs(posterior_means - target)
        eligible = [
            lvl for lvl in range(n_levels)
            if lvl in level_data and len(level_data[lvl]) >= config.n_per_cohort
        ]
        if eligible:
            eligible_dists = [distances[lvl] for lvl in eligible]
            best_idx = int(np.argmin(eligible_dists))
            new_level = eligible[best_idx]
        else:
            new_level = int(np.argmin(distances))

        if new_level != current_level:
            direction = "escalate" if new_level > current_level else "de-escalate"
            decision_log.append(f"  CRM recommends: {direction} to level {new_level}")
        else:
            decision_log.append(f"  CRM recommends: stay at level {new_level}")

        # Convergence check
        if (new_level == current_level
                and current_level in level_data
                and len(level_data[current_level]) >= 2 * config.n_per_cohort):
            decision_log.append("Converged at current level. Stopping.")
            break

        current_level = max(0, min(new_level, max_level))

    levels_tested = sorted(level_data.keys())
    if not levels_tested:
        return _make_empty_phase_i(dose_levels, decision_log), level_data

    # MTD: level with posterior mean closest to target
    posterior_means = alpha_prior / (alpha_prior + beta_prior)
    distances = np.abs(posterior_means - target)
    mtd_level = int(np.argmin(distances))
    mtd_level = min(mtd_level, max_level)

    mtd_estimate = dose_levels[mtd_level]
    rp2d = dose_levels[mtd_level]

    decision_log.append(f"MTD estimated at level {mtd_level} ({mtd_estimate} mg)")
    decision_log.append(f"RP2D set at {rp2d} mg")

    return _build_phase_i_result(
        mtd_estimate, rp2d, dose_levels, level_data, decision_log
    ), level_data


def simulate_phase_i(
    dlt_prob_fn: Callable[[float, pd.Series], float],
    cohort: pd.DataFrame,
    config: PhaseIConfig | None = None,
) -> PhaseIResult:
    """Simulate Phase I dose escalation.

    Parameters
    ----------
    dlt_prob_fn : callable
        Takes (dose: float, patient_row: pd.Series) and returns DLT probability.
    cohort : pd.DataFrame
        Virtual patient cohort from ``generate_cohort``.
    config : PhaseIConfig, optional
        Phase I configuration.

    Returns
    -------
    PhaseIResult
    """
    if config is None:
        config = PhaseIConfig()
    rng = np.random.default_rng(config.seed)

    design = config.design.lower().strip()
    if design == "3+3":
        result, _ = _phase_i_3p3(dlt_prob_fn, cohort, config, rng)
    elif design == "boin":
        result, _ = _phase_i_boin(dlt_prob_fn, cohort, config, rng)
    elif design == "crm":
        result, _ = _phase_i_crm(dlt_prob_fn, cohort, config, rng)
    else:
        raise ValueError(
            f"Unknown Phase I design: {config.design!r}. "
            f"Choose from '3+3', 'BOIN', 'CRM'."
        )

    return result


# ---------------------------------------------------------------------------
# 3. Phase II efficacy simulation
# ---------------------------------------------------------------------------


@dataclass
class PhaseIIConfig:
    n_arm_treatment: int = 50
    n_arm_control: int = 25
    primary_endpoint: str = "ORR"  # "ORR", "DCR", "PFS_6mo"
    orr_threshold: float = 0.20  # null hypothesis
    alpha: float = 0.05
    seed: int = 42


@dataclass
class PhaseIIResult:
    treatment_arm: Dict[str, float]  # endpoint values
    control_arm: Dict[str, float]
    p_value: float
    statistically_significant: bool
    power_estimate: float  # post-hoc
    biomarker_subgroup: Dict[str, float]  # biomarker+ subgroup results
    km_data_treatment: pd.DataFrame  # time, event columns for KM curve
    km_data_control: pd.DataFrame


def _generate_km_data(
    rng: np.random.Generator,
    n_patients: int,
    monthly_rate: float,
    follow_up_months: float = 12.0,
    dropout_rate: float = 0.02,
) -> pd.DataFrame:
    """Generate (time, event) pairs for Kaplan-Meier plotting.

    Uses an exponential survival model with given monthly event rate.
    Patients are censored at the end of follow-up or by random dropout.
    """
    times = rng.exponential(1.0 / max(monthly_rate, 1e-6), size=n_patients)
    times = np.clip(times, 0.01, follow_up_months * 3)

    # Administrative censoring at follow-up
    admin_censor = rng.random(n_patients) > (follow_up_months / times.max()) if times.max() > 0 else np.ones(n_patients, dtype=bool)
    events = np.ones(n_patients, dtype=np.float64)  # 1 = event occurred

    # Random dropout censoring
    dropout_times = rng.exponential(1.0 / max(dropout_rate, 1e-6), size=n_patients)
    censored_by_dropout = dropout_times < times
    events = np.where(censored_by_dropout, 0.0, events)
    times = np.where(censored_by_dropout, dropout_times, times)

    # Administrative censoring
    times = np.minimum(times, follow_up_months)
    events = np.where(times >= follow_up_months, 0.0, events)

    return pd.DataFrame({"time": times, "event": events})


def simulate_phase_ii(
    efficacy_fn: Callable[[float, pd.Series], Dict[str, float]],
    cohort: pd.DataFrame,
    rp2d: float,
    soc_efficacy: Dict[str, float],
    config: PhaseIIConfig | None = None,
) -> PhaseIIResult:
    """Simulate Phase II efficacy trial.

    Parameters
    ----------
    efficacy_fn : callable
        Takes (dose: float, patient_row: pd.Series) and returns a dict of
        efficacy metrics including at minimum: "ORR", "DCR", "PFS_rate_6mo".
    cohort : pd.DataFrame
        Virtual patient cohort.
    rp2d : float
        Recommended Phase 2 dose from Phase I.
    soc_efficacy : dict
        Standard of care baseline efficacy (e.g. {"ORR": 0.15, "DCR": 0.40, "PFS_rate_6mo": 0.30}).
    config : PhaseIIConfig, optional
        Phase II configuration.

    Returns
    -------
    PhaseIIResult
    """
    if config is None:
        config = PhaseIIConfig()
    rng = np.random.default_rng(config.seed)

    n_trt = config.n_arm_treatment
    n_ctrl = config.n_arm_control

    # Randomly assign patients to arms (without replacement from pool)
    shuffled = rng.permutation(len(cohort))
    trt_idx = shuffled[:n_trt]
    ctrl_idx = shuffled[n_trt : n_trt + n_ctrl]

    # Treatment arm: get efficacy for each patient
    trt_orr_responses: List[float] = []
    trt_dcr_responses: List[float] = []
    trt_pfs_rates: List[float] = []
    for i in trt_idx:
        metrics = efficacy_fn(rp2d, cohort.iloc[i])
        trt_orr_responses.append(metrics.get("ORR", 0.0))
        trt_dcr_responses.append(metrics.get("DCR", 0.0))
        trt_pfs_rates.append(metrics.get("PFS_rate_6mo", 0.0))

    # Convert probability to binary response for each patient
    trt_orr_binary = np.array([rng.random() < p for p in trt_orr_responses], dtype=np.float64)
    trt_dcr_binary = np.array([rng.random() < p for p in trt_dcr_responses], dtype=np.float64)

    # Control arm: use SOC efficacy with patient-level variability
    ctrl_orr_responses = []
    ctrl_dcr_responses = []
    ctrl_pfs_rates = []
    for i in ctrl_idx:
        patient = cohort.iloc[i]
        risk = float(patient.get("baseline_risk_score", 0.5))
        noise = rng.normal(0.0, 0.03)
        ctrl_orr_responses.append(
            np.clip(soc_efficacy.get("ORR", 0.15) * (1.0 - 0.3 * risk) + noise, 0.0, 1.0)
        )
        ctrl_dcr_responses.append(
            np.clip(soc_efficacy.get("DCR", 0.40) * (1.0 - 0.2 * risk) + noise, 0.0, 1.0)
        )
        ctrl_pfs_rates.append(
            np.clip(soc_efficacy.get("PFS_rate_6mo", 0.30) * (1.0 - 0.25 * risk) + noise, 0.0, 1.0)
        )

    ctrl_orr_binary = np.array([rng.random() < p for p in ctrl_orr_responses], dtype=np.float64)
    ctrl_dcr_binary = np.array([rng.random() < p for p in ctrl_dcr_responses], dtype=np.float64)

    # Summary statistics
    trt_arm = {
        "ORR": float(np.mean(trt_orr_binary)),
        "DCR": float(np.mean(trt_dcr_binary)),
        "PFS_6mo": float(np.mean(trt_pfs_rates)),
        "n": float(n_trt),
    }
    ctrl_arm = {
        "ORR": float(np.mean(ctrl_orr_binary)),
        "DCR": float(np.mean(ctrl_dcr_binary)),
        "PFS_6mo": float(np.mean(ctrl_pfs_rates)),
        "n": float(n_ctrl),
    }

    # Statistical test: Fisher exact or chi-squared for ORR
    if config.primary_endpoint == "ORR":
        trt_pos = int(np.sum(trt_orr_binary))
        trt_neg = n_trt - trt_pos
        ctrl_pos = int(np.sum(ctrl_orr_binary))
        ctrl_neg = n_ctrl - ctrl_pos
        contingency = np.array([[trt_pos, trt_neg], [ctrl_pos, ctrl_neg]])
        if contingency.min() >= 5:
            _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
        else:
            _, p_value = stats.fisher_exact(contingency)
    elif config.primary_endpoint == "DCR":
        trt_pos = int(np.sum(trt_dcr_binary))
        trt_neg = n_trt - trt_pos
        ctrl_pos = int(np.sum(ctrl_dcr_binary))
        ctrl_neg = n_ctrl - ctrl_pos
        contingency = np.array([[trt_pos, trt_neg], [ctrl_pos, ctrl_neg]])
        if contingency.min() >= 5:
            _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
        else:
            _, p_value = stats.fisher_exact(contingency)
    else:
        # PFS_6mo: two-sample t-test on PFS rates
        _, p_value = stats.ttest_ind(trt_pfs_rates, ctrl_pfs_rates)

    statistically_significant = p_value < config.alpha

    # Post-hoc power estimate
    if config.primary_endpoint == "ORR":
        p1 = trt_arm["ORR"]
        p0 = ctrl_arm["ORR"]
        n1 = n_trt
        n2 = n_ctrl
        p_pooled = (p1 * n1 + p0 * n2) / (n1 + n2)
        if p_pooled > 0 and p_pooled < 1:
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
            if se > 0:
                z_stat = abs(p1 - p0) / se
                # Approximate power using normal distribution
                from scipy.stats import norm as _norm
                alpha_z = _norm.ppf(1 - config.alpha / 2)
                power = float(_norm.cdf(z_stat - alpha_z))
            else:
                power = 0.0
        else:
            power = 0.0
    else:
        # Approximate power from t-test
        n1 = n_trt
        n2 = n_ctrl
        d1 = trt_pfs_rates
        d2 = ctrl_pfs_rates
        pooled_std = np.sqrt(
            (np.var(d1, ddof=1) * (n1 - 1) + np.var(d2, ddof=1) * (n2 - 1))
            / (n1 + n2 - 2)
        )
        if pooled_std > 0:
            effect_size = abs(np.mean(d1) - np.mean(d2)) / pooled_std
            from scipy.stats import nct as _nct
            nc = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
            df = n1 + n2 - 2
            t_crit = stats.t.ppf(1 - config.alpha / 2, df)
            power = float(1 - _nct.cdf(t_crit, df, nc) + _nct.cdf(-t_crit, df, nc))
        else:
            power = 0.0

    power_estimate = float(np.clip(power, 0.0, 1.0))

    # Biomarker subgroup analysis (treatment arm only)
    bio_subgroup: Dict[str, float] = {}
    trt_patients = cohort.iloc[trt_idx].copy()
    trt_patients["orr_response"] = trt_orr_binary
    if "biomarker_positive" in trt_patients.columns:
        bio_pos = trt_patients[trt_patients["biomarker_positive"] == True]
        bio_neg = trt_patients[trt_patients["biomarker_positive"] == False]
        bio_subgroup["ORR_biomarker_positive"] = float(bio_pos["orr_response"].mean()) if len(bio_pos) > 0 else 0.0
        bio_subgroup["ORR_biomarker_negative"] = float(bio_neg["orr_response"].mean()) if len(bio_neg) > 0 else 0.0
        bio_subgroup["n_biomarker_positive"] = float(len(bio_pos))
        bio_subgroup["n_biomarker_negative"] = float(len(bio_neg))
        # Odds ratio for biomarker
        if len(bio_pos) > 0 and len(bio_neg) > 0:
            p1 = bio_subgroup["ORR_biomarker_positive"]
            p0 = bio_subgroup["ORR_biomarker_negative"]
            if p0 > 0 and p0 < 1 and p1 > 0 and p1 < 1:
                bio_subgroup["biomarker_OR"] = float(
                    (p1 / (1 - p1)) / (p0 / (1 - p0))
                )
            else:
                bio_subgroup["biomarker_OR"] = float("nan")

    # KM data
    trt_mean_pfs = float(np.mean(trt_pfs_rates))
    ctrl_mean_pfs = float(np.mean(ctrl_pfs_rates))
    # Convert PFS rate to monthly event rate for exponential model
    trt_event_rate = max(-np.log(max(trt_mean_pfs, 1e-6)) / 6.0, 0.01)
    ctrl_event_rate = max(-np.log(max(ctrl_mean_pfs, 1e-6)) / 6.0, 0.01)

    km_data_treatment = _generate_km_data(rng, n_trt, trt_event_rate, follow_up_months=12.0)
    km_data_control = _generate_km_data(rng, n_ctrl, ctrl_event_rate, follow_up_months=12.0)

    return PhaseIIResult(
        treatment_arm=trt_arm,
        control_arm=ctrl_arm,
        p_value=float(p_value),
        statistically_significant=statistically_significant,
        power_estimate=power_estimate,
        biomarker_subgroup=bio_subgroup,
        km_data_treatment=km_data_treatment,
        km_data_control=km_data_control,
    )


# ---------------------------------------------------------------------------
# 4. Phase III comparative simulation
# ---------------------------------------------------------------------------


@dataclass
class PhaseIIIConfig:
    n_arm_treatment: int = 300
    n_arm_control: int = 300
    stratification_factors: List[str] = field(
        default_factory=lambda: ["disease_stage", "biomarker_positive"]
    )
    alpha: float = 0.05
    seed: int = 42


@dataclass
class PhaseIIIResult:
    hazard_ratio: float
    hr_ci_lower: float
    hr_ci_upper: float
    p_value: float
    significant: bool
    subgroup_analysis: pd.DataFrame  # subgroup, n, hr, ci_lower, ci_upper, p_value
    km_data_treatment: pd.DataFrame
    km_data_control: pd.DataFrame
    median_survival_treatment: float
    median_survival_control: float


def _log_rank_test(
    time_trt: np.ndarray,
    event_trt: np.ndarray,
    time_ctrl: np.ndarray,
    event_ctrl: np.ndarray,
) -> float:
    """Perform a simplified log-rank test and return the p-value.

    Uses the Mantel-Haenszel approach for the observed vs expected events.
    """
    # Combine all times and sort
    all_times = np.concatenate([time_trt, time_ctrl])
    all_events = np.concatenate([event_trt, np.zeros(len(time_ctrl))])
    n_trt = len(time_trt)

    # Sort by time
    order = np.argsort(all_times)
    all_times = all_times[order]
    all_events = all_events[order]
    group = np.concatenate([np.ones(n_trt), np.zeros(len(time_ctrl))])[order]

    # Compute observed and expected events for treatment group
    O = 0.0  # observed events in treatment
    E = 0.0  # expected events in treatment
    V = 0.0  # variance

    n_at_risk_trt = float(n_trt)
    n_at_risk_ctrl = float(len(time_ctrl))
    prev_time = -1.0

    for t in range(len(all_times)):
        if all_times[t] != prev_time:
            n_total = n_at_risk_trt + n_at_risk_ctrl
            if n_total > 0 and all_events[t] > 0:
                e_trt = n_at_risk_trt * all_events[t] / n_total
                O += 1.0 if group[t] == 1 else 0.0
                E += e_trt
                if 0 < n_at_risk_trt < n_total:
                    V += e_trt * (1 - n_at_risk_trt / n_total) * (
                        n_total - all_events[t]
                    ) / (n_total - 1)
        if group[t] == 1:
            n_at_risk_trt -= 1.0
        else:
            n_at_risk_ctrl -= 1.0
        prev_time = all_times[t]

    if V <= 0:
        return 1.0

    chi2 = (O - E) ** 2 / V
    p_value = float(1.0 - stats.chi2.cdf(chi2, df=1))
    return p_value


def _estimate_hr(
    time_trt: np.ndarray,
    event_trt: np.ndarray,
    time_ctrl: np.ndarray,
    event_ctrl: np.ndarray,
) -> Tuple[float, float, float]:
    """Estimate hazard ratio using Mantel-Haenszel method.

    Returns (hr, ci_lower, ci_upper).
    """
    # Combine all times
    all_times = np.concatenate([time_trt, time_ctrl])
    all_events = np.concatenate([event_trt, np.zeros(len(time_ctrl))])
    n_trt = len(time_trt)
    n_ctrl = len(time_ctrl)

    order = np.argsort(all_times)
    all_times = all_times[order]
    all_events = all_events[order]
    group = np.concatenate([np.ones(n_trt), np.zeros(n_ctrl)])[order]

    # Mantel-Haenszel estimator
    O_trt = 0.0  # observed events in treatment
    E_trt = 0.0  # expected events in treatment
    V_trt = 0.0  # variance

    n_at_risk_trt = float(n_trt)
    n_at_risk_ctrl = float(n_ctrl)
    prev_time = -1.0

    for t in range(len(all_times)):
        if all_times[t] != prev_time:
            n_total = n_at_risk_trt + n_at_risk_ctrl
            if n_total > 0 and all_events[t] > 0:
                e_trt = n_at_risk_trt * all_events[t] / n_total
                O_trt += 1.0 if group[t] == 1 else 0.0
                E_trt += e_trt
                if 0 < n_at_risk_trt < n_total:
                    V_trt += e_trt * (1 - n_at_risk_trt / n_total) * (
                        n_total - all_events[t]
                    ) / (n_total - 1)
        if group[t] == 1:
            n_at_risk_trt -= 1.0
        else:
            n_at_risk_ctrl -= 1.0
        prev_time = all_times[t]

    if E_trt <= 0 or V_trt <= 0:
        return 1.0, 0.5, 2.0

    # HR = O_trt / E_trt (hazard ratio of treatment vs control)
    hr = O_trt / E_trt
    hr = max(hr, 0.01)  # floor to avoid log(0)

    # 95% CI using log(HR) with SE from Mantel-Haenszel variance
    log_hr = np.log(hr)
    se_log_hr = 1.0 / np.sqrt(max(O_trt, 1))  # approximate SE

    # More accurate SE from delta method
    if E_trt > 0 and O_trt > 0:
        se_log_hr = np.sqrt(max(V_trt, 1e-6)) / max(E_trt, 1e-6)

    z = stats.norm.ppf(1 - 0.05 / 2)  # 1.96
    ci_lower = float(np.exp(log_hr - z * se_log_hr))
    ci_upper = float(np.exp(log_hr + z * se_log_hr))

    return float(hr), float(ci_lower), float(ci_upper)


def _subgroup_hr(
    time_trt: np.ndarray,
    event_trt: np.ndarray,
    time_ctrl: np.ndarray,
    event_ctrl: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute HR and log-rank p-value for a subgroup."""
    hr, ci_lo, ci_hi = _estimate_hr(time_trt, event_trt, time_ctrl, event_ctrl)
    p_val = _log_rank_test(time_trt, event_trt, time_ctrl, event_ctrl)
    return hr, ci_lo, ci_hi, p_val


def simulate_phase_iii(
    survival_fn: Callable[[float, pd.Series], float],
    cohort: pd.DataFrame,
    rp2d: float,
    soc_median_survival: float,
    config: PhaseIIIConfig | None = None,
) -> PhaseIIIResult:
    """Simulate Phase III comparative trial.

    Parameters
    ----------
    survival_fn : callable
        Takes (dose: float, patient_row: pd.Series) and returns survival
        time in months.
    cohort : pd.DataFrame
        Virtual patient cohort.
    rp2d : float
        Recommended Phase 2 dose from Phase I.
    soc_median_survival : float
        Median survival time (months) for standard of care.
    config : PhaseIIIConfig, optional
        Phase III configuration.

    Returns
    -------
    PhaseIIIResult
    """
    if config is None:
        config = PhaseIIIConfig()
    rng = np.random.default_rng(config.seed)

    n_trt = config.n_arm_treatment
    n_ctrl = config.n_arm_control
    follow_up = 24.0  # months

    # Randomly assign patients (sample with replacement if cohort is small)
    n_total_needed = n_trt + n_ctrl
    if len(cohort) >= n_total_needed:
        shuffled = rng.permutation(len(cohort))
        trt_idx = shuffled[:n_trt]
        ctrl_idx = shuffled[n_trt : n_trt + n_ctrl]
    else:
        # Sample with replacement to fill arms
        trt_idx = rng.choice(len(cohort), size=n_trt, replace=True)
        ctrl_idx = rng.choice(len(cohort), size=n_ctrl, replace=True)

    # Treatment arm: get survival time from model
    trt_survival_times = np.zeros(n_trt, dtype=np.float64)
    trt_events = np.ones(n_trt, dtype=np.float64)
    for j, i in enumerate(trt_idx):
        t = survival_fn(rp2d, cohort.iloc[i])
        trt_survival_times[j] = t
        # Censor at follow-up
        if t > follow_up:
            trt_survival_times[j] = follow_up
            trt_events[j] = 0.0

    # Control arm: exponential survival with SOC median
    soc_rate = np.log(2) / max(soc_median_survival, 0.1)
    ctrl_survival_times = rng.exponential(1.0 / soc_rate, size=n_ctrl)
    ctrl_events = np.ones(n_ctrl, dtype=np.float64)
    for j in range(n_ctrl):
        patient = cohort.iloc[ctrl_idx[j]]
        risk = float(patient.get("baseline_risk_score", 0.5))
        # Adjust by patient risk: higher risk -> shorter survival
        rate_adj = soc_rate * (1.0 + 0.5 * risk)
        ctrl_survival_times[j] = rng.exponential(1.0 / rate_adj)
        if ctrl_survival_times[j] > follow_up:
            ctrl_survival_times[j] = follow_up
            ctrl_events[j] = 0.0

    # HR estimation
    hr, hr_ci_lower, hr_ci_upper = _estimate_hr(
        trt_survival_times, trt_events,
        ctrl_survival_times, ctrl_events,
    )

    # Log-rank test
    p_value = _log_rank_test(
        trt_survival_times, trt_events,
        ctrl_survival_times, ctrl_events,
    )

    significant = p_value < config.alpha

    # Subgroup analysis
    subgroup_rows: List[Dict[str, Any]] = []
    trt_patients = cohort.iloc[trt_idx].copy()
    ctrl_patients = cohort.iloc[ctrl_idx].copy()
    trt_patients["survival_time"] = trt_survival_times
    trt_patients["event"] = trt_events
    ctrl_patients["survival_time"] = ctrl_survival_times
    ctrl_patients["event"] = ctrl_events

    # Overall
    subgroup_rows.append({
        "subgroup": "Overall",
        "n_trt": n_trt,
        "n_ctrl": n_ctrl,
        "hr": hr,
        "ci_lower": hr_ci_lower,
        "ci_upper": hr_ci_upper,
        "p_value": p_value,
    })

    # Stratification by each factor
    for factor in config.stratification_factors:
        if factor not in cohort.columns:
            continue
        groups = cohort[factor].unique()
        for g in sorted(groups, key=str):
            if pd.isna(g):
                continue
            mask_trt = trt_patients[factor] == g
            mask_ctrl = ctrl_patients[factor] == g
            n_sub_trt = int(mask_trt.sum())
            n_sub_ctrl = int(mask_ctrl.sum())
            if n_sub_trt < 5 or n_sub_ctrl < 5:
                continue
            sub_hr, sub_ci_lo, sub_ci_hi, sub_p = _subgroup_hr(
                trt_patients.loc[mask_trt, "survival_time"].values,
                trt_patients.loc[mask_trt, "event"].values,
                ctrl_patients.loc[mask_ctrl, "survival_time"].values,
                ctrl_patients.loc[mask_ctrl, "event"].values,
            )
            subgroup_rows.append({
                "subgroup": f"{factor}={g}",
                "n_trt": n_sub_trt,
                "n_ctrl": n_sub_ctrl,
                "hr": sub_hr,
                "ci_lower": sub_ci_lo,
                "ci_upper": sub_ci_hi,
                "p_value": sub_p,
            })

    subgroup_df = pd.DataFrame(subgroup_rows)

    # KM data
    km_data_treatment = pd.DataFrame({
        "time": trt_survival_times,
        "event": trt_events,
    })
    km_data_control = pd.DataFrame({
        "time": ctrl_survival_times,
        "event": ctrl_events,
    })

    # Median survival (Kaplan-Meier median estimate)
    median_survival_treatment = float(np.median(trt_survival_times[trt_events > 0])) if np.any(trt_events > 0) else float(np.max(trt_survival_times))
    median_survival_control = float(np.median(ctrl_survival_times[ctrl_events > 0])) if np.any(ctrl_events > 0) else float(np.max(ctrl_survival_times))

    return PhaseIIIResult(
        hazard_ratio=hr,
        hr_ci_lower=hr_ci_lower,
        hr_ci_upper=hr_ci_upper,
        p_value=p_value,
        significant=significant,
        subgroup_analysis=subgroup_df,
        km_data_treatment=km_data_treatment,
        km_data_control=km_data_control,
        median_survival_treatment=median_survival_treatment,
        median_survival_control=median_survival_control,
    )


# ---------------------------------------------------------------------------
# 5. Default model functions (for testing / standalone use)
# ---------------------------------------------------------------------------


def default_dlt_prob(dose: float, patient_row: pd.Series) -> float:
    """Default DLT probability function using a sigmoid dose-toxicity model.

    The probability is adjusted by the patient's baseline_risk_score.
    DLT probability = sigmoid((log(dose) - mu) / sigma) * (1 + 0.3 * risk)

    Parameters
    ----------
    dose : float
        Dose level in mg.
    patient_row : pd.Series
        Patient row from the cohort DataFrame.

    Returns
    -------
    float
        DLT probability in [0, 1].
    """
    risk = float(patient_row.get("baseline_risk_score", 0.5))
    log_dose = np.log(max(dose, 0.01))
    mu = 1.5  # center of sigmoid on log-dose scale
    sigma = 0.8  # width
    base_prob = 1.0 / (1.0 + np.exp(-(log_dose - mu) / sigma))
    # Adjust by patient risk
    adjusted = base_prob * (1.0 + 0.3 * risk)
    return float(np.clip(adjusted, 0.0, 0.95))


def default_efficacy_fn(dose: float, patient_row: pd.Series) -> Dict[str, float]:
    """Default efficacy function using sigmoid dose-response.

    Returns a dict with keys: "ORR", "DCR", "PFS_rate_6mo".

    Parameters
    ----------
    dose : float
        Dose level in mg.
    patient_row : pd.Series
        Patient row from the cohort DataFrame.

    Returns
    -------
    dict
        Efficacy metrics.
    """
    risk = float(patient_row.get("baseline_risk_score", 0.5))
    biomarker = bool(patient_row.get("biomarker_positive", False))

    log_dose = np.log(max(dose, 0.01))
    # ORR: dose-response curve
    orr_base = 0.1 + 0.45 / (1.0 + np.exp(-(log_dose - 1.2) / 0.6))
    orr = orr_base * (1.0 - 0.15 * risk)
    if biomarker:
        orr *= 1.3  # biomarker-positive patients respond better

    # DCR: higher than ORR
    dcr = min(orr * 1.6 + 0.15, 0.95) * (1.0 - 0.1 * risk)

    # PFS rate at 6 months
    pfs_6mo = min(orr * 1.4 + 0.10, 0.85) * (1.0 - 0.2 * risk)
    if biomarker:
        pfs_6mo *= 1.15

    return {
        "ORR": float(np.clip(orr, 0.0, 1.0)),
        "DCR": float(np.clip(dcr, 0.0, 1.0)),
        "PFS_rate_6mo": float(np.clip(pfs_6mo, 0.0, 1.0)),
    }


def default_survival_fn(dose: float, patient_row: pd.Series) -> float:
    """Default survival function using exponential model with dose-dependent rate.

    Returns survival time in months. Higher dose and biomarker-positive status
    lead to longer survival.

    Parameters
    ----------
    dose : float
        Dose level in mg.
    patient_row : pd.Series
        Patient row from the cohort DataFrame.

    Returns
    -------
    float
        Survival time in months.
    """
    risk = float(patient_row.get("baseline_risk_score", 0.5))
    biomarker = bool(patient_row.get("biomarker_positive", False))
    age = float(patient_row.get("age", 55.0))

    # Baseline monthly hazard rate (approximating SOC with median ~10 months)
    base_rate = np.log(2) / 10.0

    # Dose effect: higher dose reduces hazard
    log_dose = np.log(max(dose, 0.01))
    dose_reduction = 0.3 / (1.0 + np.exp(-(log_dose - 1.5) / 0.7))

    # Patient factors
    risk_adjustment = 1.0 + 0.4 * risk
    age_adjustment = 1.0 + 0.005 * (age - 55.0)  # slight age effect

    # Biomarker benefit
    bio_factor = 0.7 if biomarker else 1.0

    # Final hazard rate
    hazard_rate = base_rate * risk_adjustment * age_adjustment * bio_factor * (1.0 - dose_reduction)
    hazard_rate = max(hazard_rate, 0.01)

    # Sample survival time from exponential
    # Use a local rng seeded by patient_id for reproducibility
    pid = int(patient_row.get("patient_id", 0))
    local_rng = np.random.default_rng(pid + int(dose * 1000))
    survival_time = local_rng.exponential(1.0 / hazard_rate)

    return float(survival_time)


# ---------------------------------------------------------------------------
# 6. Trial report generation
# ---------------------------------------------------------------------------


def generate_trial_report(
    phase_i: PhaseIResult | None = None,
    phase_ii: PhaseIIResult | None = None,
    phase_iii: PhaseIIIResult | None = None,
    cohort_summary: pd.DataFrame | None = None,
    drug_name: str = "circRNA-X",
) -> str:
    """Generate a CSR-style Markdown report string summarizing trial results.

    Parameters
    ----------
    phase_i : PhaseIResult, optional
        Results from Phase I dose escalation.
    phase_ii : PhaseIIResult, optional
        Results from Phase II efficacy trial.
    phase_iii : PhaseIIIResult, optional
        Results from Phase III comparative trial.
    cohort_summary : pd.DataFrame, optional
        Summary of the patient cohort used.
    drug_name : str
        Name of the investigational drug.

    Returns
    -------
    str
        Markdown report string.
    """
    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add(f"# Clinical Study Report: {drug_name}")
    add()
    add("**Virtual Clinical Trial Simulation Report**")
    add()

    # Table of contents
    add("## Table of Contents")
    add()
    add("1. [Study Overview](#study-overview)")
    if cohort_summary is not None:
        add("2. [Patient Cohort](#patient-cohort)")
    if phase_i is not None:
        add("3. [Phase I: Dose Escalation](#phase-i-dose-escalation)")
    if phase_ii is not None:
        add("4. [Phase II: Efficacy](#phase-ii-efficacy)")
    if phase_iii is not None:
        add("5. [Phase III: Comparative](#phase-iii-comparative)")
    add()

    # Study overview
    add("## Study Overview")
    add()
    add(f"- **Drug**: {drug_name}")
    add(f"- **Study Type**: Virtual Clinical Trial Simulation")
    phases = []
    if phase_i is not None:
        phases.append("Phase I")
    if phase_ii is not None:
        phases.append("Phase II")
    if phase_iii is not None:
        phases.append("Phase III")
    add(f"- **Phases Simulated**: {', '.join(phases) if phases else 'None'}")
    add()

    # Cohort summary
    if cohort_summary is not None:
        add("## Patient Cohort")
        add()
        add(f"- **Total Patients**: {len(cohort_summary)}")
        add(f"- **Age (mean +/- sd)**: {cohort_summary['age'].mean():.1f} +/- {cohort_summary['age'].std():.1f}")
        add(f"- **Sex (% Female)**: {100 * (cohort_summary['sex'] == 'F').mean():.1f}%")
        add(f"- **Weight (mean +/- sd)**: {cohort_summary['weight_kg'].mean():.1f} +/- {cohort_summary['weight_kg'].std():.1f} kg")
        add(f"- **Mean Baseline Risk Score**: {cohort_summary['baseline_risk_score'].mean():.3f}")
        if "disease_stage" in cohort_summary.columns:
            add()
            add("| Disease Stage | Count | Percentage |")
            add("|---|---|---|")
            stage_counts = cohort_summary["disease_stage"].value_counts().sort_index()
            for stage, count in stage_counts.items():
                add(f"| {stage} | {count} | {100 * count / len(cohort_summary):.1f}% |")
        if "biomarker_positive" in cohort_summary.columns:
            n_bio = int(cohort_summary["biomarker_positive"].sum())
            add()
            add(f"- **Biomarker Positive**: {n_bio} ({100 * n_bio / len(cohort_summary):.1f}%)")
        add()

    # Phase I
    if phase_i is not None:
        add("## Phase I: Dose Escalation")
        add()
        add(f"- **MTD Estimate**: {phase_i.mtd_estimate:.2f} mg")
        add(f"- **RP2D**: {phase_i.rp2d:.2f} mg")
        add(f"- **Dose Levels Tested**: {len(phase_i.dose_levels_tested)}")
        add()
        add("### Dose-Toxicity Summary")
        add()
        add("| Dose (mg) | Patients | DLTs | DLT Rate |")
        add("|---|---|---|---|")
        for j in range(len(phase_i.dose_levels_tested)):
            dose = phase_i.dose_levels_tested[j]
            n_pat = phase_i.patients_per_level[j]
            n_dlt = phase_i.dlts_per_level[j]
            rate = n_dlt / max(n_pat, 1)
            add(f"| {dose:.2f} | {n_pat} | {n_dlt} | {rate:.2f} |")
        add()
        add("### Decision Log")
        add()
        add("```")
        for entry in phase_i.decision_log:
            add(entry)
        add("```")
        add()

    # Phase II
    if phase_ii is not None:
        add("## Phase II: Efficacy")
        add()
        trt = phase_ii.treatment_arm
        ctrl = phase_ii.control_arm
        add("### Primary Endpoint Results")
        add()
        add("| Metric | Treatment | Control |")
        add("|---|---|---|")
        add(f"| ORR | {trt.get('ORR', 0):.1%} | {ctrl.get('ORR', 0):.1%} |")
        add(f"| DCR | {trt.get('DCR', 0):.1%} | {ctrl.get('DCR', 0):.1%} |")
        add(f"| PFS at 6mo | {trt.get('PFS_6mo', 0):.1%} | {ctrl.get('PFS_6mo', 0):.1%} |")
        add(f"| N | {int(trt.get('n', 0))} | {int(ctrl.get('n', 0))} |")
        add()
        add(f"- **p-value**: {phase_ii.p_value:.4f}")
        add(f"- **Statistically Significant**: {'Yes' if phase_ii.statistically_significant else 'No'}")
        add(f"- **Post-hoc Power Estimate**: {phase_ii.power_estimate:.2f}")
        add()

        if phase_ii.biomarker_subgroup:
            add("### Biomarker Subgroup Analysis")
            add()
            add("| Metric | Value |")
            add("|---|---|")
            for key, val in phase_ii.biomarker_subgroup.items():
                if isinstance(val, float) and not np.isnan(val):
                    if key.startswith("ORR") or key.startswith("PFS"):
                        add(f"| {key} | {val:.1%} |")
                    elif key.startswith("n_"):
                        add(f"| {key} | {int(val)} |")
                    else:
                        add(f"| {key} | {val:.2f} |")
                elif key == "biomarker_OR":
                    add(f"| {key} | {val:.2f} |")
            add()

    # Phase III
    if phase_iii is not None:
        add("## Phase III: Comparative")
        add()
        add("### Overall Survival Analysis")
        add()
        add(f"- **Hazard Ratio**: {phase_iii.hazard_ratio:.3f} "
            f"(95% CI: {phase_iii.hr_ci_lower:.3f} - {phase_iii.hr_ci_upper:.3f})")
        add(f"- **p-value**: {phase_iii.p_value:.4f}")
        add(f"- **Statistically Significant**: {'Yes' if phase_iii.significant else 'No'}")
        add(f"- **Median Survival (Treatment)**: {phase_iii.median_survival_treatment:.1f} months")
        add(f"- **Median Survival (Control)**: {phase_iii.median_survival_control:.1f} months")
        add()

        if phase_iii.subgroup_analysis is not None and len(phase_iii.subgroup_analysis) > 0:
            add("### Subgroup Analysis")
            add()
            sub = phase_iii.subgroup_analysis
            add("| Subgroup | N (Trt) | N (Ctrl) | HR | 95% CI | p-value |")
            add("|---|---|---|---|---|---|")
            for _, row in sub.iterrows():
                add(
                    f"| {row['subgroup']} | {int(row['n_trt'])} | {int(row['n_ctrl'])} | "
                    f"{row['hr']:.3f} | {row['ci_lower']:.3f}-{row['ci_upper']:.3f} | "
                    f"{row['p_value']:.4f} |"
                )
            add()

    # Conclusion
    add("---")
    add()
    add("*This report was generated by a virtual clinical trial simulation engine. "
        "All data are synthetic and for research purposes only.*")
    add()

    return "\n".join(lines)
