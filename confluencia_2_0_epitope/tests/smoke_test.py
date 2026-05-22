from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training import (
    export_epitope_model_bytes,
    import_epitope_model_bytes,
    predict_epitope_model,
    train_epitope_model,
)
from core.reliability import credible_eval_epitope
from core.torch_mamba import TorchMambaConfig
from core.report_template import generate_experiment_report, save_report_csv


def _make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    seq = ["SLYNTVATL", "GILGFVFTL", "NLVPMVATV", "LLFGYPVYV"]
    df = pd.DataFrame(
        {
            "epitope_seq": [seq[i % len(seq)] for i in range(n)],
            "dose": rng.uniform(0.2, 6.0, size=n),
            "freq": rng.uniform(0.5, 3.0, size=n),
            "treatment_time": rng.uniform(0, 96, size=n),
            "circ_expr": rng.uniform(0, 2.0, size=n),
            "ifn_score": rng.uniform(0, 1.5, size=n),
        }
    )
    df["efficacy"] = 0.2 * df["dose"] + 0.18 * df["freq"] + 0.1 * df["circ_expr"] + rng.normal(0, 0.15, size=n)
    return df


def test_train_predict_export() -> None:
    df = _make_df(50)
    model_bundle, report = train_epitope_model(
        df,
        compute_mode="low",
        model_backend="ridge",
        torch_cfg=None,
    )
    result_df, sens = predict_epitope_model(model_bundle, df, sensitivity_sample_idx=0)

    assert len(result_df) == len(df), "result rows mismatch"
    assert "efficacy_pred" in result_df.columns, "missing efficacy_pred"
    assert "pred_uncertainty" in result_df.columns, "missing uncertainty"
    assert model_bundle.model_backend == "ridge", "backend mismatch"
    assert report.used_label is True, "label usage mismatch"
    assert sens.sample_index == 0, "sensitivity index mismatch"

    exported = export_epitope_model_bytes(model_bundle)
    loaded_bundle = import_epitope_model_bytes(exported)
    loaded_pred_df, _ = predict_epitope_model(loaded_bundle, df, sensitivity_sample_idx=0)
    delta = np.abs(
        pd.to_numeric(loaded_pred_df["efficacy_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        - pd.to_numeric(result_df["efficacy_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    )
    assert float(delta.max(initial=0.0)) < 1e-6, "imported model prediction mismatch"


def test_auto_save_repro() -> None:
    """Test that training auto-saves a reproducibility bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        df = _make_df(50)
        from core.training import _auto_save_repro

        run_id = _auto_save_repro(
            module="smoke-test",
            data_df=df,
            config={"model_backend": "ridge", "compute_mode": "low"},
            metrics={"mae": 0.1, "rmse": 0.15, "r2": 0.8},
            log_dir=tmpdir,
        )

        runs_csv = Path(tmpdir) / "runs.csv"
        assert runs_csv.exists(), "runs.csv not created"
        runs_df = pd.read_csv(runs_csv)
        assert len(runs_df) == 1, "runs.csv should have 1 row"
        assert runs_df.iloc[0]["module"] == "smoke-test"

        md_path = Path(tmpdir) / f"{run_id}.md"
        assert md_path.exists(), "markdown report not created"
        md_content = md_path.read_text(encoding="utf-8")
        assert "Experiment Report" in md_content, "report missing header"
        assert "Config" in md_content, "report missing Config section"
        assert "Metrics" in md_content, "report missing Metrics section"
        assert "Environment Dependencies" in md_content, "report missing env deps section"


def test_aa_composition_stratification() -> None:
    """Test that credible_eval_epitope returns AA composition stratification."""
    df = _make_df(60)
    result = credible_eval_epitope(
        df=df,
        backend="ridge",
        compute_mode="low",
        seed=42,
        test_ratio=0.2,
        val_ratio=0.2,
        cv_folds=3,
        top_n_failures=5,
        torch_cfg=TorchMambaConfig(),
        external_df=None,
    )

    assert result.get("enabled", False), f"credible eval not enabled: {result.get('reason')}"
    aa_strat = result.get("aa_composition_strat_df")
    assert aa_strat is not None, "aa_composition_strat_df missing from credible_eval result"
    assert isinstance(aa_strat, pd.DataFrame), "aa_composition_strat_df should be a DataFrame"
    assert not aa_strat.empty, "aa_composition_strat_df should not be empty"
    assert "property" in aa_strat.columns, "missing property column"
    assert "bin" in aa_strat.columns, "missing bin column"
    assert "rmse" in aa_strat.columns, "missing rmse column"
    # Should have hydrophobic, charged, and aromatic rows
    properties = set(aa_strat["property"].unique())
    assert "hydrophobic" in properties, "missing hydrophobic stratification"
    assert "charged" in properties, "missing charged stratification"
    assert "aromatic" in properties, "missing aromatic stratification"


def test_report_template() -> None:
    """Test unified experiment report template."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = generate_experiment_report(
            module="test-module",
            config={"backend": "ridge"},
            metrics={"mae": 0.1, "rmse": 0.15, "r2": 0.9},
            data_hash="abc123",
            n_rows=100,
            env_deps={"python": "3.11.0", "numpy": "1.24.0"},
            python_executable="/usr/bin/python",
        )
        assert "Experiment Report" in report
        assert "test-module" in report
        assert "abc123" in report
        assert "Config" in report
        assert "Metrics" in report
        assert "Environment Dependencies" in report

        csv_path = save_report_csv(
            module="test-module",
            config={"backend": "ridge"},
            metrics={"mae": 0.1},
            data_hash="abc123",
            n_rows=100,
            env_deps={"python": "3.11.0"},
            log_dir=tmpdir,
        )
        assert csv_path.exists()
        runs_df = pd.read_csv(csv_path)
        assert len(runs_df) == 1


def main() -> None:
    test_train_predict_export()
    print("[PASS] train_predict_export")

    test_auto_save_repro()
    print("[PASS] auto_save_repro")

    test_aa_composition_stratification()
    print("[PASS] aa_composition_stratification")

    test_report_template()
    print("[PASS] report_template")

    print("epitope smoke test passed")


if __name__ == "__main__":
    main()
