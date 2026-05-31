"""Confluencia Bridge — thin Python adapter for R (reticulate) and VS Code (JSON-RPC).

Design: only accepts JSON-safe inputs (str, float, dict of primitives),
        only returns JSON-safe outputs (dict, list, float).
        Never passes complex Python objects across the language boundary.
"""

import sys
import os
import json
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Path setup — resolve confluencia module locations
# ---------------------------------------------------------------------------

def _setup_paths() -> List[str]:
    """Return ordered sys.path entries so confluencia modules are importable."""
    # Try common project root locations
    candidates: List[Path] = []
    # 1. CONFLUENCIA_ROOT env var (explicit user setting)
    env_root = os.environ.get("CONFLUENCIA_ROOT", "")
    if env_root:
        candidates.append(Path(env_root))
    # 2. Relative to this script (inst/python/ → project root)
    here = Path(__file__).resolve().parent
    candidates.append(here.parent.parent.parent)  # confluencia-rpkg/ → IGEM集成方案/
    # 3. Current working directory
    candidates.append(Path(os.getcwd()))

    paths: List[str] = []
    module_names = [
        "confluencia_shared",
        "confluencia-2.0-drug",
        "confluencia-2.0-epitope",
        "confluencia_circrna",
        "confluencia_joint",
    ]
    for root in candidates:
        root = Path(root)
        if not root.is_dir():
            continue
        for name in module_names:
            p = root / name
            if p.is_dir() and str(p) not in paths:
                paths.append(str(p))
        # Always include the root itself for confluencia_shared etc.
        if str(root) not in paths:
            paths.append(str(root))
    return paths


for p in _setup_paths():
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Module alias registration (hyphenated dirs)
# ---------------------------------------------------------------------------

def _register_aliases() -> None:
    """Make confluencia-2.0-drug importable as a package."""
    import importlib.util

    ALIASES = {
        "confluencia_2_0_drug": "confluencia-2.0-drug",
        "confluencia_2_0_epitope": "confluencia-2.0-epitope",
    }
    for alias_name, real_name in ALIASES.items():
        try:
            importlib.import_module(alias_name)
            continue
        except ImportError:
            pass
        for base in _setup_paths():
            real_dir = Path(base) / real_name
            if real_dir.is_dir():
                class _AliasFinder:
                    def __init__(self, alias: str, real_path: Path):
                        self.alias = alias
                        self.real_path = real_path
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname != self.alias and not fullname.startswith(self.alias + "."):
                            return None
                        return importlib.util.spec_from_file_location(
                            fullname,
                            self.real_path / "__init__.py",
                            submodule_search_locations=[str(self.real_path)],
                        )
                sys.meta_path.insert(0, _AliasFinder(alias_name, real_dir))
                break


_register_aliases()


# ---------------------------------------------------------------------------
# ConfluenciaBridge class — all methods return JSON-safe primitives
# ---------------------------------------------------------------------------

class ConfluenciaBridge:
    """Thin adapter wrapping Confluencia Python APIs for R/VS Code consumption.

    Every method accepts only JSON-safe inputs (str, float, dict of primitives)
    and returns only JSON-safe outputs (dict, list of dicts, float).
    Complex Python objects (DataFrames, ndarrays, model bundles) never cross
    the language boundary.
    """

    # ---- PK Simulation (CTM) ----

    def ctm_params(self, binding: float = 0.5,
                    immune: float = 0.5,
                    inflammation: float = 0.5) -> Dict[str, float]:
        """Derive CTMParams from micro scores. Returns dict of floats."""
        from confluencia_2_0_drug.core.ctm import params_from_micro_scores
        p = params_from_micro_scores(binding, immune, inflammation)
        return dataclasses.asdict(p)

    def ctm_simulate(self, dose: float, freq: float,
                      params_dict: Optional[Dict[str, float]] = None,
                      binding: float = 0.5, immune: float = 0.5,
                      inflammation: float = 0.5,
                      horizon: int = 72) -> Dict[str, List[float]]:
        """Simulate small-molecule 6-compartment PK. Returns dict of lists."""
        from confluencia_2_0_drug.core.ctm import simulate_ctm, CTMParams, params_from_micro_scores
        if params_dict is None:
            params = params_from_micro_scores(binding, immune, inflammation)
        else:
            params = CTMParams(**params_dict)
        df = simulate_ctm(dose, freq, params, horizon=horizon)
        return df.to_dict(orient="list")

    def rna_ctm_params(self, modification: str = "none",
                        delivery_vector: str = "LNP_standard",
                        route: str = "IV",
                        ires_score: float = 0.5,
                        gc_content: float = 0.5,
                        struct_stability: float = 0.5,
                        innate_immune_score: float = 0.0) -> Dict[str, float]:
        """Derive RNACTMParams from modification/delivery config."""
        from confluencia_2_0_drug.core.ctm import infer_rna_ctm_params
        p = infer_rna_ctm_params(
            modification=modification, delivery_vector=delivery_vector,
            route=route, ires_score=ires_score, gc_content=gc_content,
            struct_stability=struct_stability,
            innate_immune_score=innate_immune_score,
        )
        return dataclasses.asdict(p)

    def rna_ctm_simulate(self, dose: float, freq: float,
                          params_dict: Optional[Dict[str, float]] = None,
                          modification: str = "none",
                          delivery_vector: str = "LNP_standard",
                          route: str = "IV",
                          horizon: int = 168) -> Dict[str, List[float]]:
        """Simulate circRNA 6-compartment PK. Returns dict of lists."""
        from confluencia_2_0_drug.core.ctm import simulate_rna_ctm, RNACTMParams, infer_rna_ctm_params
        if params_dict is None:
            params = infer_rna_ctm_params(modification=modification,
                                           delivery_vector=delivery_vector,
                                           route=route)
        else:
            params = RNACTMParams(**params_dict)
        df = simulate_rna_ctm(dose, freq, params, horizon=horizon)
        return df.to_dict(orient="list")

    # ---- Drug Prediction ----

    def drug_predict(self, bundle_path: str, smiles: str,
                      env_params: Optional[Dict[str, float]] = None) -> float:
        """Predict drug efficacy for a single SMILES. Returns scalar float."""
        import joblib
        from confluencia_2_0_drug.core.predictor import predict_one
        bundle = joblib.load(bundle_path)
        result = predict_one(bundle, smiles, env_params or {})
        return float(result)

    def drug_train(self, csv_path: str,
                    smiles_col: str = "smiles",
                    target_col: str = "efficacy",
                    env_cols: Optional[List[str]] = None,
                    model_name: str = "gbr",
                    test_size: float = 0.2,
                    random_state: int = 42,
                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train a drug model from CSV. Saves bundle as .joblib, returns metrics.

        If save_path is None, saves alongside the CSV as <csv_stem>_drug_bundle.joblib.
        Returns dict with keys: mae, rmse, r2, n_train, n_val, bundle_path.
        """
        import joblib
        import pandas as pd
        from confluencia_2_0_drug.core.predictor import train_bundle

        df = pd.read_csv(csv_path)
        bundle, metrics = train_bundle(
            df, smiles_col=smiles_col, target_col=target_col,
            env_cols=env_cols, model_name=model_name,
            test_size=test_size, random_state=random_state,
        )

        if save_path is None:
            stem = Path(csv_path).stem
            save_path = str(Path(csv_path).parent / f"{stem}_drug_bundle.joblib")
        joblib.dump(bundle, save_path)

        return {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "explained_variance": metrics.get("explained_variance"),
            "n_train": metrics["n_train"],
            "n_val": metrics["n_val"],
            "n_features": metrics["n_features"],
            "bundle_path": save_path,
        }

    # ---- Epitope Prediction ----

    def epitope_predict(self, bundle_path: str, sequence: str,
                         env_params: Optional[Dict[str, float]] = None) -> float:
        """Predict epitope binding for a single sequence. Returns scalar float."""
        import joblib
        from confluencia_2_0_epitope.core.predictor import predict_one
        bundle = joblib.load(bundle_path)
        result = predict_one(bundle, sequence, env_params or {})
        return float(result)

    def epitope_train(self, csv_path: str,
                       sequence_col: str = "sequence",
                       target_col: str = "binding",
                       env_cols: Optional[List[str]] = None,
                       model_name: str = "hgb",
                       test_size: float = 0.2,
                       random_state: int = 42,
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train an epitope model from CSV. Saves bundle as .joblib, returns metrics.

        If save_path is None, saves alongside the CSV as <csv_stem>_epitope_bundle.joblib.
        Returns dict with keys: mae, rmse, r2, n_train, n_val, bundle_path.
        """
        import joblib
        import pandas as pd
        from confluencia_2_0_epitope.core.predictor import train_bundle

        df = pd.read_csv(csv_path)
        bundle, metrics = train_bundle(
            df, sequence_col=sequence_col, target_col=target_col,
            env_cols=env_cols or [], model_name=model_name,
            test_size=test_size, random_state=random_state,
        )

        if save_path is None:
            stem = Path(csv_path).stem
            save_path = str(Path(csv_path).parent / f"{stem}_epitope_bundle.joblib")
        joblib.dump(bundle, save_path)

        return {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "explained_variance": metrics.get("explained_variance"),
            "n_train": metrics["n_train"],
            "n_val": metrics["n_val"],
            "n_features": metrics["n_features"],
            "bundle_path": save_path,
        }

    # ---- circRNA Immunogenicity ----

    def circrna_immunogenicity(self, sequence: str) -> Dict[str, Any]:
        """Predict circRNA immunogenicity scores (RIG-I, TLR, PKR)."""
        from confluencia_circrna.core.immune_sensing import predict_circrna_immunogenicity
        result = predict_circrna_immunogenicity(sequence)
        return {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in result.items()}

    def circrna_pipeline(self, sequence: str,
                          gene_expression: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Run full circRNA pipeline. Returns structured dict."""
        from confluencia_circrna.pipeline.circrna_pipeline import CircRNAPipeline
        pipeline = CircRNAPipeline()
        result = pipeline.run(sequence, gene_expression=gene_expression)
        # Convert dataclass result to dict
        if dataclasses.is_dataclass(result):
            return self._dataclass_to_dict(result)
        return dict(result)

    # ---- MHC Encoding ----

    def mhc_encode(self, peptide: str, allele: str) -> List[float]:
        """Encode peptide-allele pair as MHC feature vector."""
        from confluencia_2_0_epitope.core.mhc_features import MHCFeatureEncoder, MHCIIFeatureEncoder, detect_mhc_class
        mhc_class = detect_mhc_class(allele)
        if mhc_class == "II":
            encoder = MHCIIFeatureEncoder()
        else:
            encoder = MHCFeatureEncoder()
        arr = encoder.encode(peptide, allele)
        return arr.tolist()

    def mhc_detect_class(self, allele: str) -> str:
        """Detect MHC class (I or II) from allele string."""
        from confluencia_2_0_epitope.core.mhc_features import detect_mhc_class
        return detect_mhc_class(allele)

    # ---- Mamba3 Encoding ----

    def mamba3_encode(self, sequence: str) -> Dict[str, List[float]]:
        """Encode sequence with Mamba3Lite encoder."""
        from confluencia_2_0_epitope.core.mamba3 import Mamba3LiteEncoder
        encoder = Mamba3LiteEncoder()
        result = encoder.encode(sequence)
        return {k: v.tolist() for k, v in result.items()}

    # ---- Joint Evaluation ----

    def joint_evaluate(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run 5D joint evaluation. Returns structured result dict."""
        from confluencia_joint.joint_input import JointInput
        from confluencia_joint.joint_evaluator import JointEvaluationEngine

        # Filter input_dict to only fields that JointInput accepts
        field_names = {f.name for f in dataclasses.fields(JointInput)}
        filtered = {k: v for k, v in input_dict.items() if k in field_names}
        inp = JointInput(**filtered)

        engine = JointEvaluationEngine()
        result = engine.evaluate_single(inp)
        return self._dataclass_to_dict(result)

    # ---- Regression Metrics ----

    def reg_metrics(self, y_true: List[float], y_pred: List[float],
                     prefix: str = "") -> Dict[str, float]:
        """Compute MAE, RMSE, R2 regression metrics."""
        from confluencia_shared.metrics import reg_metrics
        import numpy as np
        return reg_metrics(np.array(y_true), np.array(y_pred), prefix=prefix)

    # ---- Community Hub (Model & Data Sharing) ----

    def hub_push_model(self, bundle_path: str, metadata: Optional[Dict[str, Any]] = None,
                       uploader: str = "anonymous", license: str = "MIT",
                       strip_env_medians: bool = False) -> Dict[str, str]:
        """Share a trained model with the community (federated — no raw data exposed).

        Args:
            strip_env_medians: If True, remove env_medians to eliminate
                statistical traces of training data.
        """
        from confluencia_cli.hub import ConfluenciaHub
        hub = ConfluenciaHub()
        model_id = hub.push_model(bundle_path, metadata=metadata, uploader=uploader,
                                   license=license, strip_env_medians=strip_env_medians)
        return {"model_id": model_id, "message": "Model shared successfully"}

    def hub_pull_model(self, model_id: str) -> Dict[str, str]:
        """Download a community model. Returns local path."""
        from confluencia_cli.hub import ConfluenciaHub
        hub = ConfluenciaHub()
        local_path = hub.pull_model(model_id)
        return {"bundle_path": local_path}

    def hub_list_models(self, task: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """List available community models."""
        from confluencia_cli.hub import ConfluenciaHub
        hub = ConfluenciaHub()
        return hub.list_models(task=task, limit=limit)

    def hub_push_data(self, csv_path: str, license: str = "CC-BY-4.0",
                      anonymous: bool = True, uploader: str = "anonymous") -> Dict[str, str]:
        """Contribute a dataset to the community pool (with license)."""
        from confluencia_cli.hub import ConfluenciaHub
        hub = ConfluenciaHub()
        dataset_id = hub.push_data(csv_path, license=license, anonymous=anonymous, uploader=uploader)
        return {"dataset_id": dataset_id, "message": "Data contributed successfully"}

    def hub_data_stats(self) -> Dict[str, Any]:
        """Get community dataset statistics (no raw data exposed)."""
        from confluencia_cli.hub import ConfluenciaHub
        hub = ConfluenciaHub()
        return hub.data_stats()

    # ---- Plugin System (Extensible Platform) ----

    def plugin_register_model(self, name: str, module_path: str,
                               function_name: str) -> Dict[str, str]:
        """Register a custom model from a Python module.

        Args:
            name: Model name (e.g., "xgboost")
            module_path: Python module path (e.g., "xgboost")
            function_name: Factory function name (e.g., "XGBRegressor")
        """
        from confluencia_cli.plugins import register_model
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, function_name)
        register_model(name, lambda **kw: cls(**kw))
        return {"name": name, "message": f"Model '{name}' registered from {module_path}.{function_name}"}

    def plugin_register_encoder(self, name: str, module_path: str,
                                 function_name: str) -> Dict[str, str]:
        """Register a custom sequence encoder from a Python module."""
        from confluencia_cli.plugins import register_encoder
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, function_name)
        register_encoder(name, fn)
        return {"name": name, "message": f"Encoder '{name}' registered from {module_path}.{function_name}"}

    def plugin_register_dimension(self, name: str, weight: float,
                                   description: str = "") -> Dict[str, str]:
        """Register a new evaluation dimension with weight."""
        from confluencia_cli.plugins import register_dimension
        register_dimension(name, weight=weight, description=description)
        return {"name": name, "weight": weight, "message": f"Dimension '{name}' registered (weight={weight})"}

    def plugin_set_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Set scoring weights for evaluation dimensions."""
        from confluencia_cli.plugins import set_weights, get_weights
        set_weights(**weights)
        return {"weights": get_weights()}

    def plugin_list(self) -> Dict[str, Any]:
        """List all registered plugins (models, encoders, PK solvers, dimensions)."""
        from confluencia_cli.plugins import list_registry
        return list_registry()

    # ---- Utility ----

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Recursively convert dataclass to JSON-safe dict."""
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            result = {}
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                result[f.name] = self._serialize_value(val)
            return result
        return self._serialize_value(obj)

    def _serialize_value(self, val: Any) -> Any:
        """Convert a value to JSON-safe format."""
        if val is None:
            return None
        if isinstance(val, (bool, int, float, str)):
            return val
        if isinstance(val, (list, tuple)):
            return [self._serialize_value(v) for v in val]
        if isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}
        if dataclasses.is_dataclass(val):
            return self._dataclass_to_dict(val)
        if hasattr(val, "tolist"):  # numpy
            return val.tolist()
        if hasattr(val, "to_dict"):  # pandas DataFrame/Series
            try:
                return val.to_dict(orient="list")
            except TypeError:
                # dataclass or other object with to_dict but no orient parameter
                return val.to_dict()
        return str(val)


# ---------------------------------------------------------------------------
# JSON-RPC mode for VS Code extension
# ---------------------------------------------------------------------------

def _run_json_rpc() -> None:
    """Run as JSON-RPC server over stdin/stdout for VS Code extension."""
    bridge = ConfluenciaBridge()

    # Emit ready event
    print(json.dumps({"event": "ready", "data": {"version": "1.0.0"}}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            method = msg.get("method", "")
            params = msg.get("params", {})
            id_ = msg.get("id", 0)

            if method == "exit":
                break

            # Dispatch to bridge method
            fn = getattr(bridge, method, None)
            if fn is None:
                resp = {"id": id_, "error": f"Unknown method: {method}"}
            else:
                try:
                    result = fn(**params)
                    resp = {"id": id_, "result": result}
                except Exception as ex:
                    import traceback
                    resp = {"id": id_, "error": f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"}

            print(json.dumps(resp), flush=True)

        except json.JSONDecodeError as ex:
            print(json.dumps({"id": 0, "error": f"Invalid JSON: {ex}"}), flush=True)


# ---------------------------------------------------------------------------
# CLI mode (one-shot invocation for simple testing)
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    """Run a single bridge method from CLI arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Confluencia Bridge CLI")
    parser.add_argument("--method", required=True, help="Bridge method name")
    parser.add_argument("--args", default="{}", help="JSON dict of arguments")
    args = parser.parse_args()

    bridge = ConfluenciaBridge()
    fn = getattr(bridge, args.method, None)
    if fn is None:
        print(f"Error: Unknown method '{args.method}'")
        sys.exit(1)

    params = json.loads(args.args)
    result = fn(**params)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Confluencia Bridge")
    parser.add_argument("--mode", choices=["rpc", "cli"], default="cli",
                        help="Run mode: rpc (JSON-RPC stdin/stdout) or cli (one-shot)")
    parser.add_argument("--method", help="Method name (cli mode)")
    parser.add_argument("--args", default="{}", help="JSON args (cli mode)")
    parsed = parser.parse_args()

    if parsed.mode == "rpc":
        _run_json_rpc()
    else:
        # CLI mode
        bridge = ConfluenciaBridge()
        fn = getattr(bridge, parsed.method, None)
        if fn is None:
            print(f"Error: Unknown method '{parsed.method}'")
            sys.exit(1)
        params = json.loads(parsed.args)
        result = fn(**params)
        print(json.dumps(result, indent=2))