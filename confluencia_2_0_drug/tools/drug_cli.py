from __future__ import annotations

import argparse
import re
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.plotting import save_regression_diagnostic_plots
from src.common.dataset_fetch import concat_tables, read_table_any
from src.drug.crawler import (
    crawl_pubchem_activity_dataset,
    crawl_docking_training_dataset,
    crawl_multiscale_training_dataset,
)
from src.drug.featurizer import MoleculeFeatures
from src.drug.generative import GanConfig, PropertyFilters, calc_props, default_score, generate_molecules, passes_filters, select_diverse_ranked
from src.drug.predictor import DrugModelBundle, build_model, cross_validate, infer_env_cols, make_xy, predict_one, train_bundle
from src.common.optim.hyperopt import run_hyper_search


def cmd_sites(args: argparse.Namespace) -> int:
    print("== Supported crawl sites (drug) ==")
    print("pubchem: PubChem PUG-REST API (默认)，输出列: cid, smiles, activity_score, n_active, n_inactive, n_total")
    print("  可选扩展字段: 分子性质（分子量、logP、TPSA、HBD/HBA 等）与同义名（synonyms）")
    print("urlcsv: 用户提供的CSV/TSV/Excel URL或本地路径（会合并多个表并增加 _source 列）")
    print("docking: 分子对接数据表（标准化列: ligand_smiles, protein_sequence/protein_pdb, binding_score, pocket_path）")
    print("multiscale: 多尺度训练数据表（标准化列: smiles, target, D, Vmax, Km）")
    print("提示：pubchem 可用 --cid-list 指定 CID 列表文件（支持 csv/tsv/xlsx/txt）")
    print("\n示例:")
    print("  drug_cli.py crawl --site pubchem --start-cid 1 --n 200 --out data/pubchem_activity.csv")
    print("  drug_cli.py crawl --site urlcsv --source https://.../a.csv https://.../b.csv --out data/merged.csv")
    print("  drug_cli.py crawl-train --site urlcsv --source data/merged.csv --smiles-col smiles --target-col activity_score")
    return 0


def _parse_kv(items: Optional[List[str]]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    if not items:
        return params
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid param '{item}', expected key=value")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid param '{item}', empty key")
        params[k] = float(v)
    return params


def _parse_cid_text(text: str) -> List[int]:
    if not text:
        return []
    tokens = re.split(r"[\s,;]+", str(text).strip())
    out: List[int] = []
    for t in tokens:
        if not t:
            continue
        try:
            v = int(float(t))
            if v > 0:
                out.append(v)
        except Exception:
            continue
    return out


def _parse_property_fields(text: str) -> Optional[List[str]]:
    if not text:
        return None
    parts = [p.strip() for p in str(text).split(",")]
    fields = [p for p in parts if p]
    return fields or None


def _load_cids_from_path(path: str, *, cid_col: Optional[str] = None) -> List[int]:
    if not path:
        return []

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}:
        df = read_table_any(path)
        col = None
        if cid_col and cid_col in df.columns:
            col = cid_col
        else:
            for c in ("cid", "CID", "Cid"):
                if c in df.columns:
                    col = c
                    break
        if col is None:
            # fallback: use first column
            series = df.iloc[:, 0]
        else:
            series = df[col]
        text = "\n".join([str(x) for x in series.tolist()])
        return _parse_cid_text(text)

    # fallback: treat as plain text file
    if not p.exists():
        raise FileNotFoundError(str(p))
    return _parse_cid_text(p.read_text(encoding="utf-8", errors="ignore"))


def cmd_train(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)

    env_cols = infer_env_cols(
        df,
        smiles_col=args.smiles_col,
        target_col=args.target,
        env_cols=args.env_cols,
    )

    bundle, metrics = train_bundle(
        df,
        smiles_col=args.smiles_col,
        target_col=args.target,
        env_cols=env_cols,
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        featurizer_version=args.featurizer_version,
        drop_invalid_smiles=(not bool(args.keep_invalid_smiles)),
    )

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    print("== Training done ==")
    print(f"model_out: {out_path}")
    print(f"smiles_col: {bundle.smiles_col}")
    print(f"target_col: {bundle.target_col}")
    print(f"env_cols: {bundle.env_cols}")
    print(f"n_features: {metrics['n_features']}")
    print(f"invalid_smiles: {metrics['invalid_smiles']}")
    if 'dropped_invalid_smiles' in metrics:
        print(f"dropped_invalid_smiles: {metrics['dropped_invalid_smiles']}")
    print(f"MAE:  {metrics['mae']:.6g}")
    print(f"RMSE: {metrics['rmse']:.6g}")
    print(f"R2:   {metrics['r2']:.6g}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)

    env_cols = infer_env_cols(
        df,
        smiles_col=args.smiles_col,
        target_col=args.target,
        env_cols=args.env_cols,
    )

    x, y, env_medians, feature_names = make_xy(
        df,
        smiles_col=args.smiles_col,
        target_col=args.target,
        env_cols=list(env_cols),
        featurizer=MoleculeFeatures(version=int(args.featurizer_version), radius=int(getattr(args, "radius", 2)), n_bits=int(getattr(args, "n_bits", 2048))),
        env_medians=None,
    )

    base = build_model(model_name=args.model, random_state=int(args.seed))

    param_grids = {
        "hgb": {"l2_regularization": [0.0, 1e-6, 1e-4], "max_iter": [100, 300]},
        "gbr": {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1]},
        "rf": {"n_estimators": [100, 300, 500], "max_depth": [None, 8, 16]},
        "mlp": {"mlp__alpha": [1e-4, 1e-3], "mlp__hidden_layer_sizes": [(128, 64), (256, 128)]},
        "ridge": {"alpha": [0.1, 1.0, 10.0]},
    }

    grid = param_grids.get(args.model, {})
    if args.param_grid:
        import json

        with open(args.param_grid, "r", encoding="utf-8") as f:
            grid = json.load(f)

    best_est, best_params, cv_results = run_hyper_search(
        base,
        grid,
        x,
        y,
        strategy=args.strategy,
        n_iter=int(args.n_iter or 20),
        cv=int(args.cv or 5),
        scoring=None,
        n_jobs=-1,
        random_state=int(args.seed),
    )

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_est, out_path)

    print("== Tuning done ==")
    print(f"best_params: {best_params}")
    print(f"saved_model: {out_path}")
    return 0


def cmd_cv(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)

    env_cols = infer_env_cols(
        df,
        smiles_col=args.smiles_col,
        target_col=args.target,
        env_cols=args.env_cols,
    )

    report = cross_validate(
        df,
        smiles_col=args.smiles_col,
        target_col=args.target,
        env_cols=env_cols,
        model_name=args.model,
        n_splits=args.n_splits,
        random_state=args.seed,
        featurizer_version=args.featurizer_version,
        drop_invalid_smiles=(not bool(args.keep_invalid_smiles)),
    )

    summary = cast(Dict[str, float], report["summary"])
    print("== Cross Validation ==")
    print(f"data: {args.data}")
    print(f"model: {args.model}")
    print(f"n_splits: {args.n_splits}")
    print(f"n_samples: {summary['n_samples']}")
    print(f"n_features: {summary['n_features']}")
    print(f"invalid_smiles: {summary['invalid_smiles']}")
    print(f"dropped_invalid_smiles: {summary['dropped_invalid_smiles']}")
    print(f"MAE:  {summary['mae_mean']:.6g} ± {summary['mae_std']:.6g}")
    print(f"RMSE: {summary['rmse_mean']:.6g} ± {summary['rmse_std']:.6g}")
    print(f"R2:   {summary['r2_mean']:.6g} ± {summary['r2_std']:.6g}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    bundle: DrugModelBundle = cast(DrugModelBundle, joblib.load(args.model))
    env_params = _parse_kv(args.param)

    y = predict_one(bundle, smiles=args.smiles, env_params=env_params)
    print("== Prediction ==")
    print(f"smiles: {args.smiles}")
    print(f"pred: {y:.6g}")
    if bundle.env_cols:
        resolved = {c: float(env_params.get(c, bundle.env_medians.get(c, 0.0))) for c in bundle.env_cols}
        print(f"env: {resolved}")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    bundle: DrugModelBundle = cast(DrugModelBundle, joblib.load(args.model))
    df = pd.read_csv(args.candidates)

    if args.smiles_col not in df.columns:
        raise ValueError(f"Missing smiles_col '{args.smiles_col}' in candidates CSV")

    for c in bundle.env_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].astype(float)
        df[c] = df[c].fillna(bundle.env_medians.get(c, 0.0))

    x, valids = _make_x_only_drug(
        df,
        smiles_col=args.smiles_col,
        env_cols=list(bundle.env_cols),
        env_medians=bundle.env_medians,
        featurizer_version=int(getattr(bundle, "featurizer_version", 2)),
        radius=int(getattr(bundle, "radius", 2)),
        n_bits=int(getattr(bundle, "n_bits", 2048)),
    )

    preds = np.empty((x.shape[0],), dtype=np.float32)
    chunk = 10000
    for start in range(0, x.shape[0], chunk):
        end = min(x.shape[0], start + chunk)
        preds[start:end] = np.asarray(bundle.model.predict(x[start:end]), dtype=np.float32).reshape(-1)
    if valids is not None:
        invalid_mask = ~valids.astype(bool)
        preds = preds.astype(float)
        preds[invalid_mask] = float("nan")

    out = df.copy()
    out[args.out_col] = preds

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("== Screening done ==")
    print(f"candidates: {args.candidates}")
    print(f"out: {out_path}")
    print(f"out_col: {args.out_col}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    bundle: DrugModelBundle = joblib.load(args.model)
    d = asdict(bundle)
    d["model"] = str(type(bundle.model))
    print(d)
    return 0


def cmd_suggest_env(args: argparse.Namespace) -> int:
    bundle: DrugModelBundle = cast(DrugModelBundle, joblib.load(args.model))
    smiles = args.smiles
    n = len(bundle.env_cols) if bundle.env_cols else 0
    if n == 0:
        raise ValueError("模型没有环境变量可优化")

    bounds_input = args.bounds
    if bounds_input:
        parts = [p.strip() for p in bounds_input.split(",") if p.strip()]
        if len(parts) != n:
            raise ValueError(f"--bounds 需要 {n} 个 low:high 对，当前提供 {len(parts)} 个")
        env_bounds = []
        for p in parts:
            if ":" not in p:
                raise ValueError(f"bounds 格式错误: {p}，应为 low:high")
            lo_s, hi_s = p.split(":", 1)
            env_bounds.append((float(lo_s), float(hi_s)))
    else:
        env_bounds = []
        for c in bundle.env_cols:
            med = float(bundle.env_medians.get(c, 0.0))
            if med == 0.0:
                env_bounds.append((-1.0, 1.0))
            else:
                env_bounds.append((med * 0.5, med * 1.5))

    # choose appropriate suggest function based on bundle type
    # prefer torch if bundle is TorchDrugModelBundle like
    try:
        from src.drug.torch_predictor import suggest_env_by_de_torch
    except Exception:
        suggest_env_by_de_torch = None

    try:
        from src.drug.predictor import suggest_env_by_de_drug
    except Exception:
        suggest_env_by_de_drug = None

    # try torch version first
    if suggest_env_by_de_torch is not None and hasattr(bundle, "model_state"):
        # Torch bundle has model_state attribute; use torch optimizer
        best_env, best_val = suggest_env_by_de_torch(bundle, smiles=smiles, env_bounds=env_bounds, use_cuda=bool(args.use_cuda))
    elif suggest_env_by_de_drug is not None:
        best_env, best_val = suggest_env_by_de_drug(bundle, smiles=smiles, env_bounds=env_bounds)
    else:
        raise RuntimeError("无法找到适用于该模型的 DE 建议函数")

    mapped = {c: float(v) for c, v in zip(bundle.env_cols, best_env.tolist())}
    print("== DE Suggestion Result ==")
    print(f"smiles: {smiles}")
    print(f"pred: {best_val:.6g}")
    print(f"env: {mapped}")
    return 0
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    site = str(getattr(args, "site", "pubchem") or "pubchem").lower()

    if site == "pubchem":
        cid_list = _load_cids_from_path(getattr(args, "cid_list", "") or "", cid_col=getattr(args, "cid_col", None))
        prop_fields = _parse_property_fields(getattr(args, "property_fields", "") or "")
        df = crawl_pubchem_activity_dataset(
            start_cid=int(args.start_cid),
            n=int(args.n),
            cids=(cid_list if cid_list else None),
            sleep_seconds=float(args.sleep),
            min_total_outcomes=int(args.min_total),
            min_active=int(getattr(args, "min_active", 1)),
            treat_zero_unlabeled=(not bool(getattr(args, "keep_zero", False))),
            drop_invalid=(not bool(getattr(args, "keep_invalid", False))),
            cache_dir=str(args.cache_dir),
            timeout=float(args.timeout),
            include_properties=bool(getattr(args, "include_properties", False)),
            property_fields=prop_fields,
            include_synonyms=bool(getattr(args, "include_synonyms", False)),
        )
    elif site == "urlcsv":
        sources = list(getattr(args, "source", []) or [])
        if not sources:
            raise ValueError("site=urlcsv 需要提供 --source <url_or_path...>")
        df = concat_tables(
            sources,
            cache_dir=str(getattr(args, "cache_dir", "data/cache/http")),
            timeout=float(getattr(args, "timeout", 30.0)),
            sleep_seconds=float(getattr(args, "sleep", 0.2)),
            headers={"User-Agent": "drug-urlcsv/1.0 (research; contact: local)"},
        )
    elif site == "docking":
        sources = list(getattr(args, "source", []) or [])
        if not sources:
            raise ValueError("site=docking 需要提供 --source <url_or_path...>")
        df = crawl_docking_training_dataset(
            sources=sources,
            cache_dir=str(getattr(args, "cache_dir", "data/cache/http")),
            timeout=float(getattr(args, "timeout", 30.0)),
            sleep_seconds=float(getattr(args, "sleep", 0.2)),
            ligand_smiles_col=getattr(args, "ligand_smiles_col", None),
            protein_seq_col=getattr(args, "protein_seq_col", None),
            protein_pdb_col=getattr(args, "protein_pdb_col", None),
            binding_score_col=getattr(args, "binding_score_col", None),
            pocket_path_col=getattr(args, "pocket_path_col", None),
            normalize_smiles=bool(getattr(args, "normalize_smiles", False)),
            drop_invalid=(not bool(getattr(args, "keep_invalid", False))),
        )
    elif site == "multiscale":
        sources = list(getattr(args, "source", []) or [])
        if not sources:
            raise ValueError("site=multiscale 需要提供 --source <url_or_path...>")
        df = crawl_multiscale_training_dataset(
            sources=sources,
            cache_dir=str(getattr(args, "cache_dir", "data/cache/http")),
            timeout=float(getattr(args, "timeout", 30.0)),
            sleep_seconds=float(getattr(args, "sleep", 0.2)),
            smiles_col=getattr(args, "smiles_col", None),
            target_col=getattr(args, "target_col", None),
            d_col=getattr(args, "d_col", None),
            vmax_col=getattr(args, "vmax_col", None),
            km_col=getattr(args, "km_col", None),
            default_D=float(getattr(args, "default_D", 0.1)),
            default_Vmax=float(getattr(args, "default_Vmax", 0.5)),
            default_Km=float(getattr(args, "default_Km", 0.1)),
            normalize_smiles=bool(getattr(args, "normalize_smiles", False)),
            drop_invalid=(not bool(getattr(args, "keep_invalid", False))),
        )
    else:
        raise ValueError(f"未知站点: {site}（可用: pubchem, urlcsv, docking, multiscale）")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    usable = int(df["activity_score"].notna().sum()) if "activity_score" in df.columns else 0
    print("== Crawl done ==")
    print(f"site: {site}")
    print(f"out: {out_path}")
    print(f"rows: {len(df)}")
    if "activity_score" in df.columns:
        print(f"usable_rows(activity_score non-NaN): {usable}")
    return 0


def cmd_crawl_train(args: argparse.Namespace) -> int:
    site = str(getattr(args, "site", "pubchem") or "pubchem").lower()

    if site == "pubchem":
        cid_list = _load_cids_from_path(getattr(args, "cid_list", "") or "", cid_col=getattr(args, "cid_col", None))
        prop_fields = _parse_property_fields(getattr(args, "property_fields", "") or "")
        df = crawl_pubchem_activity_dataset(
            start_cid=int(args.start_cid),
            n=int(args.n),
            cids=(cid_list if cid_list else None),
            sleep_seconds=float(args.sleep),
            min_total_outcomes=int(args.min_total),
            min_active=int(getattr(args, "min_active", 1)),
            treat_zero_unlabeled=(not bool(getattr(args, "keep_zero", False))),
            drop_invalid=(not bool(getattr(args, "keep_invalid", False))),
            cache_dir=str(args.cache_dir),
            timeout=float(args.timeout),
            include_properties=bool(getattr(args, "include_properties", False)),
            property_fields=prop_fields,
            include_synonyms=bool(getattr(args, "include_synonyms", False)),
        )
        smiles_col = "smiles"
        target_col = "activity_score"
        env_cols: List[str] = []
    elif site == "urlcsv":
        sources = list(getattr(args, "source", []) or [])
        if not sources:
            raise ValueError("site=urlcsv 需要提供 --source <url_or_path...>")
        df = concat_tables(
            sources,
            cache_dir=str(getattr(args, "cache_dir", "data/cache/http")),
            timeout=float(getattr(args, "timeout", 30.0)),
            sleep_seconds=float(getattr(args, "sleep", 0.2)),
            headers={"User-Agent": "drug-urlcsv/1.0 (research; contact: local)"},
        )
        smiles_col = str(getattr(args, "smiles_col", "smiles") or "smiles")
        target_col = str(getattr(args, "target_col", "activity_score") or "activity_score")
        env_cols = infer_env_cols(df, smiles_col=smiles_col, target_col=target_col, env_cols=None)
    elif site in {"docking", "multiscale"}:
        raise ValueError("crawl-train 仅支持 pubchem 或 urlcsv。对接/多尺度请先 crawl 导出后在相应训练模块中训练。")
    else:
        raise ValueError(f"未知站点: {site}（可用: pubchem, urlcsv, docking, multiscale）")

    if smiles_col not in df.columns:
        raise ValueError(f"数据缺少 smiles_col: {smiles_col}")
    if target_col not in df.columns:
        raise ValueError(f"数据缺少 target_col: {target_col}")

    # keep only labeled rows
    df = df[df[target_col].notna()].copy()
    min_samples = int(getattr(args, "min_samples", 10) or 10)
    if len(df) < min_samples:
        raise ValueError(
            f"可用标注样本太少（{len(df)}），不足以训练（min_samples={min_samples}）。建议增大 n 或降低 --min-total，或调小 --min-samples。"
        )

    bundle, metrics = train_bundle(
        df,
        smiles_col=smiles_col,
        target_col=target_col,
        env_cols=env_cols,
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        featurizer_version=args.featurizer_version,
    )

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    if args.data_out:
        data_path = Path(args.data_out)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"data_out: {data_path}")

    print("== Crawl + Training done ==")
    print(f"site: {site}")
    print(f"model_out: {out_path}")
    print(f"rows_used: {len(df)}")
    print(f"n_features: {metrics['n_features']}")
    print(f"MAE:  {metrics['mae']:.6g}")
    print(f"RMSE: {metrics['rmse']:.6g}")
    print(f"R2:   {metrics['r2']:.6g}")
    return 0


def _make_x_only_drug(
    df: pd.DataFrame,
    *,
    smiles_col: str,
    env_cols: List[str],
    env_medians: Dict[str, float],
    featurizer_version: int,
    radius: int,
    n_bits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    featurizer = MoleculeFeatures(version=int(featurizer_version), radius=int(radius), n_bits=int(n_bits))
    mol_x, valids = featurizer.transform_many(df[smiles_col].astype(str).tolist())

    if env_cols:
        env_df = df[env_cols].copy()
        for c in env_cols:
            env_df[c] = pd.to_numeric(env_df[c], errors="coerce").astype(float)
            env_df[c] = env_df[c].fillna(float(env_medians.get(c, float(env_df[c].median()))))
        env_x = env_df.to_numpy(dtype=np.float32)
    else:
        env_x = np.zeros((len(df), 0), dtype=np.float32)

    x = np.concatenate([mol_x, env_x], axis=1).astype(np.float32)
    return x, valids


def cmd_self_train(args: argparse.Namespace) -> int:
    labeled = pd.read_csv(args.labeled_data)
    unlabeled = pd.read_csv(args.unlabeled_data)

    if args.smiles_col not in labeled.columns:
        raise ValueError(f"labeled_data 缺少 smiles_col: {args.smiles_col}")
    if args.smiles_col not in unlabeled.columns:
        raise ValueError(f"unlabeled_data 缺少 smiles_col: {args.smiles_col}")
    if args.target not in labeled.columns:
        raise ValueError(f"labeled_data 缺少 target 列: {args.target}")

    labeled = labeled[labeled[args.target].notna()].copy()
    if len(labeled) < int(args.min_labeled):
        raise ValueError(f"标注样本太少：{len(labeled)} < min_labeled={int(args.min_labeled)}")

    env_cols = infer_env_cols(labeled, smiles_col=args.smiles_col, target_col=args.target, env_cols=args.env_cols)
    feat_v = int(args.featurizer_version)
    radius = int(getattr(args, "radius", 2) or 2)
    n_bits = int(getattr(args, "n_bits", 2048) or 2048)

    x_l, y_l, env_medians, _feature_names, valids_l = make_xy(
        labeled,
        smiles_col=args.smiles_col,
        target_col=args.target,
        env_cols=list(env_cols),
        featurizer=MoleculeFeatures(version=feat_v, radius=radius, n_bits=n_bits),
        env_medians=None,
    )

    # drop invalid smiles for pseudo-labeling stability
    keep_l = valids_l.astype(bool)
    x_l = x_l[keep_l]
    y_l = y_l[keep_l]

    x_u, valids_u = _make_x_only_drug(
        unlabeled,
        smiles_col=args.smiles_col,
        env_cols=list(env_cols),
        env_medians=env_medians,
        featurizer_version=feat_v,
        radius=radius,
        n_bits=n_bits,
    )

    n_models = int(args.n_models)
    rng = np.random.default_rng(int(args.seed))
    preds = []
    for i in range(n_models):
        idx = rng.integers(low=0, high=len(x_l), size=len(x_l), endpoint=False)
        model = build_model(model_name=args.model, random_state=int(args.seed) + i)
        model.fit(x_l[idx], y_l[idx])
        preds.append(np.asarray(model.predict(x_u), dtype=np.float32).reshape(-1))

    pred_mat = np.stack(preds, axis=0)
    mu = pred_mat.mean(axis=0)
    sigma = pred_mat.std(axis=0)

    keep_frac = float(args.keep_frac)
    keep_frac = min(max(keep_frac, 0.0), 1.0)
    if keep_frac <= 0.0:
        raise ValueError("keep_frac 不能为0")

    # optionally exclude invalid SMILES from pseudo set
    pseudo_pool = unlabeled.copy()
    pseudo_pool["_valid_smiles"] = valids_u.astype(bool)
    pool_idx = np.where(valids_u.astype(bool))[0] if not bool(args.keep_invalid_smiles) else np.arange(len(unlabeled))
    if len(pool_idx) == 0:
        raise ValueError("unlabeled_data 中没有可用SMILES（全部无效）")

    k = max(1, int(round(len(pool_idx) * keep_frac)))
    best = pool_idx[np.argsort(sigma[pool_idx])[:k]]
    pseudo = unlabeled.iloc[best].copy()
    pseudo[args.target] = mu[best]
    pseudo["pseudo_uncertainty_std"] = sigma[best]
    pseudo["pseudo_labeled"] = True

    labeled2 = labeled.copy()
    labeled2["pseudo_labeled"] = False
    combined = pd.concat([labeled2, pseudo], axis=0, ignore_index=True)

    bundle, metrics = train_bundle(
        combined,
        smiles_col=args.smiles_col,
        target_col=args.target,
        env_cols=list(env_cols),
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        featurizer_version=feat_v,
        radius=radius,
        n_bits=n_bits,
        drop_invalid_smiles=(not bool(args.keep_invalid_smiles)),
    )

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    if args.data_out:
        data_path = Path(args.data_out)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(data_path, index=False)
        print(f"data_out: {data_path}")

    print("== Self-Training done ==")
    print(f"model_out: {out_path}")
    print(f"labeled_rows_used(valid_smiles): {len(x_l)}")
    print(f"pseudo_rows_used: {len(pseudo)} (keep_frac={keep_frac})")
    print(f"n_features: {metrics['n_features']}")
    print(f"invalid_smiles: {metrics['invalid_smiles']}")
    print(f"MAE:  {metrics['mae']:.6g}")
    print(f"RMSE: {metrics['rmse']:.6g}")
    print(f"R2:   {metrics['r2']:.6g}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    bundle: DrugModelBundle = joblib.load(args.model)
    df = pd.read_csv(args.data)

    if bundle.smiles_col not in df.columns:
        raise ValueError(f"Missing smiles_col '{bundle.smiles_col}' in data")
    if bundle.target_col not in df.columns:
        raise ValueError(f"Missing target_col '{bundle.target_col}' in data")

    x, y, _, _, valids = make_xy(
        df,
        smiles_col=bundle.smiles_col,
        target_col=bundle.target_col,
        env_cols=list(bundle.env_cols),
        featurizer=MoleculeFeatures(
            version=int(getattr(bundle, "featurizer_version", 2) or 2),
            radius=int(getattr(bundle, "radius", 2) or 2),
            n_bits=int(getattr(bundle, "n_bits", 2048) or 2048),
        ),
        env_medians=dict(bundle.env_medians),
    )

    if not bool(args.keep_invalid_smiles):
        keep = valids.astype(bool)
        x = x[keep]
        y = y[keep]

    y_pred = np.asarray(bundle.model.predict(x), dtype=float).reshape(-1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix or "drug")

    save_regression_diagnostic_plots(
        y_true=y,
        y_pred=y_pred,
        out_dir=out_dir,
        prefix=prefix,
        title=str(args.title or "drug regression"),
    )

    if args.eval_out:
        p = Path(args.eval_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"y_true": y.astype(float), "y_pred": y_pred.astype(float), "residual": (y_pred - y).astype(float)})
        out.to_csv(p, index=False)
        print(f"eval_out: {p}")

    print("== Plot done ==")
    print(f"out_dir: {out_dir}")
    print(f"prefix: {prefix}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    seed_smiles: List[str] = []

    if args.data:
        df = pd.read_csv(args.data)
        if args.smiles_col not in df.columns:
            raise ValueError(f"Missing smiles_col '{args.smiles_col}' in data")
        seed_smiles.extend(df[args.smiles_col].astype(str).tolist())

    if args.seed_smiles:
        seed_smiles.extend([str(s) for s in args.seed_smiles])

    if not seed_smiles:
        raise ValueError("No seed SMILES provided. Use --data or --seed-smiles.")

    bundle: Optional[DrugModelBundle] = None
    env_params: Dict[str, float] = {}
    if args.model:
        bundle = cast(DrugModelBundle, joblib.load(args.model))
        if bundle.env_cols:
            env_params = {c: float(bundle.env_medians.get(c, 0.0)) for c in bundle.env_cols}
    if args.param:
        env_params.update(_parse_kv(args.param))

    filters = PropertyFilters(
        min_qed=(float(args.min_qed) if args.min_qed is not None else None),
        min_mw=(float(args.min_mw) if args.min_mw is not None else None),
        max_mw=(float(args.max_mw) if args.max_mw is not None else None),
        min_logp=(float(args.min_logp) if args.min_logp is not None else None),
        max_logp=(float(args.max_logp) if args.max_logp is not None else None),
        max_hbd=(int(args.max_hbd) if args.max_hbd is not None else None),
        max_hba=(int(args.max_hba) if args.max_hba is not None else None),
        max_tpsa=(float(args.max_tpsa) if args.max_tpsa is not None else None),
    )

    score_mode = str(args.score_mode or "auto").lower()
    if score_mode == "auto":
        score_mode = "model" if bundle is not None else "qed"

    w_model = float(args.weight_model)
    w_qed = float(args.weight_qed)

    def score_fn(smiles: str) -> float:
        if not passes_filters(smiles, filters):
            return float("-inf")
        if score_mode == "model" and bundle is not None:
            return float(predict_one(bundle, smiles=smiles, env_params=env_params))
        if score_mode == "combined" and bundle is not None:
            return float(w_model * predict_one(bundle, smiles=smiles, env_params=env_params) + w_qed * default_score(smiles))
        return float(default_score(smiles))

    rng = random.Random(int(args.seed))
    gan_cfg = GanConfig(
        latent_dim=int(args.gan_latent_dim),
        hidden_dim=int(args.gan_hidden_dim),
        epochs=int(args.gan_epochs),
        batch_size=int(args.gan_batch_size),
        lr=float(args.gan_lr),
        device=str(args.gan_device),
    )

    ranked = generate_molecules(
        seed_smiles=seed_smiles,
        use_gan=bool(args.use_gan),
        gan_cfg=gan_cfg,
        radius=int(args.radius),
        n_bits=int(args.n_bits),
        gan_samples=int(args.gan_samples),
        score_fn=score_fn,
        population_size=int(args.population),
        generations=int(args.generations),
        elite_frac=float(args.elite_frac),
        mutation_rate=float(args.mutation_rate),
        crossover_rate=float(args.crossover_rate),
        rng=rng,
    )

    ranked = select_diverse_ranked(
        ranked,
        radius=int(args.radius),
        n_bits=int(args.n_bits),
        max_sim=float(args.diversity_max_sim),
    )

    top_k = min(int(args.top_k), len(ranked))
    out_rows = []
    for i, (smi, score) in enumerate(ranked[:top_k], start=1):
        row = {"rank": i, "smiles": smi, "score": float(score)}
        if args.with_props:
            props = calc_props(smi)
            if props:
                row.update(props)
        out_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    print("== Generation done ==")
    print(f"out: {out_path}")
    print(f"top_k: {top_k}")
    print(f"use_gan: {bool(args.use_gan)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drug efficacy predictor: SMILES + numeric conditions -> efficacy (regression).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sites = sub.add_parser("sites", help="List supported crawl sites")
    p_sites.set_defaults(func=cmd_sites)

    p_train = sub.add_parser("train", help="Train a model from a CSV dataset")
    p_train.add_argument("--data", required=True, help="Training CSV path")
    p_train.add_argument("--smiles-col", default="smiles", help="Column name for SMILES")
    p_train.add_argument("--target", required=True, help="Target column (e.g., efficacy)")
    p_train.add_argument("--env-cols", nargs="*", default=None, help="Optional env columns list; default = auto-detect numeric columns")
    p_train.add_argument("--model", choices=["hgb", "gbr", "rf", "ridge", "mlp"], default="hgb", help="Model type (tabular regression)")
    p_train.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="1=fingerprint only, 2=fingerprint+descriptors")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("--keep-invalid-smiles", action="store_true", help="Do NOT drop invalid SMILES rows (default: drop)")
    p_train.add_argument("--model-out", default="models/drug_model.joblib", help="Output model path")
    p_train.set_defaults(func=cmd_train)

    p_cv = sub.add_parser("cv", help="KFold cross-validation on a CSV dataset")
    p_cv.add_argument("--data", required=True, help="Training CSV path")
    p_cv.add_argument("--smiles-col", default="smiles", help="Column name for SMILES")
    p_cv.add_argument("--target", required=True, help="Target column (e.g., efficacy)")
    p_cv.add_argument("--env-cols", nargs="*", default=None, help="Optional env columns list; default = auto-detect numeric columns")
    p_cv.add_argument("--model", choices=["hgb", "gbr", "rf", "ridge", "mlp"], default="hgb", help="Model type")
    p_cv.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="1=fingerprint only, 2=fingerprint+descriptors")
    p_cv.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p_cv.add_argument("--seed", type=int, default=42, help="Random seed")
    p_cv.add_argument("--keep-invalid-smiles", action="store_true", help="Do NOT drop invalid SMILES rows (default: drop)")
    p_cv.set_defaults(func=cmd_cv)

    p_pred = sub.add_parser("predict", help="Predict for a single SMILES under given conditions")
    p_pred.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_pred.add_argument("--smiles", required=True, help="SMILES string")
    p_pred.add_argument("--param", action="append", default=None, help="Condition parameter key=value (repeatable)")
    p_pred.set_defaults(func=cmd_predict)

    p_screen = sub.add_parser("screen", help="Batch screening from candidates CSV")
    p_screen.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_screen.add_argument("--candidates", required=True, help="Candidates CSV path")
    p_screen.add_argument("--smiles-col", default="smiles", help="SMILES column in candidates CSV")
    p_screen.add_argument("--out", default="drug_predictions.csv", help="Output CSV path")
    p_screen.add_argument("--out-col", default="pred", help="Prediction column name")
    p_screen.set_defaults(func=cmd_screen)

    p_info = sub.add_parser("info", help="Print model bundle metadata")
    p_info.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_info.set_defaults(func=cmd_info)

    p_crawl = sub.add_parser("crawl", help="Crawl datasets and export a merged table")
    p_crawl.add_argument("--site", choices=["pubchem", "urlcsv", "docking", "multiscale"], default="pubchem", help="Which website/source to crawl")
    p_crawl.add_argument("--source", nargs="*", default=None, help="For site=urlcsv: one or more dataset URLs or local paths")
    p_crawl.add_argument("--cid-list", default="", help="For site=pubchem: CID list file (csv/tsv/xlsx/txt)")
    p_crawl.add_argument("--cid-col", default="", help="For --cid-list: CID column name (default: cid/CID/first column)")
    p_crawl.add_argument("--start-cid", type=int, default=1, help="Start CID")
    p_crawl.add_argument("--n", type=int, default=200, help="How many CIDs to try (start..start+n-1)")
    p_crawl.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests")
    p_crawl.add_argument("--min-total", type=int, default=5, help="Min (active+inactive) outcomes to accept label")
    p_crawl.add_argument("--min-active", type=int, default=1, help="Min Active outcomes to accept label")
    p_crawl.add_argument("--keep-zero", action="store_true", help="Keep activity_score=0 as labeled (default: treat as unlabeled)")
    p_crawl.add_argument("--keep-invalid", action="store_true", help="Keep rows with missing label/SMILES (default: drop)")
    p_crawl.add_argument("--include-properties", action="store_true", help="Include extra PubChem property columns")
    p_crawl.add_argument(
        "--property-fields",
        default="",
        help="Comma-separated PubChem property fields (e.g. MolecularWeight,XLogP,TopologicalPolarSurfaceArea)",
    )
    p_crawl.add_argument("--include-synonyms", action="store_true", help="Include PubChem synonyms columns")
    p_crawl.add_argument("--timeout", type=float, default=30.0, help="Request timeout seconds")
    p_crawl.add_argument("--cache-dir", default="data/cache/pubchem", help="Cache directory")
    p_crawl.add_argument("--out", default="data/pubchem_activity.csv", help="Output CSV path")
    p_crawl.add_argument("--normalize-smiles", action="store_true", help="Normalize SMILES using RDKit when available")
    p_crawl.add_argument("--ligand-smiles-col", default="", help="For site=docking: ligand SMILES column")
    p_crawl.add_argument("--protein-seq-col", default="", help="For site=docking: protein sequence column")
    p_crawl.add_argument("--protein-pdb-col", default="", help="For site=docking: PDB ID column")
    p_crawl.add_argument("--binding-score-col", default="", help="For site=docking: binding score column")
    p_crawl.add_argument("--pocket-path-col", default="", help="For site=docking: pocket path column")
    p_crawl.add_argument("--smiles-col", default="", help="For site=multiscale: SMILES column")
    p_crawl.add_argument("--target-col", default="", help="For site=multiscale: target column")
    p_crawl.add_argument("--d-col", default="", help="For site=multiscale: diffusion (D) column")
    p_crawl.add_argument("--vmax-col", default="", help="For site=multiscale: Vmax column")
    p_crawl.add_argument("--km-col", default="", help="For site=multiscale: Km column")
    p_crawl.add_argument("--default-D", type=float, default=0.1, help="For site=multiscale: default D when missing")
    p_crawl.add_argument("--default-Vmax", type=float, default=0.5, help="For site=multiscale: default Vmax when missing")
    p_crawl.add_argument("--default-Km", type=float, default=0.1, help="For site=multiscale: default Km when missing")
    p_crawl.set_defaults(func=cmd_crawl)

    p_ct = sub.add_parser("crawl-train", help="Crawl PubChem then train a model on activity_score")
    p_ct.add_argument("--site", choices=["pubchem", "urlcsv"], default="pubchem", help="Which website/source to crawl")
    p_ct.add_argument("--source", nargs="*", default=None, help="For site=urlcsv: one or more dataset URLs or local paths")
    p_ct.add_argument("--cid-list", default="", help="For site=pubchem: CID list file (csv/tsv/xlsx/txt)")
    p_ct.add_argument("--cid-col", default="", help="For --cid-list: CID column name (default: cid/CID/first column)")
    p_ct.add_argument("--smiles-col", default="smiles", help="For site=urlcsv: SMILES column")
    p_ct.add_argument("--target-col", default="activity_score", help="For site=urlcsv: target column")
    p_ct.add_argument("--start-cid", type=int, default=1, help="Start CID")
    p_ct.add_argument("--n", type=int, default=500, help="How many CIDs to try")
    p_ct.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests")
    p_ct.add_argument("--min-total", type=int, default=5, help="Min (active+inactive) outcomes to accept label")
    p_ct.add_argument("--min-active", type=int, default=1, help="Min Active outcomes to accept label")
    p_ct.add_argument("--keep-zero", action="store_true", help="Keep activity_score=0 as labeled (default: treat as unlabeled)")
    p_ct.add_argument("--keep-invalid", action="store_true", help="Keep rows with missing label/SMILES (default: drop)")
    p_ct.add_argument("--include-properties", action="store_true", help="Include extra PubChem property columns")
    p_ct.add_argument(
        "--property-fields",
        default="",
        help="Comma-separated PubChem property fields (e.g. MolecularWeight,XLogP,TopologicalPolarSurfaceArea)",
    )
    p_ct.add_argument("--include-synonyms", action="store_true", help="Include PubChem synonyms columns")
    p_ct.add_argument("--min-samples", type=int, default=10, help="Min labeled samples required to start training")
    p_ct.add_argument("--timeout", type=float, default=30.0, help="Request timeout seconds")
    p_ct.add_argument("--cache-dir", default="data/cache/pubchem", help="Cache directory")
    p_ct.add_argument("--data-out", default="", help="Optional: save labeled dataset CSV")
    p_ct.add_argument("--model", choices=["hgb", "gbr", "rf", "ridge", "mlp"], default="hgb", help="Model type")
    p_ct.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="1=fingerprint only, 2=fingerprint+descriptors")
    p_ct.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    p_ct.add_argument("--seed", type=int, default=42, help="Random seed")
    p_ct.add_argument("--model-out", default="models/drug_pubchem_activity.joblib", help="Output model path")
    p_ct.set_defaults(func=cmd_crawl_train)

    p_st = sub.add_parser("self-train", help="Self-training via pseudo-labeling on unlabeled data")
    p_st.add_argument("--labeled-data", required=True, help="Labeled training CSV path")
    p_st.add_argument("--unlabeled-data", required=True, help="Unlabeled CSV path (no target required)")
    p_st.add_argument("--smiles-col", default="smiles", help="SMILES column")
    p_st.add_argument("--target", required=True, help="Target column name (in labeled_data; will be created for pseudo labels)")
    p_st.add_argument("--env-cols", nargs="*", default=None, help="Optional env columns list; default = auto-detect numeric columns from labeled")
    p_st.add_argument("--min-labeled", type=int, default=20, help="Min labeled rows required")
    p_st.add_argument("--n-models", type=int, default=5, help="Bootstrap ensemble size for uncertainty")
    p_st.add_argument("--keep-frac", type=float, default=0.5, help="Keep this fraction of lowest-uncertainty pseudo labels")
    p_st.add_argument("--keep-invalid-smiles", action="store_true", help="Do NOT drop invalid SMILES rows")
    p_st.add_argument("--model", choices=["hgb", "gbr", "rf", "ridge", "mlp"], default="hgb", help="Model type")
    p_st.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="1=fingerprint only, 2=fingerprint+descriptors")
    p_st.add_argument("--radius", type=int, default=2, help="Morgan radius")
    p_st.add_argument("--n-bits", type=int, default=2048, help="Morgan nBits")
    p_st.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    p_st.add_argument("--seed", type=int, default=42, help="Random seed")
    p_st.add_argument("--data-out", default="", help="Optional: save combined dataset CSV")
    p_st.add_argument("--model-out", default="models/drug_self_trained.joblib", help="Output model path")
    p_st.set_defaults(func=cmd_self_train)

    p_gen = sub.add_parser("generate", help="Generate molecules via GAN + evolutionary algorithm")
    p_gen.add_argument("--data", default="", help="Seed CSV with smiles column (optional if --seed-smiles provided)")
    p_gen.add_argument("--smiles-col", default="smiles", help="SMILES column name in --data")
    p_gen.add_argument("--seed-smiles", nargs="*", default=None, help="Additional seed SMILES strings")
    p_gen.add_argument("--model", default="", help="Optional model bundle for scoring (.joblib)")
    p_gen.add_argument("--param", action="append", default=None, help="Condition parameter key=value (repeatable; used with model scoring)")
    p_gen.add_argument("--score-mode", choices=["auto", "qed", "model", "combined"], default="auto", help="Scoring mode")
    p_gen.add_argument("--weight-model", type=float, default=1.0, help="Weight for model score (combined mode)")
    p_gen.add_argument("--weight-qed", type=float, default=1.0, help="Weight for QED score (combined mode)")
    p_gen.add_argument("--use-gan", action="store_true", help="Use GAN to expand initial seed pool")
    p_gen.add_argument("--gan-samples", type=int, default=256, help="Number of GAN samples to draw")
    p_gen.add_argument("--gan-latent-dim", type=int, default=64, help="GAN latent dimension")
    p_gen.add_argument("--gan-hidden-dim", type=int, default=256, help="GAN hidden dimension")
    p_gen.add_argument("--gan-epochs", type=int, default=200, help="GAN training epochs")
    p_gen.add_argument("--gan-batch-size", type=int, default=128, help="GAN batch size")
    p_gen.add_argument("--gan-lr", type=float, default=2e-4, help="GAN learning rate")
    p_gen.add_argument("--gan-device", default="cpu", help="GAN device (cpu/cuda)")
    p_gen.add_argument("--population", type=int, default=200, help="Population size for evolution")
    p_gen.add_argument("--generations", type=int, default=20, help="Number of evolution generations")
    p_gen.add_argument("--elite-frac", type=float, default=0.2, help="Elite fraction (0-1)")
    p_gen.add_argument("--mutation-rate", type=float, default=0.4, help="Mutation probability")
    p_gen.add_argument("--crossover-rate", type=float, default=0.6, help="Crossover probability")
    p_gen.add_argument("--radius", type=int, default=2, help="Morgan radius")
    p_gen.add_argument("--n-bits", type=int, default=2048, help="Morgan nBits")
    p_gen.add_argument("--diversity-max-sim", type=float, default=0.9, help="Max Tanimoto similarity in output (1.0 disables)")
    p_gen.add_argument("--min-qed", type=float, default=None, help="Filter: min QED")
    p_gen.add_argument("--min-mw", type=float, default=None, help="Filter: min molecular weight")
    p_gen.add_argument("--max-mw", type=float, default=None, help="Filter: max molecular weight")
    p_gen.add_argument("--min-logp", type=float, default=None, help="Filter: min LogP")
    p_gen.add_argument("--max-logp", type=float, default=None, help="Filter: max LogP")
    p_gen.add_argument("--max-hbd", type=int, default=None, help="Filter: max HBD")
    p_gen.add_argument("--max-hba", type=int, default=None, help="Filter: max HBA")
    p_gen.add_argument("--max-tpsa", type=float, default=None, help="Filter: max TPSA")
    p_gen.add_argument("--top-k", type=int, default=50, help="Top K molecules to export")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--with-props", action="store_true", help="Include properties (qed/mw/logp/hbd/hba/tpsa) in output")
    p_gen.add_argument("--out", default="generated_molecules.csv", help="Output CSV path")
    p_gen.set_defaults(func=cmd_generate)

    p_plot = sub.add_parser("plot", help="Generate regression diagnostic plots on a dataset")
    p_plot.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_plot.add_argument("--data", required=True, help="CSV path containing features + target")
    p_plot.add_argument("--out-dir", default="plots", help="Output directory")
    p_plot.add_argument("--prefix", default="drug", help="Filename prefix")
    p_plot.add_argument("--title", default="", help="Optional plot title")
    p_plot.add_argument("--keep-invalid-smiles", action="store_true", help="Do NOT drop invalid SMILES rows")
    p_plot.add_argument("--eval-out", default="", help="Optional: export eval CSV (y_true,y_pred,residual)")
    p_plot.set_defaults(func=cmd_plot)

    p_suggest = sub.add_parser("suggest-env", help="Suggest environment params via differential evolution")
    p_suggest.add_argument("--model", required=True, help="Model bundle path (.joblib or .pt)")
    p_suggest.add_argument("--smiles", required=True, help="SMILES string to optimize for")
    p_suggest.add_argument("--bounds", default=None, help="Optional comma-separated bounds for env vars in order: low:high,low:high")
    p_suggest.add_argument("--use-cuda", action="store_true", help="Use CUDA for torch models if available")
    p_suggest.set_defaults(func=cmd_suggest_env)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
