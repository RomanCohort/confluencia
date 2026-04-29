"""Compatibility shim: forward to refactored CLI in `src.cli.epitope_cli`.

This file preserves the original entrypoint path so existing scripts
and shortcuts (e.g., running this file directly) keep working. The
actual implementation lives in `src.cli.epitope_cli`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.cli.epitope_cli import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())


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


def cmd_train(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)

    env_cols = infer_env_cols(
        df,
        sequence_col=args.sequence_col,
        target_col=args.target,
        env_cols=args.env_cols,
    )

    bundle, metrics = train_bundle(
        df,
        sequence_col=args.sequence_col,
        target_col=args.target,
        env_cols=env_cols,
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        featurizer_version=args.featurizer_version,
    )

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    print("== Training done ==")
    print(f"model_out: {out_path}")
    print(f"sequence_col: {bundle.sequence_col}")
    print(f"target_col: {bundle.target_col}")
    print(f"env_cols: {bundle.env_cols}")
    print(f"n_features: {metrics['n_features']}")
    print(f"MAE:  {metrics['mae']:.6g}")
    print(f"RMSE: {metrics['rmse']:.6g}")
    print(f"R2:   {metrics['r2']:.6g}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)
    env_cols = infer_env_cols(df, sequence_col=args.sequence_col, target_col=args.target, env_cols=args.env_cols)

    x, y, env_medians, feature_names = make_xy(
        df,
        sequence_col=args.sequence_col,
        target_col=args.target,
        env_cols=list(env_cols),
        featurizer=SequenceFeatures(version=int(args.featurizer_version)),
        env_medians=None,
    )

    # build base estimator
    base = build_model(model_name=args.model, random_state=int(args.seed))

    # default param grids for common models
    param_grids = {
        "hgb": {"l2_regularization": [0.0, 1e-4, 1e-3], "max_iter": [100, 300]},
        "gbr": {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1]},
        "rf": {"n_estimators": [100, 300, 500], "max_depth": [None, 8, 16]},
        "mlp": {"mlp__alpha": [1e-4, 1e-3], "mlp__hidden_layer_sizes": [(128, 64), (256, 128)]},
        "sgd": {"sgd__alpha": [1e-4, 1e-3], "sgd__l1_ratio": [0.0, 0.15, 0.5]},
    }

    grid = param_grids.get(args.model, {})
    if args.param_grid:
        # user-provided JSON file with param grid
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
        sequence_col=args.sequence_col,
        target_col=args.target,
        env_cols=args.env_cols,
    )

    report = cross_validate(
        df,
        sequence_col=args.sequence_col,
        target_col=args.target,
        env_cols=env_cols,
        model_name=args.model,
        n_splits=args.n_splits,
        random_state=args.seed,
        featurizer_version=args.featurizer_version,
    )

    summary = cast(Dict[str, Any], report["summary"])
    print("== Cross Validation ==")
    print(f"data: {args.data}")
    print(f"model: {args.model}")
    print(f"n_splits: {args.n_splits}")
    print(f"n_samples: {summary['n_samples']}")
    print(f"n_features: {summary['n_features']}")
    print(f"MAE:  {summary['mae_mean']:.6g} ± {summary['mae_std']:.6g}")
    print(f"RMSE: {summary['rmse_mean']:.6g} ± {summary['rmse_std']:.6g}")
    print(f"R2:   {summary['r2_mean']:.6g} ± {summary['r2_std']:.6g}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    bundle: EpitopeModelBundle = joblib.load(args.model)
    env_params = _parse_kv(args.param)

    y = predict_one(bundle, sequence=args.sequence, env_params=env_params)
    print("== Prediction ==")
    print(f"sequence: {args.sequence}")
    print(f"pred: {y:.6g}")
    if bundle.env_cols:
        resolved = {c: float(env_params.get(c, bundle.env_medians.get(c, 0.0))) for c in bundle.env_cols}
        print(f"env: {resolved}")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    bundle: EpitopeModelBundle = joblib.load(args.model)
    df = pd.read_csv(args.candidates)

    if args.sequence_col not in df.columns:
        raise ValueError(f"Missing sequence_col '{args.sequence_col}' in candidates CSV")

    # Build env param dataframe with medians fill
    for c in bundle.env_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].astype(float)
        df[c] = df[c].fillna(bundle.env_medians.get(c, 0.0))

    x = _make_x_only_epitope(
        df,
        sequence_col=args.sequence_col,
        env_cols=list(bundle.env_cols),
        env_medians=bundle.env_medians,
        featurizer_version=int(getattr(bundle, "featurizer_version", 1)),
    )

    preds = np.empty((x.shape[0],), dtype=np.float32)
    chunk = 10000
    for start in range(0, x.shape[0], chunk):
        end = min(x.shape[0], start + chunk)
        preds[start:end] = np.asarray(bundle.model.predict(x[start:end]), dtype=np.float32).reshape(-1)

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
    bundle: EpitopeModelBundle = joblib.load(args.model)
    d = asdict(bundle)
    # model object is not JSON-serializable; summarize it
    d["model"] = str(type(bundle.model))
    print(d)
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    site = str(getattr(args, "site", "urlcsv") or "urlcsv").lower()
    sources = list(getattr(args, "source", []) or [])
    if not sources:
        raise ValueError("请提供 --source <url_or_path...>")

    if site == "urlcsv":
        df = crawl_epitope_csv_datasets(
            sources,
            cache_dir=str(args.cache_dir),
            timeout=float(args.timeout),
            sleep_seconds=float(args.sleep),
            sequence_col=str(getattr(args, "sequence_col", "") or "").strip() or None,
            min_len=int(getattr(args, "min_len", 8)),
            max_len=int(getattr(args, "max_len", 25)),
            allow_x=bool(getattr(args, "allow_x", False)),
            drop_duplicates=(not bool(getattr(args, "keep_duplicates", False))),
        )
    elif site in {"fasta", "uniprot"}:
        if site == "uniprot":
            sources = [f"uniprot:{s}" for s in sources]
        df = crawl_epitope_fasta_sources(
            sources,
            cache_dir=str(args.cache_dir),
            timeout=float(args.timeout),
            sleep_seconds=float(args.sleep),
            min_len=int(getattr(args, "min_len", 8)),
            max_len=int(getattr(args, "max_len", 25)),
            allow_x=bool(getattr(args, "allow_x", False)),
            drop_duplicates=(not bool(getattr(args, "keep_duplicates", False))),
        )
    else:
        raise ValueError(f"未知站点: {site}（可用: urlcsv, fasta, uniprot）")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("== Crawl done ==")
    print(f"site: {site}")
    print(f"out: {out_path}")
    print(f"rows: {len(df)}")
    return 0


def cmd_suggest_env(args: argparse.Namespace) -> int:
    bundle: EpitopeModelBundle = joblib.load(args.model)
    seq = args.sequence
    # parse bounds: expected as comma-separated low:high pairs, or None to auto
    bounds_input = args.bounds
    n = len(bundle.env_cols) if bundle.env_cols else 0
    if n == 0:
        raise ValueError("模型没有环境变量可优化")

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
        # auto: use medians +/- 50% or [-1,1] when median==0
        env_bounds = []
        for c in bundle.env_cols:
            med = float(bundle.env_medians.get(c, 0.0))
            if med == 0.0:
                env_bounds.append((-1.0, 1.0))
            else:
                env_bounds.append((med * 0.5, med * 1.5))

    # run optimization
    from src.epitope.predictor import suggest_env_by_de_epitope

    best_env, best_val = suggest_env_by_de_epitope(bundle, sequence=seq, env_bounds=env_bounds)
    mapped = {c: float(v) for c, v in zip(bundle.env_cols, best_env.tolist())}
    print("== DE Suggestion Result ==")
    print(f"sequence: {seq}")
    print(f"pred: {best_val:.6g}")
    print(f"env: {mapped}")
    return 0


def cmd_crawl_train(args: argparse.Namespace) -> int:
    site = str(getattr(args, "site", "urlcsv") or "urlcsv").lower()
    if site != "urlcsv":
        raise ValueError(f"未知站点: {site}（当前仅支持: urlcsv）")

    df = crawl_epitope_csv_datasets(
        args.source,
        cache_dir=str(args.cache_dir),
        timeout=float(args.timeout),
        sleep_seconds=float(args.sleep),
    )

    if args.target not in df.columns:
        raise ValueError(f"Crawl得到的数据缺少 target 列: {args.target}")
    if args.sequence_col not in df.columns:
        raise ValueError(f"Crawl得到的数据缺少 sequence_col 列: {args.sequence_col}")

    df = df[df[args.target].notna()].copy()
    min_samples = int(getattr(args, "min_samples", 10) or 10)
    if len(df) < min_samples:
        raise ValueError(f"可用标注样本太少（{len(df)}），不足以训练（min_samples={min_samples}）。")

    env_cols = infer_env_cols(df, sequence_col=args.sequence_col, target_col=args.target, env_cols=args.env_cols)

    bundle, metrics = train_bundle(
        df,
        sequence_col=args.sequence_col,
        target_col=args.target,
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


def _make_x_only_epitope(
    df: pd.DataFrame,
    *,
    sequence_col: str,
    env_cols: List[str],
    env_medians: Dict[str, float],
    featurizer_version: int,
) -> np.ndarray:
    featurizer = SequenceFeatures(version=int(featurizer_version))
    seq_x = featurizer.transform_many(df[sequence_col].astype(str).tolist())

    if env_cols:
        env_df = df[env_cols].copy()
        for c in env_cols:
            env_df[c] = pd.to_numeric(env_df[c], errors="coerce").astype(float)
            env_df[c] = env_df[c].fillna(float(env_medians.get(c, float(env_df[c].median()))))
        env_x = env_df.to_numpy(dtype=np.float32)
    else:
        env_x = np.zeros((len(df), 0), dtype=np.float32)

    return np.concatenate([seq_x, env_x], axis=1).astype(np.float32)


def cmd_self_train(args: argparse.Namespace) -> int:
    labeled = pd.read_csv(args.labeled_data)
    unlabeled = pd.read_csv(args.unlabeled_data)

    if args.sequence_col not in labeled.columns:
        raise ValueError(f"labeled_data 缺少 sequence_col: {args.sequence_col}")
    if args.sequence_col not in unlabeled.columns:
        raise ValueError(f"unlabeled_data 缺少 sequence_col: {args.sequence_col}")
    if args.target not in labeled.columns:
        raise ValueError(f"labeled_data 缺少 target 列: {args.target}")

    labeled = labeled[labeled[args.target].notna()].copy()
    if len(labeled) < int(args.min_labeled):
        raise ValueError(f"标注样本太少：{len(labeled)} < min_labeled={int(args.min_labeled)}")

    env_cols = infer_env_cols(labeled, sequence_col=args.sequence_col, target_col=args.target, env_cols=args.env_cols)
    feat_v = int(args.featurizer_version)

    x_l, y_l, env_medians, _feature_names = make_xy(
        labeled,
        sequence_col=args.sequence_col,
        target_col=args.target,
        env_cols=list(env_cols),
        featurizer=SequenceFeatures(version=feat_v),
        env_medians=None,
    )
    x_u = _make_x_only_epitope(
        unlabeled,
        sequence_col=args.sequence_col,
        env_cols=list(env_cols),
        env_medians=env_medians,
        featurizer_version=feat_v,
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

    k = max(1, int(round(len(unlabeled) * keep_frac)))
    keep_idx = np.argsort(sigma)[:k]
    pseudo = unlabeled.iloc[keep_idx].copy()
    pseudo[args.target] = mu[keep_idx]
    pseudo["pseudo_uncertainty_std"] = sigma[keep_idx]
    pseudo["pseudo_labeled"] = True

    labeled2 = labeled.copy()
    labeled2["pseudo_labeled"] = False
    combined = pd.concat([labeled2, pseudo], axis=0, ignore_index=True)

    bundle, metrics = train_bundle(
        combined,
        sequence_col=args.sequence_col,
        target_col=args.target,
        env_cols=list(env_cols),
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        featurizer_version=feat_v,
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
    print(f"labeled_rows: {len(labeled)}")
    print(f"pseudo_rows_used: {len(pseudo)} (keep_frac={keep_frac})")
    print(f"n_features: {metrics['n_features']}")
    print(f"MAE:  {metrics['mae']:.6g}")
    print(f"RMSE: {metrics['rmse']:.6g}")
    print(f"R2:   {metrics['r2']:.6g}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    bundle: EpitopeModelBundle = joblib.load(args.model)
    df = pd.read_csv(args.data)

    if bundle.sequence_col not in df.columns:
        raise ValueError(f"Missing sequence_col '{bundle.sequence_col}' in data")
    if bundle.target_col not in df.columns:
        raise ValueError(f"Missing target_col '{bundle.target_col}' in data")

    x, y, _, _ = make_xy(
        df,
        sequence_col=bundle.sequence_col,
        target_col=bundle.target_col,
        env_cols=list(bundle.env_cols),
        featurizer=SequenceFeatures(version=int(getattr(bundle, "featurizer_version", 1) or 1)),
        env_medians=dict(bundle.env_medians),
    )
    y_pred = np.asarray(bundle.model.predict(x), dtype=float).reshape(-1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix or "epitope")

    save_regression_diagnostic_plots(
        y_true=y,
        y_pred=y_pred,
        out_dir=out_dir,
        prefix=prefix,
        title=str(args.title or "epitope regression"),
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Epitope virtual screening predictor: sequence + experimental params -> efficacy (regression).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sites = sub.add_parser("sites", help="List supported crawl sites")
    p_sites.set_defaults(func=cmd_sites)

    p_train = sub.add_parser("train", help="Train a model from a CSV dataset")
    p_train.add_argument("--data", required=True, help="Training CSV path")
    p_train.add_argument("--sequence-col", default="sequence", help="Column name for epitope sequence")
    p_train.add_argument("--target", required=True, help="Target column (e.g., fluorescence, killing_rate)")
    p_train.add_argument(
        "--env-cols",
        nargs="*",
        default=None,
        help="Optional env columns list; default = auto-detect numeric columns",
    )
    p_train.add_argument(
        "--model",
        choices=["hgb", "gbr", "rf", "mlp", "sgd"],
        default="hgb",
        help="Model type: hgb/gbr/rf/mlp/sgd (sgd适合大规模；hgb默认稳健)",
    )
    p_train.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="Sequence feature version")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("--model-out", default="models/epitope_model.joblib", help="Output model path")
    p_train.set_defaults(func=cmd_train)

    p_tune = sub.add_parser("tune", help="Hyperparameter tuning (Grid or Randomized search)")
    p_tune.add_argument("--data", required=True, help="Training CSV path")
    p_tune.add_argument("--sequence-col", default="sequence", help="Column name for epitope sequence")
    p_tune.add_argument("--target", required=True, help="Target column (e.g., fluorescence, killing_rate)")
    p_tune.add_argument(
        "--env-cols",
        nargs="*",
        default=None,
        help="Optional env columns list; default = auto-detect numeric columns",
    )
    p_tune.add_argument(
        "--model",
        choices=["hgb", "gbr", "rf", "mlp", "sgd"],
        default="hgb",
        help="Model type: hgb/gbr/rf/mlp/sgd",
    )
    p_tune.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="Sequence feature version")
    p_tune.add_argument("--seed", type=int, default=42, help="Random seed")
    p_tune.add_argument("--model-out", default="models/epitope_model_tuned.joblib", help="Output model path")
    p_tune.add_argument("--strategy", choices=["grid", "random"], default="grid", help="Search strategy")
    p_tune.add_argument("--n-iter", default=20, help="Number of iterations for randomized search (ignored for grid)")
    p_tune.add_argument("--cv", default=5, help="CV folds")
    p_tune.add_argument("--param-grid", default="", help="Optional JSON file with parameter grid/objective")
    p_tune.set_defaults(func=cmd_tune)

    p_cv = sub.add_parser("cv", help="KFold cross-validation on a CSV dataset")
    p_cv.add_argument("--data", required=True, help="Training CSV path")
    p_cv.add_argument("--sequence-col", default="sequence", help="Column name for epitope sequence")
    p_cv.add_argument("--target", required=True, help="Target column (e.g., fluorescence, killing_rate)")
    p_cv.add_argument(
        "--env-cols",
        nargs="*",
        default=None,
        help="Optional env columns list; default = auto-detect numeric columns",
    )
    p_cv.add_argument(
        "--model",
        choices=["hgb", "gbr", "rf", "mlp", "sgd"],
        default="hgb",
        help="Model type: hgb/gbr/rf/mlp/sgd (sgd适合大规模；hgb默认稳健)",
    )
    p_cv.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="Sequence feature version")
    p_cv.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p_cv.add_argument("--seed", type=int, default=42, help="Random seed")
    p_cv.set_defaults(func=cmd_cv)

    p_pred = sub.add_parser("predict", help="Predict for a single sequence under given env params")
    p_pred.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_pred.add_argument("--sequence", required=True, help="Epitope sequence")
    p_pred.add_argument(
        "--param",
        action="append",
        default=None,
        help="Experimental parameter key=value (repeatable)",
    )
    p_pred.set_defaults(func=cmd_predict)

    p_suggest = sub.add_parser("suggest-env", help="Suggest environment params via differential evolution")
    p_suggest.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_suggest.add_argument("--sequence", required=True, help="Epitope sequence")
    p_suggest.add_argument("--bounds", default=None, help="Optional comma-separated bounds for env vars in order: low:high,low:high")
    p_suggest.set_defaults(func=cmd_suggest_env)

    p_screen = sub.add_parser("screen", help="Batch screening from candidates CSV")
    p_screen.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_screen.add_argument("--candidates", required=True, help="Candidates CSV path")
    p_screen.add_argument("--sequence-col", default="sequence", help="Sequence column in candidates CSV")
    p_screen.add_argument("--out", default="predictions.csv", help="Output CSV path")
    p_screen.add_argument("--out-col", default="pred", help="Prediction column name")
    p_screen.set_defaults(func=cmd_screen)

    p_info = sub.add_parser("info", help="Print model bundle metadata")
    p_info.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_info.set_defaults(func=cmd_info)

    p_crawl = sub.add_parser("crawl", help="Fetch user-provided datasets (URL/path) and export")
    p_crawl.add_argument("--site", choices=["urlcsv", "fasta", "uniprot"], default="urlcsv", help="Which website/source to crawl")
    p_crawl.add_argument("--source", nargs="+", required=True, help="One or more dataset URLs or local paths (or accession list for uniprot)")
    p_crawl.add_argument("--cache-dir", default="data/cache/epitope", help="Cache directory")
    p_crawl.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests")
    p_crawl.add_argument("--timeout", type=float, default=30.0, help="Request timeout seconds")
    p_crawl.add_argument("--sequence-col", default="sequence", help="Column name for epitope sequence (urlcsv)")
    p_crawl.add_argument("--min-len", type=int, default=8, help="Minimum sequence length")
    p_crawl.add_argument("--max-len", type=int, default=25, help="Maximum sequence length")
    p_crawl.add_argument("--allow-x", action="store_true", help="Allow 'X' in sequences")
    p_crawl.add_argument("--keep-duplicates", action="store_true", help="Do NOT drop duplicate sequences")
    p_crawl.add_argument("--out", default="data/epitope_crawled.csv", help="Output CSV path")
    p_crawl.set_defaults(func=cmd_crawl)

    p_ct = sub.add_parser("crawl-train", help="Fetch datasets (URL/path) then train a model")
    p_ct.add_argument("--site", choices=["urlcsv"], default="urlcsv", help="Which website/source to crawl")
    p_ct.add_argument("--source", nargs="+", required=True, help="One or more dataset URLs or local paths")
    p_ct.add_argument("--sequence-col", default="sequence", help="Column name for epitope sequence")
    p_ct.add_argument("--target", required=True, help="Target column")
    p_ct.add_argument("--env-cols", nargs="*", default=None, help="Optional env columns list; default = auto-detect numeric columns")
    p_ct.add_argument("--min-samples", type=int, default=10, help="Min labeled samples required to start training")
    p_ct.add_argument("--cache-dir", default="data/cache/epitope", help="Cache directory")
    p_ct.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests")
    p_ct.add_argument("--timeout", type=float, default=30.0, help="Request timeout seconds")
    p_ct.add_argument("--data-out", default="", help="Optional: save labeled dataset CSV")
    p_ct.add_argument(
        "--model",
        choices=["hgb", "gbr", "rf", "mlp", "sgd"],
        default="hgb",
        help="Model type: hgb/gbr/rf/mlp/sgd (sgd适合大规模；hgb默认稳健)",
    )
    p_ct.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="Sequence feature version")
    p_ct.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    p_ct.add_argument("--seed", type=int, default=42, help="Random seed")
    p_ct.add_argument("--model-out", default="models/epitope_crawled.joblib", help="Output model path")
    p_ct.set_defaults(func=cmd_crawl_train)

    p_st = sub.add_parser("self-train", help="Self-training via pseudo-labeling on unlabeled data")
    p_st.add_argument("--labeled-data", required=True, help="Labeled training CSV path")
    p_st.add_argument("--unlabeled-data", required=True, help="Unlabeled CSV path (no target required)")
    p_st.add_argument("--sequence-col", default="sequence", help="Sequence column")
    p_st.add_argument("--target", required=True, help="Target column name (in labeled_data; will be created for pseudo labels)")
    p_st.add_argument("--env-cols", nargs="*", default=None, help="Optional env columns list; default = auto-detect numeric columns from labeled")
    p_st.add_argument("--min-labeled", type=int, default=20, help="Min labeled rows required")
    p_st.add_argument("--n-models", type=int, default=5, help="Bootstrap ensemble size for uncertainty")
    p_st.add_argument("--keep-frac", type=float, default=0.5, help="Keep this fraction of lowest-uncertainty pseudo labels")
    p_st.add_argument(
        "--model",
        choices=["hgb", "gbr", "rf", "mlp", "sgd"],
        default="hgb",
        help="Model type: hgb/gbr/rf/mlp/sgd (sgd适合大规模；hgb默认稳健)",
    )
    p_st.add_argument("--featurizer-version", type=int, choices=[1, 2], default=2, help="Sequence feature version")
    p_st.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    p_st.add_argument("--seed", type=int, default=42, help="Random seed")
    p_st.add_argument("--data-out", default="", help="Optional: save combined dataset CSV")
    p_st.add_argument("--model-out", default="models/epitope_self_trained.joblib", help="Output model path")
    p_st.set_defaults(func=cmd_self_train)

    p_plot = sub.add_parser("plot", help="Generate regression diagnostic plots on a dataset")
    p_plot.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    p_plot.add_argument("--data", required=True, help="CSV path containing features + target")
    p_plot.add_argument("--out-dir", default="plots", help="Output directory")
    p_plot.add_argument("--prefix", default="epitope", help="Filename prefix")
    p_plot.add_argument("--title", default="", help="Optional plot title")
    p_plot.add_argument("--eval-out", default="", help="Optional: export eval CSV (y_true,y_pred,residual)")
    p_plot.set_defaults(func=cmd_plot)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
