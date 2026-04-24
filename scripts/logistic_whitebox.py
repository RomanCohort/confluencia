#!/usr/bin/env python3
"""训练与白箱化解释（逻辑回归）工具

用法示例:
  python scripts/logistic_whitebox.py train --csv data/example_drug.csv --target label --out models/logistic_whitebox.joblib
  python scripts/logistic_whitebox.py explain --csv data/example_drug.csv --index 0 --model models/logistic_whitebox.joblib

功能:
 - 训练二分类逻辑回归并保存模型/Scaler
 - 绘制系数条形图、ROC曲线
 - 对单条样本给出每个特征对预测概率的贡献（白箱化）并绘图
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def train(csv_path, target, features, test_size, random_state, out_model,
          C=1.0, penalty='l2', class_weight=None, max_iter=1000,
          solver=None, l1_ratio=None, tol=1e-4, fit_intercept=True, warm_start=False, random_state_lr=None):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in CSV")

    if features:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"features not found in CSV: {missing}")
        X = df[features].copy()
    else:
        X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # select solver if not provided
    if solver is None:
        if penalty == 'elasticnet' or (l1_ratio is not None and penalty == 'elasticnet'):
            chosen_solver = 'saga'
        elif penalty == 'l1':
            chosen_solver = 'liblinear'
        elif penalty == 'none':
            chosen_solver = 'lbfgs'
        else:
            chosen_solver = 'liblinear' if penalty in ('l1','l2') else 'lbfgs'
    else:
        chosen_solver = solver

    clf_kwargs = {
        'C': C,
        'penalty': penalty if penalty != 'none' else 'none',
        'class_weight': class_weight,
        'solver': chosen_solver,
        'max_iter': int(max_iter),
        'tol': float(tol),
        'fit_intercept': bool(fit_intercept),
        'warm_start': bool(warm_start),
    }
    if penalty == 'elasticnet' and l1_ratio is not None:
        clf_kwargs['l1_ratio'] = float(l1_ratio)
    # sklearn may warn if solver doesn't support a param; let it raise if incompatible
    clf = LogisticRegression(**clf_kwargs)


def train_tree(csv_path, target, features, test_size, random_state, out_model, max_depth=None, min_samples_leaf=1):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in CSV")
    if features:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"features not found in CSV: {missing}")
        X = df[features].copy()
    else:
        X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target]

    is_class = set(np.unique(y)).issubset({0,1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if is_class:
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    else:
        from sklearn.tree import DecisionTreeRegressor
        clf = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)

    clf.fit(X_train_s, y_train)

    try:
        if is_class:
            y_prob = clf.predict_proba(X_test_s)[:,1]
            y_pred = clf.predict(X_test_s)
        else:
            y_prob = None
            y_pred = clf.predict(X_test_s)
    except Exception:
        y_prob = None
        y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred) if is_class else float('nan')
    try:
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float('nan')
    except Exception:
        auc = float('nan')

    print(f"Tree - Accuracy: {acc:.4f}")
    print(f"Tree - ROC AUC: {auc:.4f}")

    # save model + scaler + feature names
    ensure_dir(out_model)
    joblib.dump({'model': clf, 'scaler': scaler, 'features': list(X.columns), 'is_class': is_class}, out_model)
    print(f"Saved tree model to {out_model}")

    # plot feature importances
    try:
        importances = getattr(clf, 'feature_importances_', None)
        if importances is not None:
            feat_names = list(X.columns)
            idx = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, max(4, len(feat_names)*0.2)))
            plt.bar([feat_names[i] for i in idx], importances[idx])
            plt.xticks(rotation=90)
            plt.ylabel('Feature importance')
            plt.title('Tree Feature Importances')
            figp = os.path.splitext(out_model)[0] + '_coeffs.png'
            plt.tight_layout()
            plt.savefig(figp, dpi=150)
            plt.close()
            print(f"Saved importance plot to {figp}")
    except Exception:
        pass
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float('nan')
    except Exception:
        auc = float('nan')

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    # save model + scaler + feature names
    ensure_dir(out_model)
    joblib.dump({'model': clf, 'scaler': scaler, 'features': list(X.columns)}, out_model)
    print(f"Saved model to {out_model}")

    # plots: coefficients
    coef = clf.coef_.ravel()
    feat_names = list(X.columns)

    idx = np.argsort(np.abs(coef))[::-1]
    plt.figure(figsize=(10, max(4, len(feat_names)*0.2)))
    plt.bar([feat_names[i] for i in idx], coef[idx])
    plt.xticks(rotation=90)
    plt.ylabel('Coefficient (signed)')
    plt.title('Logistic Regression Coefficients')
    coef_plot = os.path.splitext(out_model)[0] + '_coeffs.png'
    plt.tight_layout()
    plt.savefig(coef_plot, dpi=150)
    plt.close()
    print(f"Saved coefficient plot to {coef_plot}")

    # ROC
    try:
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
            plt.plot([0,1],[0,1],'--',color='gray')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC curve')
            plt.legend()
            roc_plot = os.path.splitext(out_model)[0] + '_roc.png'
            plt.tight_layout()
            plt.savefig(roc_plot, dpi=150)
            plt.close()
            print(f"Saved ROC plot to {roc_plot}")
    except Exception:
        pass


def explain(csv_path, index, model_path, out_prefix=None, export_csv=None, model_type='logistic'):
    data = pd.read_csv(csv_path)
    obj = joblib.load(model_path)
    clf = obj['model']
    scaler = obj.get('scaler')
    feat_names = obj.get('features', [])
    is_class = obj.get('is_class', None)

    if isinstance(index, (list, tuple, np.ndarray)):
        indices = list(index)
    elif isinstance(index, int):
        for idx in indices:
            if idx not in data.index:
                raise IndexError(f'index {idx} not found in CSV')
            sample = data.loc[idx, feat_names]
            x = sample.values.reshape(1, -1)
            if scaler is not None:
                x_s = scaler.transform(x)
            else:
                x_s = x

            if model_type == 'tree':
                # try SHAP if available for better per-sample attributions
                contributions = None
                try:
                    import shap

                    explainer = shap.TreeExplainer(clf)
                    vals = explainer.shap_values(x_s)
                    if isinstance(vals, list):
                        vals = vals[1]
                    contributions = np.asarray(vals).ravel()
                except Exception:
                    importances = getattr(clf, 'feature_importances_', None)
                    if importances is not None:
                        contributions = importances * x_s.ravel()
                    else:
                        contributions = np.zeros_like(x_s.ravel())

                total = contributions.sum()
                prob = None
                if is_class:
                    try:
                        prob = clf.predict_proba(x_s)[0,1]
                    except Exception:
                        prob = None

                contrib_df = pd.DataFrame({'feature': feat_names, 'value': x.ravel(), 'contribution': contributions})
                contrib_df = contrib_df.sort_values('contribution', key=np.abs, ascending=False)

                print(f"Index: {idx}")
                print(f"Sum proxy contribution = {total:.6f}")
                if prob is not None:
                    print(f"Predicted probability = {prob:.6f}")
                print('\nTop feature contributions:')
                print(contrib_df.head(20).to_string(index=False))

                plt.figure(figsize=(8, max(3, len(feat_names)*0.2)))
                plt.barh(contrib_df['feature'], contrib_df['contribution'])
                plt.xlabel('Contribution (proxy for tree)')
                plt.title(f'Feature contributions for index {idx}')
                plt.tight_layout()

                if out_prefix is None:
                    out_prefix = os.path.splitext(model_path)[0] + f'_explain_{idx}'
                ensure_dir(out_prefix + '.png')
                out_plot = out_prefix + '.png'
                plt.savefig(out_plot, dpi=150)
                plt.close()
                print(f"Saved contribution plot to {out_plot}")

                df_out = contrib_df.copy()
                df_out['index'] = idx
                rows.append(df_out)
            else:
                coef = None
                intercept = 0.0
                if hasattr(clf, 'coef_'):
                    coef = clf.coef_.ravel()
                    intercept = float(clf.intercept_[0]) if hasattr(clf, 'intercept_') else 0.0
                else:
                    coef = np.zeros(len(feat_names))

                contributions = coef * x_s.ravel()
                total = contributions.sum() + intercept
                prob = 1 / (1 + np.exp(-total))

                contrib_df = pd.DataFrame({'feature': feat_names, 'value': x.ravel(), 'contribution': contributions})
                contrib_df = contrib_df.sort_values('contribution', key=np.abs, ascending=False)

                print(f"Index: {idx}")
                print(f"Linear sum + intercept = {total:.6f}")
                print(f"Predicted probability (sigmoid) = {prob:.6f}")
                print('\nTop feature contributions:')
                print(contrib_df.head(20).to_string(index=False))

                plt.figure(figsize=(8, max(3, len(feat_names)*0.2)))
                plt.barh(contrib_df['feature'], contrib_df['contribution'])
                plt.xlabel('Contribution to logit (coef * scaled_value)')
                plt.title(f'Feature contributions for index {idx} (prob={prob:.3f})')
                plt.tight_layout()

                if out_prefix is None:
                    out_prefix = os.path.splitext(model_path)[0] + f'_explain_{idx}'
                ensure_dir(out_prefix + '.png')
                out_plot = out_prefix + '.png'
                plt.savefig(out_plot, dpi=150)
                plt.close()
                print(f"Saved contribution plot to {out_plot}")

                df_out = contrib_df.copy()
                df_out['index'] = idx
                rows.append(df_out)

        df_out = contrib_df.copy()
        df_out['index'] = idx
        rows.append(df_out)

    if export_csv and rows:
        all_df = pd.concat(rows, ignore_index=True)
        ensure_dir(export_csv)
        all_df.to_csv(export_csv, index=False)
        print(f"Exported contributions CSV to {export_csv}")


def main():
    p = argparse.ArgumentParser(description='Logistic regression white-box tools')
    sub = p.add_subparsers(dest='cmd')

    t = sub.add_parser('train')
    t.add_argument('--csv', required=True)
    t.add_argument('--target', required=True)
    t.add_argument('--features', nargs='+', default=None)
    t.add_argument('--test-size', type=float, default=0.2)
    t.add_argument('--random-state', type=int, default=42)
    t.add_argument('--out', required=True, help='output path for joblib model')
    t.add_argument('--C', type=float, default=1.0, help='inverse regularization strength')
    t.add_argument('--penalty', choices=['l1','l2','elasticnet','none'], default='l2')
    t.add_argument('--class-weight', default=None, help="class weight ('balanced' or None)")
    t.add_argument('--max-iter', type=int, default=1000)
    t.add_argument('--solver', default=None, help='solver to use (overrides automatic choice)')
    t.add_argument('--l1-ratio', type=float, default=None, help='elasticnet mixing parameter (only if penalty=elasticnet)')
    t.add_argument('--tol', type=float, default=1e-4, help='tolerance for stopping criteria')
    t.add_argument('--fit-intercept', type=str, default='True', choices=['True','False'], help='whether to fit intercept')
    t.add_argument('--warm-start', action='store_true', help='reuse previous solution to speed up')
    t.add_argument('--random-state-lr', type=int, default=None, help='random state for logistic regression')

    e = sub.add_parser('explain')
    e.add_argument('--csv', required=True)
    e.add_argument('--index', type=int, nargs='+', required=True, help='row index(es) in CSV to explain')
    e.add_argument('--model', required=True, help='path to saved joblib model')
    e.add_argument('--out-prefix', default=None)
    e.add_argument('--export-csv', default=None, help='optional path to export contributions CSV')
    e.add_argument('--model-type', choices=['logistic','tree'], default='logistic', help='which model type was saved (logistic or tree)')

    args = p.parse_args()

    if args.cmd == 'train':
        class_weight = args.class_weight if args.class_weight != 'None' else None
        fit_intercept = True if str(args.fit_intercept) == 'True' else False
        if getattr(args, 'model_type', 'logistic') == 'tree' or getattr(args, 'model_type', None) == 'tree':
            # tree-specific params
            train_tree(args.csv, args.target, args.features, args.test_size, args.random_state, args.out, max_depth=getattr(args, 'max_depth', None), min_samples_leaf=getattr(args, 'min_samples_leaf', 1))
        else:
            train(
                args.csv,
                args.target,
                args.features,
                args.test_size,
                args.random_state,
                args.out,
                C=args.C,
                penalty=args.penalty,
                class_weight=class_weight,
                max_iter=args.max_iter,
                solver=args.solver,
                l1_ratio=args.l1_ratio,
                tol=args.tol,
                fit_intercept=fit_intercept,
                warm_start=args.warm_start,
                random_state_lr=args.random_state_lr,
            )
    elif args.cmd == 'explain':
        explain(args.csv, args.index, args.model, args.out_prefix, export_csv=args.export_csv, model_type=args.model_type)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
