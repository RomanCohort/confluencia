"""
VAE Denoise + Binary Classification Benchmark (288k IEDB)
==========================================================
Pipeline:
1. Load 288k data, sequence-aware split
2. Extract 317-dim features (Mamba3Lite + kmer + biochem + env)
3. Train VAE on training features → denoise
4. Train classifiers on: (a) raw features, (b) VAE-denoised features
5. Compare results
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

from core.features import build_feature_matrix

# Suppress TF warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATA_PATH = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
OUT_PATH = PROJECT / "benchmarks" / "results" / "vae_denoise_288k.json"


class SimpleVAE(keras.Model):
    """Lightweight VAE for feature denoising."""

    def __init__(self, input_dim, latent_dim=32, hidden_dims=(128, 64), beta=0.1, **kw):
        super().__init__(**kw)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder_body = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(hidden_dims[0], activation="relu"),
            keras.layers.Dense(hidden_dims[1], activation="relu"),
        ])
        self.z_mean = keras.layers.Dense(latent_dim)
        self.z_log_var = keras.layers.Dense(latent_dim)

        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(hidden_dims[1], activation="relu"),
            keras.layers.Dense(hidden_dims[0], activation="relu"),
            keras.layers.Dense(input_dim, activation="linear"),
        ])

    def encode(self, x):
        h = self.encoder_body(x)
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs, training=None):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(x)
            z = self.reparameterize(z_mean, z_log_var)
            x_hat = self.decode(z)
            recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            ))
            loss = recon + self.beta * kl
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients([(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None])
        return {"loss": loss, "recon": recon, "kl": kl}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decode(z)
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        ))
        return {"loss": recon + self.beta * kl, "recon": recon, "kl": kl}


def train_clf(name, model, X_tr, y_tr, X_te, y_te):
    """Train classifier and return metrics."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    return {
        "auc": float(roc_auc_score(y_te, prob)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "mcc": float(matthews_corrcoef(y_te, pred)),
        "train_time": elapsed,
    }, elapsed


def main():
    print("=" * 60)
    print("VAE Denoise + Binary Classification (288k IEDB)")
    print("=" * 60)

    # 1. Load + split
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["label"] = (df["efficacy"] >= 3.0).astype(int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["epitope_seq"].values))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Binders: {train_df['label'].mean():.1%}")

    # 2. Features
    print("\n[2] Feature extraction...")
    t0 = time.time()
    X_train, names, _ = build_feature_matrix(train_df)
    X_test, _, _ = build_feature_matrix(test_df)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape} ({time.time()-t0:.1f}s)")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # 3. Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # =============================================
    # PART A: Baseline (no denoise)
    # =============================================
    print("\n" + "=" * 60)
    print("PART A: Baseline (raw features)")
    print("=" * 60)
    baseline_results = {}

    models_a = {
        "LR": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "HGB": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True, random_state=42),
    }
    for name, model in models_a.items():
        print(f"  [{name}]", end=" ", flush=True)
        r, _ = train_clf(name, model, X_train if name != "LR" and name != "MLP" else X_train_s,
                         y_train,
                         X_test if name != "LR" and name != "MLP" else X_test_s,
                         y_test)
        baseline_results[name] = r
        print(f"AUC={r['auc']:.4f} Acc={r['accuracy']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f}")

    # =============================================
    # PART B: VAE Denoise
    # =============================================
    print("\n" + "=" * 60)
    print("PART B: VAE Denoise")
    print("=" * 60)

    # Train VAE
    print("\n[3] Training VAE...")
    tf.random.set_seed(42)
    np.random.seed(42)
    vae = SimpleVAE(input_dim=X_train_s.shape[1], latent_dim=64, hidden_dims=(256, 128), beta=0.05)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))

    t0 = time.time()
    # Use a subset for VAE training if too slow
    vae_train_size = min(len(X_train_s), 50000)
    idx = np.random.choice(len(X_train_s), vae_train_size, replace=False)
    vae.fit(X_train_s[idx], epochs=50, batch_size=256, verbose=1, validation_split=0.1)
    t_vae = time.time() - t0
    print(f"  VAE training: {t_vae:.1f}s (on {vae_train_size} samples)")

    # Denoise features
    print("\n[4] Denoising features...")
    t0 = time.time()
    X_train_den = vae(tf.constant(X_train_s), training=False).numpy().astype(np.float32)
    X_test_den = vae(tf.constant(X_test_s), training=False).numpy().astype(np.float32)
    t_denoise = time.time() - t0
    print(f"  Denoising: {t_denoise:.1f}s")

    # Evaluate reconstruction quality
    recon_error = np.mean((X_train_s - X_train_den) ** 2)
    print(f"  Reconstruction MSE: {recon_error:.6f}")

    # Train classifiers on denoised features
    print("\n[5] Classifiers on VAE-denoised features...")
    denoise_results = {}
    for name, model_cls in [
        ("LR", lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ("HGB", lambda: HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42)),
        ("RF", lambda: RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
        ("MLP", lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True, random_state=42)),
    ]:
        model = model_cls()
        use_scaled = name in ("LR", "MLP")
        print(f"  [{name}]", end=" ", flush=True)
        r, _ = train_clf(name, model,
                         X_train_den if use_scaled else X_train_den,
                         y_train,
                         X_test_den if use_scaled else X_test_den,
                         y_test)
        denoise_results[name] = r
        print(f"AUC={r['auc']:.4f} Acc={r['accuracy']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f}")

    # =============================================
    # PART C: VAE latent features
    # =============================================
    print("\n" + "=" * 60)
    print("PART C: VAE Latent Space (z_mean as features)")
    print("=" * 60)

    # Encode to latent space
    z_train = vae.encode(tf.constant(X_train_s, dtype=tf.float32))[0].numpy()
    z_test = vae.encode(tf.constant(X_test_s, dtype=tf.float32))[0].numpy()
    print(f"  Latent dim: {z_train.shape[1]}")

    latent_results = {}
    for name, model_cls in [
        ("LR", lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ("HGB", lambda: HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42)),
    ]:
        model = model_cls()
        print(f"  [{name}]", end=" ", flush=True)
        r, _ = train_clf(name, model, z_train, y_train, z_test, y_test)
        latent_results[name] = r
        print(f"AUC={r['auc']:.4f} Acc={r['accuracy']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f}")

    # =============================================
    # Summary
    # =============================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':20s} | {'Raw AUC':>8s} {'Raw F1':>7s} | {'Denoise AUC':>11s} {'Den F1':>7s} | {'Latent AUC':>10s} {'Lat F1':>7s}")
    print("-" * 90)
    for name in ["LR", "HGB", "RF", "MLP"]:
        b = baseline_results.get(name, {})
        d = denoise_results.get(name, {})
        l = latent_results.get(name, {})
        b_auc = f"{b.get('auc', 0):.4f}" if b else "  ---"
        b_f1 = f"{b.get('f1', 0):.4f}" if b else " ---"
        d_auc = f"{d.get('auc', 0):.4f}" if d else "  ---"
        d_f1 = f"{d.get('f1', 0):.4f}" if d else " ---"
        l_auc = f"{l.get('auc', 0):.4f}" if l else "  ---"
        l_f1 = f"{l.get('f1', 0):.4f}" if l else " ---"
        print(f"{name:20s} | {b_auc:>8s} {b_f1:>7s} | {d_auc:>11s} {d_f1:>7s} | {l_auc:>10s} {l_f1:>7s}")

    # Save
    output = {
        "baseline": baseline_results,
        "vae_denoise": denoise_results,
        "vae_latent": latent_results,
        "vae_config": {
            "latent_dim": 64,
            "hidden_dims": [256, 128],
            "beta": 0.05,
            "epochs": 50,
            "train_samples": vae_train_size,
            "reconstruction_mse": float(recon_error),
        },
        "data": {
            "total": len(df),
            "train": len(train_df),
            "test": len(test_df),
            "features": X_train.shape[1],
            "binder_rate": float(train_df["label"].mean()),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
