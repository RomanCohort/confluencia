import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam

# 新增：用于湿实验小样本表格数据（CSV）的 VAE 增强/重构去噪
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import io
import json
import os
import tempfile
import zipfile

import pandas as pd
import tensorflow as tf
import keras


def load_table(uploaded_or_path) -> pd.DataFrame:
    """读取表格数据（CSV/TSV/Excel）。

    - Streamlit UploadedFile: 根据文件名后缀判断
    - 字符串路径: 根据后缀判断
    """
    name = None
    if hasattr(uploaded_or_path, "name"):
        name = str(getattr(uploaded_or_path, "name"))
    elif isinstance(uploaded_or_path, (str, os.PathLike)):
        name = str(uploaded_or_path)

    ext = (os.path.splitext(name)[1].lower() if name else "")
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_or_path)

    # CSV/TSV
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(uploaded_or_path, sep="\t")
    return pd.read_csv(uploaded_or_path)

# 生成训练数据
def generate_data(num_samples=1000):
    x = np.random.uniform(-1, 1, (num_samples, 1))
    y = 3 * x + np.random.normal(0, 0.1, (num_samples, 1))
    return x, y

# 构建生成器和判别器
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='linear'))
    return model


def build_discriminator():
    model = Sequential()
    model.add(Dense(32, input_dim=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# 构建GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# 训练GAN
def train_gan(generator, discriminator, gan, x_train, epochs=100, batch_size=32, verbose=0):
    half_batch = batch_size // 2
    # 某些打包/GUI/Streamlit 环境下 sys.stdout 可能为 None，Keras 进度条会因此崩溃。
    # 这里自动降级为静默输出。
    safe_verbose = int(verbose)
    try:
        import sys

        if sys.stdout is None:
            safe_verbose = 0
    except Exception:
        safe_verbose = 0
    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_samples = x_train[idx]
        noise = np.random.normal(0, 1, (half_batch, generator.input_shape[1]))
        fake_samples = generator.predict(noise, verbose=safe_verbose)
        discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
        discriminator.train_on_batch(fake_samples, np.zeros((half_batch, 1)))
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        valid_y = np.ones((batch_size, 1))
        gan.train_on_batch(noise, valid_y)

# 生成样本
def generate_samples(generator, n=100, latent_dim=5):
    noise = np.random.normal(0, 1, (n, latent_dim))
    # 生成样本不需要进度条，避免 stdout 缺失导致异常
    return generator.predict(noise, verbose=0)


@dataclass
class StandardScalerStats:
    mean: np.ndarray
    std: np.ndarray


def save_vae_bundle_to_zip_bytes(
    vae: "TabularVAE",
    stats: Optional[StandardScalerStats] = None,
    columns: Optional[Sequence[str]] = None,
    preproc: Optional["TabularPreprocessor"] = None,
) -> bytes:
    """将(模型+预处理器+元信息)打包成zip字节，便于下载复用。

    兼容：若只传 stats+columns，则会生成仅数值列的预处理器。
    """
    if preproc is None:
        if stats is None or columns is None:
            raise ValueError("需要提供 preproc 或 (stats, columns)")
        preproc = TabularPreprocessor(
            numeric_cols=list(columns),
            categorical_cols=[],
            categories={},
            cat_slices={},
            stats=stats,
            numeric_fill={},
            categorical_fill={},
            missing_strategy="drop",
            max_categories=0,
        )

    assert preproc is not None

    meta = {
        "latent_dim": int(getattr(vae, "latent_dim", -1)),
        "beta": float(getattr(vae, "beta", 1.0)),
        "input_dim": int(getattr(vae, "input_dim", -1)),
        "format": "vae_bundle_v2",
        "preprocessor": preproc.to_dict(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.keras")
        stats_path = os.path.join(tmpdir, "scaler_stats.npz")
        meta_path = os.path.join(tmpdir, "meta.json")

        # 保存模型（确保已build，避免Keras提示“未build可能无权重”）
        try:
            if not bool(getattr(vae, "built", False)):
                dummy = tf.zeros((1, int(getattr(vae, "input_dim", preproc.stats.mean.shape[1]))), dtype=tf.float32)
                _ = vae(dummy, training=False)
        except Exception:
            pass

        # 保存模型
        vae.save(model_path)
        # 保存标准化参数（只针对数值列）
        np.savez(stats_path, mean=preproc.stats.mean, std=preproc.stats.std)
        # 保存元信息
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path, arcname="model.keras")
            zf.write(stats_path, arcname="scaler_stats.npz")
            zf.write(meta_path, arcname="meta.json")
        return buffer.getvalue()


def load_vae_bundle_from_zip_bytes(zip_bytes: bytes) -> Tuple["TabularVAE", StandardScalerStats, List[str], dict]:
    """从zip字节恢复(模型+标准化参数+列信息)。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "bundle.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        model_path = os.path.join(tmpdir, "model.keras")
        stats_path = os.path.join(tmpdir, "scaler_stats.npz")
        meta_path = os.path.join(tmpdir, "meta.json")

        if not (os.path.exists(model_path) and os.path.exists(stats_path) and os.path.exists(meta_path)):
            raise ValueError("模型包缺少必要文件：需要 model.keras / scaler_stats.npz / meta.json")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with np.load(stats_path) as data:
            stats = StandardScalerStats(mean=data["mean"].astype(np.float32), std=data["std"].astype(np.float32))

        fmt = str(meta.get("format", "vae_bundle_v1"))
        if fmt == "vae_bundle_v1":
            columns = list(meta.get("columns", []))
            if not columns:
                raise ValueError("meta.json 中未找到 columns")
            # v1: 不包含预处理器
            meta["preprocessor"] = TabularPreprocessor(
                numeric_cols=columns,
                categorical_cols=[],
                categories={},
                cat_slices={},
                stats=stats,
                numeric_fill={},
                categorical_fill={},
                missing_strategy="drop",
                max_categories=0,
            ).to_dict()
        elif fmt == "vae_bundle_v2":
            if "preprocessor" not in meta:
                raise ValueError("vae_bundle_v2 缺少 preprocessor")
        else:
            raise ValueError(f"未知模型包格式: {fmt}")

        model = keras.models.load_model(
            model_path,
            custom_objects={"TabularVAE": TabularVAE},
            compile=False,
        )
        # 兼容返回值：仍返回 stats/columns，同时把完整preprocessor放进 meta
        pre_d = cast(dict, meta.get("preprocessor"))
        pre = TabularPreprocessor.from_dict(pre_d, stats)
        columns = list(pre.numeric_cols)
        return cast("TabularVAE", model), stats, columns, cast(dict, meta)


def train_tabular_vae_preprocessed(
    x_pre: np.ndarray,
    latent_dim: int = 4,
    hidden_dims: Sequence[int] = (64, 64),
    beta: float = 1.0,
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.0,
    callbacks: Optional[Sequence[Any]] = None,
    seed: Optional[int] = None,
    verbose: int = 0,
) -> Tuple["TabularVAE", keras.callbacks.History]:
    """在已预处理（数值已标准化、分类已one-hot）的矩阵上训练VAE。"""
    if x_pre.ndim != 2:
        raise ValueError("x_pre 必须是二维矩阵 (N, D)")
    if seed is not None:
        tf.random.set_seed(int(seed))
        np.random.seed(int(seed))
    vae = TabularVAE(
        input_dim=int(x_pre.shape[1]),
        latent_dim=int(latent_dim),
        hidden_dims=tuple(hidden_dims),
        beta=float(beta),
    )
    optimizer = keras.optimizers.Adam(learning_rate=float(learning_rate))
    vae.compile(optimizer=cast(Any, optimizer))
    fit_kwargs: Any = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "shuffle": True,
        "verbose": int(verbose),
        "validation_split": float(validation_split),
    }
    if callbacks is not None:
        fit_kwargs["callbacks"] = list(callbacks)
    history = cast(Any, vae.fit(x_pre.astype(np.float32), **fit_kwargs))
    return vae, history


def vae_generate_samples_table(
    vae: "TabularVAE",
    preproc: "TabularPreprocessor",
    n: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """生成混合表格样本（返回DataFrame：数值列+分类列）。"""
    if seed is not None:
        tf.random.set_seed(int(seed))
        np.random.seed(int(seed))
    z = tf.random.normal(shape=(int(n), int(vae.latent_dim)))
    x_hat = vae.decoder(z).numpy().astype(np.float32)
    return inverse_transform_to_dataframe(x_hat, preproc)


def vae_reconstruct_denoise_table(
    vae: "TabularVAE",
    preproc: "TabularPreprocessor",
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对df进行重构去噪。

    返回 (denoised_df_selected_cols, cleaned_df_selected_cols)
    - 只包含 preproc 里定义的列
    - cleaned_df 为经过drop/impute后的对齐版本
    """
    x_pre, cleaned = transform_with_preprocessor(df, preproc)
    x_hat = vae(tf.convert_to_tensor(x_pre), training=False).numpy().astype(np.float32)
    den = inverse_transform_to_dataframe(x_hat, preproc)
    return den, cleaned.loc[:, preproc.numeric_cols + preproc.categorical_cols]


def load_csv(uploaded_or_path) -> pd.DataFrame:
    """兼容旧接口：现在支持 CSV/TSV/Excel。"""
    return load_table(uploaded_or_path)


@dataclass
class TabularPreprocessor:
    numeric_cols: List[str]
    categorical_cols: List[str]
    categories: Dict[str, List[str]]
    cat_slices: Dict[str, Tuple[int, int]]
    stats: StandardScalerStats
    numeric_fill: Dict[str, float]
    categorical_fill: Dict[str, str]
    missing_strategy: str = "drop"  # drop | impute
    max_categories: int = 50

    def to_dict(self) -> dict:
        return {
            "numeric_cols": list(self.numeric_cols),
            "categorical_cols": list(self.categorical_cols),
            "categories": {k: list(v) for k, v in self.categories.items()},
            "cat_slices": {k: [int(a), int(b)] for k, (a, b) in self.cat_slices.items()},
            "numeric_fill": {k: float(v) for k, v in self.numeric_fill.items()},
            "categorical_fill": {k: str(v) for k, v in self.categorical_fill.items()},
            "missing_strategy": self.missing_strategy,
            "max_categories": int(self.max_categories),
            "format": "tabular_preproc_v1",
        }

    @staticmethod
    def from_dict(d: dict, stats: StandardScalerStats) -> "TabularPreprocessor":
        cat_slices = {k: (int(v[0]), int(v[1])) for k, v in (d.get("cat_slices") or {}).items()}
        return TabularPreprocessor(
            numeric_cols=list(d.get("numeric_cols") or []),
            categorical_cols=list(d.get("categorical_cols") or []),
            categories=cast(Dict[str, List[str]], d.get("categories") or {}),
            cat_slices=cat_slices,
            stats=stats,
            numeric_fill=cast(Dict[str, float], d.get("numeric_fill") or {}),
            categorical_fill=cast(Dict[str, str], d.get("categorical_fill") or {}),
            missing_strategy=str(d.get("missing_strategy") or "drop"),
            max_categories=int(d.get("max_categories") or 50),
        )


def _fit_categories(series: pd.Series, max_categories: int) -> List[str]:
    s = series.astype(str)
    vc = s.value_counts(dropna=False)
    if len(vc) <= max_categories:
        cats = list(vc.index.astype(str))
        return cats
    # 留一个槽给 OTHER
    keep = list(vc.index.astype(str)[: max_categories - 1])
    return keep + ["__OTHER__"]


def _apply_categories(series: pd.Series, categories: List[str]) -> pd.Series:
    s = series.astype(str)
    if "__OTHER__" in categories:
        return s.where(s.isin(categories), "__OTHER__")
    # 没有 OTHER 时，未知值映射到第一个类别
    default = categories[0] if categories else ""
    return s.where(s.isin(categories), default)


def fit_preprocessor_and_transform(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    missing_strategy: str = "drop",
    max_categories: int = 50,
) -> Tuple[np.ndarray, TabularPreprocessor, pd.DataFrame]:
    """对混合表格数据拟合预处理器并输出训练矩阵。

    返回 (x_preprocessed, preprocessor, cleaned_df)
    - numeric: 标准化
    - categorical: one-hot
    """
    numeric_cols = list(numeric_cols)
    categorical_cols = list(categorical_cols)
    if not numeric_cols and not categorical_cols:
        raise ValueError("至少选择一个数值列或分类型列")

    sub = df.loc[:, list(dict.fromkeys(numeric_cols + categorical_cols))].copy()
    for c in numeric_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    if missing_strategy not in {"drop", "impute"}:
        raise ValueError("missing_strategy 只能是 drop 或 impute")

    if missing_strategy == "drop":
        sub = sub.dropna(axis=0, how="any")
    else:
        # 数值：均值填充；分类：众数填充
        for c in numeric_cols:
            if sub[c].isna().any():
                sub[c] = sub[c].fillna(sub[c].mean())
        for c in categorical_cols:
            if sub[c].isna().any():
                mode = sub[c].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else "__MISSING__"
                sub[c] = sub[c].fillna(fill)

    if len(sub) < 5:
        raise ValueError("有效行太少（<5），请检查列选择或缺失值处理")

    # 拟合分类类别
    categories: Dict[str, List[str]] = {}
    for c in categorical_cols:
        categories[c] = _fit_categories(sub[c], int(max_categories))

    numeric_fill: Dict[str, float] = {}
    categorical_fill: Dict[str, str] = {}

    if missing_strategy == "impute":
        for c in numeric_cols:
            numeric_fill[c] = float(pd.to_numeric(sub[c], errors="coerce").mean())
        for c in categorical_cols:
            mode = sub[c].mode(dropna=True)
            categorical_fill[c] = str(mode.iloc[0] if len(mode) else "__MISSING__")

    # 数值标准化（只对数值列）
    if numeric_cols:
        x_num = sub.loc[:, numeric_cols].to_numpy(dtype=np.float32)
        x_num_z, stats = standardize_fit_transform(x_num)
    else:
        x_num_z = np.zeros((len(sub), 0), dtype=np.float32)
        stats = StandardScalerStats(mean=np.zeros((1, 0), dtype=np.float32), std=np.ones((1, 0), dtype=np.float32))

    # one-hot
    onehots: List[np.ndarray] = []
    cat_slices: Dict[str, Tuple[int, int]] = {}
    offset = x_num_z.shape[1]
    for c in categorical_cols:
        cats = categories[c]
        s = _apply_categories(sub[c], cats)
        codes = pd.Categorical(s, categories=cats).codes
        if (codes < 0).any():
            codes = np.where(codes < 0, 0, codes)
        oh = np.eye(len(cats), dtype=np.float32)[codes]
        start, end = offset, offset + oh.shape[1]
        cat_slices[c] = (start, end)
        offset = end
        onehots.append(oh)

    x_cat = np.concatenate(onehots, axis=1) if onehots else np.zeros((len(sub), 0), dtype=np.float32)
    x = np.concatenate([x_num_z, x_cat], axis=1).astype(np.float32)
    preproc = TabularPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        categories=categories,
        cat_slices=cat_slices,
        stats=stats,
        numeric_fill=numeric_fill,
        categorical_fill=categorical_fill,
        missing_strategy=missing_strategy,
        max_categories=int(max_categories),
    )
    return x, preproc, sub


def transform_with_preprocessor(df: pd.DataFrame, preproc: TabularPreprocessor) -> Tuple[np.ndarray, pd.DataFrame]:
    """用已有预处理器将 df 转为模型输入矩阵（并返回清洗后的df）。"""
    cols = list(dict.fromkeys(preproc.numeric_cols + preproc.categorical_cols))
    sub = df.loc[:, cols].copy()
    for c in preproc.numeric_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    if preproc.missing_strategy == "drop":
        sub = sub.dropna(axis=0, how="any")
    else:
        for c in preproc.numeric_cols:
            fill = float(preproc.numeric_fill.get(c, 0.0))
            sub[c] = sub[c].fillna(fill)
        for c in preproc.categorical_cols:
            fill = str(preproc.categorical_fill.get(c, "__MISSING__"))
            sub[c] = sub[c].fillna(fill)

    if preproc.numeric_cols:
        x_num = sub.loc[:, preproc.numeric_cols].to_numpy(dtype=np.float32)
        x_num_z = standardize_transform(x_num, preproc.stats)
    else:
        x_num_z = np.zeros((len(sub), 0), dtype=np.float32)

    onehots: List[np.ndarray] = []
    for c in preproc.categorical_cols:
        cats = preproc.categories.get(c, [])
        s = _apply_categories(sub[c], cats)
        codes = pd.Categorical(s, categories=cats).codes
        if (codes < 0).any():
            codes = np.where(codes < 0, 0, codes)
        oh = np.eye(len(cats), dtype=np.float32)[codes]
        onehots.append(oh)

    x_cat = np.concatenate(onehots, axis=1) if onehots else np.zeros((len(sub), 0), dtype=np.float32)
    x_out = np.concatenate([x_num_z, x_cat], axis=1).astype(np.float32)
    return x_out, sub


def inverse_transform_to_dataframe(x_pre: np.ndarray, preproc: TabularPreprocessor) -> pd.DataFrame:
    """将模型输出矩阵反变换为 DataFrame（数值反标准化、分类argmax还原）。"""
    n = x_pre.shape[0]
    out: Dict[str, Any] = {}
    # numeric
    num_dim = len(preproc.numeric_cols)
    if num_dim:
        x_num_z = x_pre[:, :num_dim]
        x_num = standardize_inverse_transform(x_num_z.astype(np.float32), preproc.stats)
        for i, c in enumerate(preproc.numeric_cols):
            out[c] = x_num[:, i]
    # categorical
    for c in preproc.categorical_cols:
        start, end = preproc.cat_slices.get(c, (num_dim, num_dim))
        cats = preproc.categories.get(c, [])
        if end <= start or not cats:
            out[c] = ["" for _ in range(n)]
            continue
        logits = x_pre[:, start:end]
        idx = np.argmax(logits, axis=1)
        out[c] = [cats[int(i)] for i in idx]
    return pd.DataFrame(out)


def dataframe_to_numeric_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """选择列并转为 float32 数值矩阵，自动丢弃含 NaN 的行。"""
    if not columns:
        raise ValueError("columns 不能为空")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"找不到列: {missing}")
    sub = df.loc[:, list(columns)].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(axis=0, how="any")
    if len(sub) < 5:
        raise ValueError("有效数值行太少（<5），请检查列选择或数据质量")
    return sub.to_numpy(dtype=np.float32)


def standardize_fit_transform(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, StandardScalerStats]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    xz = (x - mean) / std
    return xz.astype(np.float32), StandardScalerStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def standardize_transform(x: np.ndarray, stats: StandardScalerStats) -> np.ndarray:
    return ((x - stats.mean) / stats.std).astype(np.float32)


def standardize_inverse_transform(xz: np.ndarray, stats: StandardScalerStats) -> np.ndarray:
    return (xz * stats.std + stats.mean).astype(np.float32)


class TabularVAE(keras.Model):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.beta = float(beta)

        encoder_layers: List[keras.layers.Layer] = [keras.layers.InputLayer(input_shape=(self.input_dim,))]
        for h in self.hidden_dims:
            encoder_layers.append(keras.layers.Dense(int(h), activation="relu"))
        self.encoder_body = keras.Sequential(encoder_layers, name="encoder_body")
        self.z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")

        decoder_layers: List[keras.layers.Layer] = [keras.layers.InputLayer(input_shape=(self.latent_dim,))]
        for h in reversed(self.hidden_dims):
            decoder_layers.append(keras.layers.Dense(int(h), activation="relu"))
        decoder_layers.append(keras.layers.Dense(self.input_dim, activation="linear"))
        self.decoder = keras.Sequential(decoder_layers, name="decoder")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": int(self.input_dim),
                "latent_dim": int(self.latent_dim),
                "hidden_dims": list(self.hidden_dims),
                "beta": float(self.beta),
            }
        )
        return config


    def encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h = self.encoder_body(x)
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=tf.shape(z_mean))
        half = tf.constant(0.5, dtype=z_log_var.dtype)
        std = tf.exp(tf.multiply(half, z_log_var))
        return tf.add(z_mean, tf.multiply(std, eps))

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        return self.decoder(z)

    def call(self, inputs, training=None):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decode(z)
        return x_hat

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(x)
            z = self.reparameterize(z_mean, z_log_var)
            x_hat = self.decode(z)
            recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
            one = tf.constant(1.0, dtype=z_log_var.dtype)
            half = tf.constant(0.5, dtype=z_log_var.dtype)
            kl_terms = tf.subtract(
                tf.subtract(tf.add(one, z_log_var), tf.square(z_mean)),
                tf.exp(z_log_var),
            )
            kl = tf.negative(tf.multiply(half, tf.reduce_mean(tf.reduce_sum(kl_terms, axis=1))))
            beta_t = tf.constant(self.beta, dtype=kl.dtype)
            loss = recon + tf.multiply(beta_t, kl)
        grads = tape.gradient(loss, self.trainable_variables)
        grads_any: Any = grads if grads is not None else []
        grads_list = list(grads_any)
        grads_and_vars = [(g, v) for g, v in zip(grads_list, self.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)
        return {"loss": loss, "recon": recon, "kl": kl}

    def test_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decode(z)
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
        one = tf.constant(1.0, dtype=z_log_var.dtype)
        half = tf.constant(0.5, dtype=z_log_var.dtype)
        kl_terms = tf.subtract(
            tf.subtract(tf.add(one, z_log_var), tf.square(z_mean)),
            tf.exp(z_log_var),
        )
        kl = tf.negative(tf.multiply(half, tf.reduce_mean(tf.reduce_sum(kl_terms, axis=1))))
        beta_t = tf.constant(self.beta, dtype=kl.dtype)
        loss = recon + tf.multiply(beta_t, kl)
        return {"loss": loss, "recon": recon, "kl": kl}


def train_tabular_vae(
    x: np.ndarray,
    latent_dim: int = 4,
    hidden_dims: Sequence[int] = (64, 64),
    beta: float = 1.0,
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.0,
    callbacks: Optional[Sequence[Any]] = None,
    seed: Optional[int] = None,
    verbose: int = 0,
) -> Tuple[TabularVAE, StandardScalerStats, keras.callbacks.History]:
    """训练 Tabular VAE。

    输入 x: (N, D) 的连续数值矩阵。
    返回: (vae, scaler_stats, history)
    """
    if x.ndim != 2:
        raise ValueError("x 必须是二维矩阵 (N, D)")
    if seed is not None:
        tf.random.set_seed(int(seed))
        np.random.seed(int(seed))
    xz, stats = standardize_fit_transform(x)
    vae = TabularVAE(
        input_dim=xz.shape[1],
        latent_dim=int(latent_dim),
        hidden_dims=tuple(hidden_dims),
        beta=float(beta),
    )
    optimizer = keras.optimizers.Adam(learning_rate=float(learning_rate))
    vae.compile(optimizer=cast(Any, optimizer))
    fit_kwargs: Any = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "shuffle": True,
        "verbose": int(verbose),
        "validation_split": float(validation_split),
    }
    if callbacks is not None:
        fit_kwargs["callbacks"] = list(callbacks)
    history = cast(Any, vae.fit(xz, **fit_kwargs))
    return vae, stats, history


def vae_generate_samples(
    vae: TabularVAE,
    stats: StandardScalerStats,
    n: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """从标准正态先验采样生成新样本，并反标准化回原空间。"""
    if seed is not None:
        tf.random.set_seed(int(seed))
        np.random.seed(int(seed))
    z = tf.random.normal(shape=(int(n), int(vae.latent_dim)))
    xz_hat = vae.decoder(z).numpy().astype(np.float32)
    return standardize_inverse_transform(xz_hat, stats)


def vae_reconstruct_denoise(
    vae: TabularVAE,
    stats: StandardScalerStats,
    x_noisy: np.ndarray,
) -> np.ndarray:
    """用 VAE 的重构作为连续数值特征的去噪结果。"""
    xz = standardize_transform(x_noisy, stats)
    xz_hat = vae(tf.convert_to_tensor(xz), training=False).numpy().astype(np.float32)
    return standardize_inverse_transform(xz_hat, stats)
