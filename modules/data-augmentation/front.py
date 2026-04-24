#这是实验数据增强与去噪.py的前端接口
import streamlit as st
import matplotlib.pyplot as plt
from backend import (
	generate_data,
	build_generator,
	build_discriminator,
	build_gan,
	train_gan,
	generate_samples,
	load_csv,
	fit_preprocessor_and_transform,
	train_tabular_vae_preprocessed,
	vae_generate_samples_table,
	vae_reconstruct_denoise_table,
	save_vae_bundle_to_zip_bytes,
	load_vae_bundle_from_zip_bytes,
	TabularPreprocessor,
)
from keras.optimizers import Adam
import numpy as np

import pandas as pd
from typing import Any, cast


st.set_page_config(page_title="IGEM-FBH 实验数据增强与去噪", layout="wide")
st.title("实验数据增强与去噪")
st.write("数据增强 (生成) 与简单去噪操作的交互界面。")

mode = st.sidebar.selectbox("选择操作", ["数据增强", "数据去噪"])

# 全局：VAE模型状态与管理
with st.sidebar.expander("当前VAE模型状态", expanded=False):
	vae = st.session_state.get("vae_model")
	preproc = st.session_state.get("vae_preproc")
	if vae is None or preproc is None:
		st.caption("未训练")
	else:
		st.caption("已训练")
		st.write({
			"numeric_cols": list(getattr(preproc, "numeric_cols", [])),
			"categorical_cols": list(getattr(preproc, "categorical_cols", [])),
			"latent_dim": int(getattr(vae, "latent_dim", -1)),
		})
	if st.button("清除已训练VAE模型", key="reset_vae"):
		for k in ("vae_model", "vae_stats", "vae_columns", "vae_preproc", "vae_meta"):
			if k in st.session_state:
				del st.session_state[k]
		st.rerun()

	st.markdown("---")
	st.caption("模型包：训练后可下载；下次可直接上传使用")
	model_upload = st.file_uploader("上传VAE模型包(.zip)", type=["zip"], key="vae_bundle_upload")
	if model_upload is not None:
		try:
			zip_bytes = model_upload.getvalue()
			vae2, stats2, cols2, meta2 = load_vae_bundle_from_zip_bytes(zip_bytes)
			pre_d = cast(dict, meta2.get("preprocessor"))
			pre2 = TabularPreprocessor.from_dict(pre_d, stats2)
			st.session_state["vae_model"] = vae2
			st.session_state["vae_stats"] = stats2
			st.session_state["vae_columns"] = list(cols2)
			st.session_state["vae_preproc"] = pre2
			st.session_state["vae_meta"] = meta2
			st.success("模型包加载成功，可直接生成/去噪")
			st.write({"meta": meta2})
		except Exception as e:
			st.error(f"模型包加载失败：{e}")

if mode == "数据增强":
	st.sidebar.header("数据增强设置")
	source = st.sidebar.radio("数据来源", ["内置模拟数据（GAN演示）", "上传CSV（VAE增强）"], index=0)

	if source == "内置模拟数据（GAN演示）":
		num_samples = st.sidebar.slider("样本数量", 100, 2000, 500, step=100)
		latent_dim = st.sidebar.slider("潜在维度 (latent_dim)", 1, 16, 5)
		epochs = st.sidebar.number_input("训练轮数 (为演示建议较小值)", min_value=1, max_value=10000, value=200)
		run_train = st.sidebar.button("训练并生成样本")
		quick_gen = st.sidebar.button("快速生成 (不训练)")

		x, y = generate_data(num_samples)
		st.subheader("原始模拟数据示例")
		fig, ax = plt.subplots()
		ax.scatter(x, y, alpha=0.6)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		st.pyplot(fig)

		if quick_gen:
			gen = build_generator(latent_dim)
			samples = generate_samples(gen, n=200, latent_dim=latent_dim)
			fig2, ax2 = plt.subplots()
			ax2.scatter(y, np.zeros_like(y), label='Original y (as baseline)', alpha=0.3)
			ax2.scatter(samples, np.zeros_like(samples), color='r', label='Generated (untrained)')
			ax2.legend()
			st.subheader('快速生成结果（未训练的生成器）')
			st.pyplot(fig2)

		if run_train:
			st.info('开始训练（可能需要一段时间）')
			latent = latent_dim
			gen = build_generator(latent)
			disc = build_discriminator()
			disc.compile(loss='binary_crossentropy', optimizer=cast(Any, Adam(0.0002, 0.5)), metrics=['accuracy'])
			gan = build_gan(gen, disc)
			gan.compile(loss='binary_crossentropy', optimizer=cast(Any, Adam(0.0002, 0.5)))
			x_train = y  # 使用 y 作为训练数据的输入示例（保持与原脚本风格）
			train_gan(gen, disc, gan, x_train, epochs=int(epochs), batch_size=32)
			samples = generate_samples(gen, n=200, latent_dim=latent)
			fig3, ax3 = plt.subplots()
			ax3.scatter(y, np.zeros_like(y), label='Original y (baseline)', alpha=0.3)
			ax3.scatter(samples, np.zeros_like(samples), color='r', label='Generated')
			ax3.legend()
			st.subheader('训练后生成结果')
			st.pyplot(fig3)
	else:
		uploaded = st.sidebar.file_uploader("上传数据文件", type=["csv", "tsv", "txt", "xlsx", "xls"])
		if uploaded is None:
			st.info("请先在左侧上传数据文件（CSV/TSV/Excel）")
			st.stop()

		df = load_csv(uploaded)
		st.subheader("数据预览")
		st.dataframe(df.head(20), use_container_width=True)

		def _guess_numeric_columns(_df: pd.DataFrame) -> list:
			cols = []
			for c in _df.columns:
				ser = pd.to_numeric(_df[c], errors="coerce")
				ratio = float(ser.notna().mean()) if len(ser) else 0.0
				if ratio >= 0.8:
					cols.append(c)
				if len(cols) >= 10:
					break
			return cols

		def _guess_categorical_columns(_df: pd.DataFrame) -> list:
			cols = []
			for c in _df.columns:
				if c in default_cols:
					continue
				ser = _df[c]
				# 低基数/文本列优先
				n_unique = int(ser.nunique(dropna=True))
				if n_unique > 1 and n_unique <= 30:
					cols.append(c)
				if len(cols) >= 10:
					break
			return cols

		default_cols = _guess_numeric_columns(df)
		default_cat = _guess_categorical_columns(df)
		numeric_cols = st.sidebar.multiselect("选择数值列（连续变量）", options=list(df.columns), default=default_cols, key="aug_num")
		cat_options = [c for c in df.columns if c not in set(numeric_cols)]
		categorical_cols = st.sidebar.multiselect("选择分类型列（条件/组别等）", options=cat_options, default=[c for c in default_cat if c in cat_options], key="aug_cat")
		missing_strategy_ui = st.sidebar.selectbox("缺失值处理", ["drop（丢弃含缺失值的行）", "impute（填充缺失值）"], index=0)
		missing_strategy = "drop" if missing_strategy_ui.startswith("drop") else "impute"
		max_categories = st.sidebar.slider("每个分类型列最大类别数", 2, 200, 50, step=1)

		st.sidebar.markdown("---")
		st.sidebar.subheader("VAE训练参数")
		train_seed = st.sidebar.number_input("训练随机种子（可复现）", min_value=0, max_value=10**9, value=42, key="vae_train_seed_aug")
		vae_latent_dim = st.sidebar.slider("潜在维度 (latent_dim)", 1, 32, 4)
		h1 = st.sidebar.slider("隐藏层1", 8, 256, 64, step=8)
		h2 = st.sidebar.slider("隐藏层2", 8, 256, 64, step=8)
		beta = st.sidebar.slider("beta (KL权重，=1为标准VAE)", 0.0, 5.0, 1.0, step=0.1)
		val_split = st.sidebar.slider("validation_split", 0.0, 0.5, 0.1, step=0.05)
		early_stop = st.sidebar.checkbox("EarlyStopping", value=True)
		patience = st.sidebar.slider("patience", 1, 200, 30)
		min_delta = st.sidebar.number_input("min_delta", min_value=0.0, max_value=1.0, value=0.0, format="%.6f")
		vae_epochs = st.sidebar.number_input("训练轮数", min_value=1, max_value=20000, value=500)
		vae_batch = st.sidebar.slider("batch_size", 4, 256, 32, step=4)
		vae_lr = st.sidebar.number_input("learning_rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
		train_btn = st.sidebar.button("训练VAE")

		st.sidebar.markdown("---")
		st.sidebar.subheader("生成设置")
		gen_n = st.sidebar.slider("生成样本数", 10, 5000, 200, step=10)
		seed = st.sidebar.number_input("随机种子（可复现）", min_value=0, max_value=10**9, value=42)
		show_compare = st.sidebar.checkbox("显示分布对比图（前2列）", value=True)
		gen_btn = st.sidebar.button("生成并下载")

		if train_btn:
			try:
				x_pre, preproc, cleaned = fit_preprocessor_and_transform(
					df,
					numeric_cols=numeric_cols,
					categorical_cols=categorical_cols,
					missing_strategy=missing_strategy,
					max_categories=int(max_categories),
				)
			except Exception as e:
				st.error(str(e))
				st.stop()
			callbacks = []
			if early_stop and float(val_split) > 0.0:
				import keras
				callbacks.append(cast(
					Any,
					keras.callbacks.EarlyStopping(
						monitor="val_loss",
						patience=int(patience),
						min_delta=cast(Any, float(min_delta)),
						restore_best_weights=True,
					)
				))
			with st.spinner("训练VAE中..."):
				vae, history = train_tabular_vae_preprocessed(
					x_pre,
					latent_dim=int(vae_latent_dim),
					hidden_dims=(int(h1), int(h2)),
					beta=float(beta),
					epochs=int(vae_epochs),
					batch_size=int(vae_batch),
					learning_rate=float(vae_lr),
					validation_split=float(val_split),
					callbacks=callbacks if callbacks else None,
					seed=int(train_seed),
					verbose=0,
				)
			st.session_state["vae_model"] = vae
			st.session_state["vae_preproc"] = preproc
			st.session_state["vae_stats"] = preproc.stats
			st.session_state["vae_columns"] = list(preproc.numeric_cols)
			st.session_state["last_clean_df"] = cleaned
			st.success("VAE训练完成")
			# 训练完成后提供模型包下载
			try:
				bundle_bytes = save_vae_bundle_to_zip_bytes(vae, preproc=preproc)
				st.download_button(
					label="下载VAE模型包(.zip)",
					data=bundle_bytes,
					file_name="vae_model_bundle.zip",
					mime="application/zip",
				)
			except Exception as e:
				st.info(f"模型包导出失败：{e}")
			# 简单展示训练曲线
			try:
				losses = history.history.get("loss", [])
				if losses:
					st.line_chart(pd.DataFrame({"loss": losses}))
			except Exception:
				pass

		if gen_btn:
			vae = st.session_state.get("vae_model")
			preproc = st.session_state.get("vae_preproc")
			if vae is None or preproc is None:
				st.warning("请先训练VAE或上传VAE模型包")
				st.stop()
			with st.spinner("生成样本中..."):
				syn_df = vae_generate_samples_table(vae, preproc, n=int(gen_n), seed=int(seed))
			st.subheader("生成结果预览")
			st.dataframe(syn_df.head(20), use_container_width=True)
			if show_compare:
				try:
					cleaned = st.session_state.get("last_clean_df")
					if cleaned is None:
						_, _, cleaned = fit_preprocessor_and_transform(
							df,
							numeric_cols=preproc.numeric_cols,
							categorical_cols=preproc.categorical_cols,
							missing_strategy=preproc.missing_strategy,
							max_categories=preproc.max_categories,
						)
					st.caption(f"用于对比的有效行数: {len(cleaned)}")
					compare_cols = list(preproc.numeric_cols)[:2]
					if not compare_cols:
						raise ValueError("没有可绘制的数值列")
					figc, axes = plt.subplots(1, len(compare_cols), figsize=(6 * len(compare_cols), 4))
					if len(compare_cols) == 1:
						axes = [axes]
					for i, c in enumerate(compare_cols):
						x_real = pd.to_numeric(cleaned[c], errors="coerce").dropna().to_numpy()
						x_gen = pd.to_numeric(syn_df[c], errors="coerce").dropna().to_numpy()
						axes[i].hist(x_real, bins=30, alpha=0.5, label="Real")
						axes[i].hist(x_gen, bins=30, alpha=0.5, label="Generated")
						axes[i].set_title(c)
						axes[i].legend()
					st.pyplot(figc)
				except Exception as e:
					st.info(f"对比图生成失败：{e}")
			csv_bytes = syn_df.to_csv(index=False).encode("utf-8")
			st.download_button(
				label="下载生成CSV",
				data=csv_bytes,
				file_name="vae_augmented.csv",
				mime="text/csv",
			)

elif mode == "数据去噪":
	st.sidebar.header("数据去噪设置")
	method = st.sidebar.radio("去噪方式", ["内置模拟去噪（滑动平均）", "上传CSV（VAE重构去噪）"], index=0)

	if method == "内置模拟去噪（滑动平均）":
		noise_level = st.sidebar.slider("噪声强度", 0.0, 1.0, 0.1)
		num = st.sidebar.slider("样本数量", 100, 2000, 500, step=100)
		x, y = generate_data(num)
		noisy_y = y + np.random.normal(0, noise_level, y.shape)
		st.subheader('带噪声的数据示例')
		fig4, ax4 = plt.subplots()
		ax4.scatter(x, noisy_y, alpha=0.6)
		ax4.set_xlabel('x')
		ax4.set_ylabel('y')
		st.pyplot(fig4)
		if st.button('简单均值滤波去噪'):
			# 简单的滑动窗口平均作为示例去噪方法
			window = 5
			padded = np.pad(noisy_y.flatten(), (window//2, window//2), mode='edge')
			denoised = np.convolve(padded, np.ones(window)/window, mode='valid')
			fig5, ax5 = plt.subplots()
			ax5.scatter(x, denoised, alpha=0.6)
			ax5.set_title('去噪后结果（滑动平均）')
			st.pyplot(fig5)
	else:
		uploaded = st.sidebar.file_uploader("上传数据文件", type=["csv", "tsv", "txt", "xlsx", "xls"], key="denoise_file")
		if uploaded is None:
			st.info("请先在左侧上传数据文件（CSV/TSV/Excel）")
			st.stop()
		df = load_csv(uploaded)
		st.subheader("数据预览")
		st.dataframe(df.head(20), use_container_width=True)

		def _guess_numeric_columns(_df: pd.DataFrame) -> list:
			cols = []
			for c in _df.columns:
				ser = pd.to_numeric(_df[c], errors="coerce")
				ratio = float(ser.notna().mean()) if len(ser) else 0.0
				if ratio >= 0.8:
					cols.append(c)
				if len(cols) >= 10:
					break
			return cols

		default_cols = _guess_numeric_columns(df)
		default_cat = []
		for c in df.columns:
			if c in default_cols:
				continue
			ser = df[c]
			n_unique = int(ser.nunique(dropna=True))
			if n_unique > 1 and n_unique <= 30:
				default_cat.append(c)
			if len(default_cat) >= 10:
				break

		numeric_cols = st.sidebar.multiselect("选择数值列（连续变量）", options=list(df.columns), default=default_cols, key="den_num")
		cat_options = [c for c in df.columns if c not in set(numeric_cols)]
		categorical_cols = st.sidebar.multiselect("选择分类型列（条件/组别等）", options=cat_options, default=[c for c in default_cat if c in cat_options], key="den_cat")
		missing_strategy_ui = st.sidebar.selectbox("缺失值处理", ["drop（丢弃含缺失值的行）", "impute（填充缺失值）"], index=0, key="den_missing")
		missing_strategy = "drop" if missing_strategy_ui.startswith("drop") else "impute"
		max_categories = st.sidebar.slider("每个分类型列最大类别数", 2, 200, 50, step=1, key="den_maxcat")

		st.sidebar.markdown("---")
		st.sidebar.subheader("VAE训练参数")
		train_seed = st.sidebar.number_input("训练随机种子（可复现）", min_value=0, max_value=10**9, value=42, key="vae_train_seed_denoise")
		vae_latent_dim = st.sidebar.slider("潜在维度 (latent_dim)", 1, 32, 4, key="denoise_latent")
		h1 = st.sidebar.slider("隐藏层1", 8, 256, 64, step=8, key="denoise_h1")
		h2 = st.sidebar.slider("隐藏层2", 8, 256, 64, step=8, key="denoise_h2")
		beta = st.sidebar.slider("beta (KL权重，=1为标准VAE)", 0.0, 5.0, 1.0, step=0.1, key="denoise_beta")
		val_split = st.sidebar.slider("validation_split", 0.0, 0.5, 0.1, step=0.05, key="denoise_val")
		early_stop = st.sidebar.checkbox("EarlyStopping", value=True, key="denoise_es")
		patience = st.sidebar.slider("patience", 1, 200, 30, key="denoise_pat")
		min_delta = st.sidebar.number_input("min_delta", min_value=0.0, max_value=1.0, value=0.0, format="%.6f", key="denoise_delta")
		vae_epochs = st.sidebar.number_input("训练轮数", min_value=1, max_value=20000, value=500, key="denoise_epochs")
		vae_batch = st.sidebar.slider("batch_size", 4, 256, 32, step=4, key="denoise_batch")
		vae_lr = st.sidebar.number_input("learning_rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f", key="denoise_lr")
		train_btn = st.sidebar.button("训练VAE", key="denoise_train")
		apply_btn = st.sidebar.button("重构去噪并下载", key="denoise_apply")
		show_compare = st.sidebar.checkbox("显示去噪对比图（前2列）", value=True, key="denoise_compare")

		if train_btn:
			try:
				x_pre, preproc, cleaned = fit_preprocessor_and_transform(
					df,
					numeric_cols=numeric_cols,
					categorical_cols=categorical_cols,
					missing_strategy=missing_strategy,
					max_categories=int(max_categories),
				)
			except Exception as e:
				st.error(str(e))
				st.stop()
			callbacks = []
			if early_stop and float(val_split) > 0.0:
				import keras
				callbacks.append(cast(
					Any,
					keras.callbacks.EarlyStopping(
						monitor="val_loss",
						patience=int(patience),
						min_delta=cast(Any, float(min_delta)),
						restore_best_weights=True,
					)
				))
			with st.spinner("训练VAE中..."):
				vae, history = train_tabular_vae_preprocessed(
					x_pre,
					latent_dim=int(vae_latent_dim),
					hidden_dims=(int(h1), int(h2)),
					beta=float(beta),
					epochs=int(vae_epochs),
					batch_size=int(vae_batch),
					learning_rate=float(vae_lr),
					validation_split=float(val_split),
					callbacks=callbacks if callbacks else None,
					seed=int(train_seed),
					verbose=0,
				)
			st.session_state["vae_model"] = vae
			st.session_state["vae_preproc"] = preproc
			st.session_state["vae_stats"] = preproc.stats
			st.session_state["vae_columns"] = list(preproc.numeric_cols)
			st.session_state["last_clean_df"] = cleaned
			st.success("VAE训练完成")
			try:
				bundle_bytes = save_vae_bundle_to_zip_bytes(vae, preproc=preproc)
				st.download_button(
					label="下载VAE模型包(.zip)",
					data=bundle_bytes,
					file_name="vae_model_bundle.zip",
					mime="application/zip",
					key="download_bundle_denoise",
				)
			except Exception as e:
				st.info(f"模型包导出失败：{e}")
			try:
				losses = history.history.get("loss", [])
				if losses:
					st.line_chart(pd.DataFrame({"loss": losses}))
			except Exception:
				pass

		if apply_btn:
			vae = st.session_state.get("vae_model")
			preproc = st.session_state.get("vae_preproc")
			if vae is None or preproc is None:
				st.warning("请先训练VAE或上传VAE模型包")
				st.stop()
			with st.spinner("重构去噪中..."):
				den_df, cleaned_selected = vae_reconstruct_denoise_table(vae, preproc, df)
			out = df.copy()
			# 对齐：cleaned_selected 保留原索引
			for c in den_df.columns:
				out.loc[cleaned_selected.index, c] = den_df[c].to_numpy()

			st.subheader("去噪结果预览（原始 vs 重构）")
			cols_show = list(den_df.columns)[:5]
			preview = pd.DataFrame({
				f"{c}_raw": cleaned_selected[c].to_numpy()[:20] for c in cols_show
			})
			for c in cols_show:
				preview[f"{c}_denoised"] = den_df[c].to_numpy()[:20]
			st.dataframe(preview, use_container_width=True)
			if show_compare:
				compare_cols = [c for c in preproc.numeric_cols][:2]
				if not compare_cols:
					st.info("无数值列可绘制对比图")
					compare_cols = []
				figd, axes = plt.subplots(1, len(compare_cols), figsize=(6 * len(compare_cols), 4))
				if len(compare_cols) == 1:
					axes = [axes]
				for i, c in enumerate(compare_cols):
					raw = pd.to_numeric(cleaned_selected[c], errors="coerce").to_numpy(dtype=np.float32)
					den = pd.to_numeric(den_df[c], errors="coerce").to_numpy(dtype=np.float32)
					axes[i].scatter(raw, den, alpha=0.5)
					axes[i].set_xlabel(f"{c} raw")
					axes[i].set_ylabel(f"{c} denoised")
					axes[i].set_title(f"{c}: raw vs denoised")
				st.pyplot(figd)

			csv_bytes = out.to_csv(index=False).encode("utf-8")
			st.download_button(
				label="下载去噪后CSV",
				data=csv_bytes,
				file_name="vae_denoised.csv",
				mime="text/csv",
			)

st.sidebar.markdown('实验数据增强与去噪 工具')