PINN 集成说明

- `src/pinn.py` 包含 PINN 主模块、`pinn_pde_residual`（默认扩散+Michaelis–Menten）以及内置残差函数：`heat_residual`、`poisson_residual`、`burgers_residual` 和 `default_residual`。
- 在多尺度流程中使用 `MultiScaleModel.register_physics(residual_fn, coeff_fn)` 注册外来物理残差函数或系数网络。
- 前端（Streamlit）已添加 PDE 配置区域，支持选择内置 PDE、粘贴或上传自定义 Python 残差函数，并可在会话内注册到当前模型。
- 示例脚本：`examples/pinn_poisson.py` 展示了如何注册 `poisson_residual` 并做短训。

注意：前端会 `exec` 用户提供的自定义代码，请仅在受信任环境中使用该功能。