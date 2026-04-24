"""src.ui.multiscale_ui -- 多尺度 GNN-PINN 建模 UI 模块。

从 frontend.py 提取的多尺度相关函数：
- multiscale_ui()
- render_multiscale_sidebar()
- atom_heatmap_image()
- atom_heatmap_svg()
- compute_admet_quick()
"""
from __future__ import annotations

import io as _io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from src.ui.constants import _PROJECT_ROOT


# ----------------------------------------------------------------------
# 分子可视化与计算函数
# ----------------------------------------------------------------------
def atom_heatmap_image(
    smiles: str,
    scores: dict,
    size: Tuple[int, int] = (400, 300),
) -> Optional[Image.Image]:
    """生成原子热图 PNG 图像。

    Args:
        smiles: 分子 SMILES 字符串
        scores: 原子索引到分数的映射字典
        size: 图像尺寸 (width, height)

    Returns:
        PIL Image 对象，失败时返回 None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        # 将分数映射到颜色（红色高分，蓝色低分）
        vals = [scores.get(i, 0.0) for i in range(mol.GetNumAtoms())]
        if sum(vals) == 0:
            cmap = {i: (0.8, 0.8, 0.8) for i in range(len(vals))}
        else:
            import matplotlib.pyplot as plt
            norm = plt.Normalize(min(vals), max(vals))
            cmap_map = plt.get_cmap("RdYlBu_r")
            cmap = {
                i: tuple(int(255 * c) for c in cmap_map(norm(v))[:3])
                for i, v in enumerate(vals)
            }

        highlight_atoms = list(range(mol.GetNumAtoms()))
        highlight_colors = {
            i: tuple([c / 255.0 for c in cmap[i]])
            for i in highlight_atoms
        }

        from rdkit.Chem.Draw import Draw
        drawer = Draw.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        for i in highlight_atoms:
            opts.atomHighlights[i] = highlight_colors[i]
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
        )
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        img = Image.open(_io.BytesIO(png))
        return img
    except Exception:
        return None


def atom_heatmap_svg(
    smiles: str,
    scores: dict,
    width: int = 480,
    height: int = 360,
) -> Optional[str]:
    """生成交互式原子热图 SVG HTML。

    包含原子工具提示和颜色条。
    失败时返回 None。

    Args:
        smiles: 分子 SMILES 字符串
        scores: 原子索引到分数的映射字典
        width: SVG 宽度
        height: SVG 高度

    Returns:
        HTML 字符串，包含交互式 SVG，失败时返回 None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.padding = 0.05
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)

        # 尝试获取原子绘制坐标
        atom_coords = []
        n = mol.GetNumAtoms()
        for i in range(n):
            try:
                p = drawer.GetDrawCoords(i)
                atom_coords.append((float(p.x), float(p.y)))
            except Exception:
                atom_coords = []
                break

        base_svg = drawer.GetDrawingText()
        drawer.FinishDrawing()

        if not atom_coords:
            return None

        # 构建颜色映射
        vals = [scores.get(i, 0.0) for i in range(n)]
        vmin = min(vals)
        vmax = max(vals)
        if vmin == vmax:
            vmax = vmin + 1e-6

        def to_rgb(v):
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap("RdYlBu_r")
            norm = plt.Normalize(vmin, vmax)
            c = cmap(norm(v))
            return int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)

        circles_svg = ""
        for i, (x, y) in enumerate(atom_coords):
            r = 8
            col = to_rgb(vals[i])
            color = f"rgb({col[0]},{col[1]},{col[2]})"
            tooltip = f"atom {i}: {vals[i]:.4f}"
            circles_svg += (
                f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" '
                f'fill-opacity="0.8" stroke="#222" stroke-width="0.6" '
                f'data-tip="{tooltip}" data-atom="{i}"></circle>'
            )

        # 颜色条 SVG：垂直渐变
        grad_id = "gbar"
        colorbar_svg = f'''
        <defs>
          <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="rgb{to_rgb(vmax)}"/>
            <stop offset="100%" stop-color="rgb{to_rgb(vmin)}"/>
          </linearGradient>
        </defs>
        <rect x="{width-70}" y="20" width="20" height="200" fill="url(#{grad_id})" stroke="#000" />
        <text x="{width-40}" y="30" font-size="12">{vmax:.3f}</text>
        <text x="{width-40}" y="220" font-size="12">{vmin:.3f}</text>
        '''

        # 组装 HTML：基础 SVG + 覆盖圆圈 + 工具提示脚本
        html = (
            '<div style="position:relative; width:' + str(width) + 'px; height:' + str(height) + 'px">'
            + base_svg
            + '<svg width="' + str(width) + '" height="' + str(height) + '" style="position:absolute; left:0; top:0; pointer-events:none;">'
            + circles_svg
            + colorbar_svg
            + '</svg>'
            + '<div id="tip" style="position:absolute; display:none; background:#fff; padding:6px; border:1px solid #333; border-radius:4px; font-size:12px; pointer-events:none;"></div>'
            + '</div>'
            + '<script>'
            + "const tip=document.getElementById('tip');"
            + "function showTip(evt, text){ tip.style.display='block'; tip.style.left=(evt.clientX+10)+'px'; tip.style.top=(evt.clientY+10)+'px'; tip.innerText=text; }"
            + "function hideTip(){ tip.style.display='none'; }"
            + "const svgs=document.getElementsByTagName('svg'); for(const s of svgs){ s.style.pointerEvents='auto'; }"
            + "document.querySelectorAll('circle[data-atom]').forEach(c=>{ c.style.cursor='pointer'; c.addEventListener('mouseover', (e)=>{ showTip(e, c.getAttribute('data-tip')); }); c.addEventListener('mouseout', hideTip); });"
            + '</script>'
        )
        return html
    except Exception:
        return None


def compute_admet_quick(smiles: str) -> Dict[str, float]:
    """快速基于 RDKit 的 ADMET 估计：logP、MW、TPSA、RotB 和 ESOL logS（Delaney 风格）。

    Args:
        smiles: 分子 SMILES 字符串

    Returns:
        包含 ADMET 估计值的字典，失败时返回空字典
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        rotb = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        arom_prop = float(n_arom / mol.GetNumAtoms()) if mol.GetNumAtoms() > 0 else 0.0

        # Delaney ESOL 风格估计（简单线性模型）
        esol_logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * arom_prop

        return {
            "MolLogP": logp,
            "MolWt": mw,
            "RotatableBonds": rotb,
            "TPSA": tpsa,
            "AromaticProportion": arom_prop,
            "ESOL_logS": esol_logS,
        }
    except Exception:
        return {}


# ----------------------------------------------------------------------
# 多尺度侧边栏渲染
# ----------------------------------------------------------------------
def render_multiscale_sidebar() -> None:
    """渲染多尺度 GNN-PINN 建模的侧边栏控制面板。"""
    run = False
    train_demo = False
    train_epochs = int(st.session_state.get("ms_train_epochs", 8))

    tabs = st.tabs(
        ["基础", "物理势/等变", "预测目标", "模型上传", "运行训练", "RL 采样", "显示设置"]
    )
    with tabs[0]:
        st.caption("输入分子 SMILES，计算原子敏感性并演示 PINN 微观→宏观耦合示例。")
        crawled_ms = st.session_state.get("multiscale_crawl_df")
        if isinstance(crawled_ms, pd.DataFrame) and "smiles" in crawled_ms.columns and len(crawled_ms) > 0:
            st.caption("可从爬虫抓取的多尺度数据中选择 SMILES")
            options = [s for s in crawled_ms["smiles"].astype(str).tolist() if s]
            sel = st.selectbox("选择 SMILES（来自爬虫）", options=options, index=0, key="ms_crawl_smiles_select")
            smiles = st.text_input("SMILES", value=str(sel), key="ms_smiles")
            # 自动填充 PDE 参数（若存在）
            try:
                row = crawled_ms[crawled_ms["smiles"].astype(str) == str(sel)].iloc[0]
                if "D" in crawled_ms.columns:
                    st.session_state["ms_default_D"] = float(row.get("D", st.session_state.get("ms_default_D", 0.1)))
                if "Vmax" in crawled_ms.columns:
                    st.session_state["ms_default_Vmax"] = float(row.get("Vmax", st.session_state.get("ms_default_Vmax", 0.5)))
                if "Km" in crawled_ms.columns:
                    st.session_state["ms_default_Km"] = float(row.get("Km", st.session_state.get("ms_default_Km", 0.1)))
            except Exception:
                pass
        else:
            smiles = st.text_input("SMILES", value=st.session_state.get("ms_smiles", "CCO"), key="ms_smiles")

        steps = st.number_input(
            "GNN 消息传递步数",
            min_value=1, max_value=8,
            value=int(st.session_state.get("ms_steps", 3)),
            step=1, key="ms_steps",
        )
        hidden = st.number_input("GNN 隐藏维度", min_value=8, max_value=256, value=64, step=8, key="ms_hidden")
        gnn_dropout = st.slider("GNN Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="ms_gnn_dropout")
        model_type = st.selectbox(
            "GNN 类型",
            options=["SimpleGNN", "EnhancedGNN", "PhysicsMessageGNN", "E(3)-EquivariantGNN"],
            index=1 if st.session_state.get("ms_use_enhanced", True) else 0,
            key="ms_model_type",
        )
        gat_heads = st.number_input("GAT 头数 (仅 EnhancedGNN)", min_value=1, max_value=8, value=4, step=1, key="ms_gat_heads")
        use_physics = st.checkbox("在 GAT 中使用物理调制 (距离/角度)", value=True, key="ms_use_physics")
        enable_coeff = st.checkbox("启用 coeff_net（分子嵌入 -> PDE 系数）", value=False, key="ms_enable_coeff")
        coeff_hidden = st.number_input("CoeffNet 隐藏维度", min_value=8, max_value=256, value=64, step=8, key="ms_coeff_hidden")
        readout_type = st.selectbox("Readout 类型", options=["mean", "attention"], index=0, key="ms_readout")

        st.markdown("**PDE 默认参数（可用于 PINN 训练/演示）**")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.number_input("D", min_value=0.0, value=float(st.session_state.get("ms_default_D", 0.1)), step=0.01, key="ms_default_D")
        with dcol2:
            st.number_input("Vmax", min_value=0.0, value=float(st.session_state.get("ms_default_Vmax", 0.5)), step=0.05, key="ms_default_Vmax")
        with dcol3:
            st.number_input("Km", min_value=0.0, value=float(st.session_state.get("ms_default_Km", 0.1)), step=0.01, key="ms_default_Km")

    with tabs[1]:
        st.caption("物理势参数（仅在 PhysicsMessageGNN 时生效）")
        potential_type = st.selectbox("potential_type", options=["auto", "lennard", "electrostatic"], index=0, key="ms_potential_type")
        lj_epsilon = st.number_input("LJ epsilon", min_value=0.0, max_value=10.0, value=0.1, format="%g", key="ms_lj_eps")
        lj_sigma = st.number_input("LJ sigma (A)", min_value=0.1, max_value=10.0, value=3.5, format="%g", key="ms_lj_sigma")
        dielectric = st.number_input("电介常数", min_value=1.0, max_value=1000.0, value=80.0, format="%g", key="ms_dielectric")
        st.markdown("---")
        st.caption("E(3)-等变 GNN 参数（用于 E(3)-EquivariantGNN）")
        eg_layers = st.number_input("EGNN 层数", min_value=1, max_value=12, value=3, step=1, key="ms_eg_layers")
        eg_hidden = st.number_input("EGNN 隐藏维度", min_value=8, max_value=512, value=64, step=8, key="ms_eg_hidden")
        st.caption(
            "说明: LJ epsilon 控制势深度（越大吸引/排斥越强），sigma 为粒子尺寸标度（A）。"
            "'auto' 会把 LJ 作为软项混入特征相似性衰减；'lennard' 则以 LJ 为主；"
            "'electrostatic' 目前为距离衰减近似（若提供原子电荷可扩展为 Coulomb）。"
        )

    with tabs[2]:
        st.multiselect(
            "预测目标（可多选）",
            options=[
                "原子敏感性",
                "Binding affinity (protein-ligand)",
                "ADMET: MolLogP",
                "ADMET: Solubility (ESOL)",
                "ADMET: TPSA & MW",
                "PDE coefficients (CoeffNet)",
            ],
            default=st.session_state.get("ms_prediction_targets", ["原子敏感性"]),
            key="ms_prediction_targets",
        )

    with tabs[3]:
        st.caption("模型上传（可选）：提供训练好的模型以获得更可靠的预测。")
        st.markdown(
            "**ADMET 模型说明**: 上传的回归模型应以 joblib 保存（`.pkl`/`.joblib`），"
            "并以以下描述向量为输入顺序: [MolLogP, MolWt, RotatableBonds, TPSA, AromaticProportion]。\n"
            "若模型是 sklearn 风格，将使用 `n_features_in_` 做快速校验。"
        )
        admet_model_up = st.file_uploader(
            "上传 ADMET 回归模型 (joblib: .pkl/.joblib)",
            type=["pkl", "joblib"],
            key="ms_admet_model_up",
        )
        st.markdown("---")
        st.caption("对接模型上传（可选）：上传 PyTorch `.pth/.pt` 文件以用自定义权重进行预测")
        pl_model_up = st.file_uploader(
            "上传 对接 模型 (.pth/.pt, PyTorch)",
            type=["pth", "pt"],
            key="ms_pl_model_up",
        )
        st.caption(
            "若上传模型与默认架构不完全匹配，可选择隐藏维度并启用自动匹配尝试（会用 `strict=False` 加载匹配的参数）。"
        )
        pl_hidden_choice = st.selectbox(
            "PL 模型隐藏维度 (用于尝试构建模型以加载 state_dict)",
            options=[32, 64, 128, 256],
            index=1,
            key="ms_pl_hidden",
        )
        pl_autofit = st.checkbox(
            "尝试自动匹配并部分加载 state_dict (strict=False)",
            value=True,
            key="ms_pl_autofit",
        )
        st.caption("示例：蛋白口袋 CSV 样例下载")
        if st.button("下载蛋白口袋样例 CSV", key="ms_download_prot_example"):
            sample = pd.DataFrame(
                {
                    "element": ["C", "N", "O", "C"],
                    "x": [0.0, 1.2, -0.8, 0.5],
                    "y": [0.0, 0.1, -1.0, 1.5],
                    "z": [0.0, -0.2, 0.5, -0.7],
                }
            )
            buf = _io.StringIO()
            sample.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button(
                "下载示例 CSV",
                data=buf.getvalue(),
                file_name="protein_pocket_example.csv",
                mime="text/csv",
            )

    with tabs[4]:
        run = st.button("运行多尺度分析", key="ms_run")
        train_demo = st.button("训练并演示 PINN（本地短训）", key="ms_train_demo")
        train_epochs = st.number_input(
            "PINN 训练轮数（示例）",
            min_value=1, max_value=200,
            value=8, step=1,
            key="ms_train_epochs",
        )
        pinn_lr = st.number_input("PINN 学习率", min_value=1e-6, max_value=1e-1, value=1e-3, format="%g", key="ms_pinn_lr")
        pinn_weight_decay = st.number_input(
            "PINN weight_decay",
            min_value=0.0, max_value=1e-2,
            value=1e-4, format="%.6f",
            key="ms_pinn_wd",
        )
        pinn_lr_schedule = st.selectbox(
            "PINN 学习率调度",
            options=["cosine", "step", "none"],
            index=0,
            key="ms_pinn_sched",
        )
        pinn_step_size = st.number_input(
            "PINN 阶梯步长",
            min_value=1, max_value=200,
            value=20, step=1,
            key="ms_pinn_step",
        )
        pinn_gamma = st.number_input(
            "PINN 阶梯衰减系数",
            min_value=0.1, max_value=0.99,
            value=0.5, step=0.05,
            key="ms_pinn_gamma",
        )
        pinn_min_lr = st.number_input(
            "PINN 最小学习率",
            min_value=1e-8, max_value=1e-3,
            value=1e-6, format="%.8f",
            key="ms_pinn_minlr",
        )
        pinn_early_pat = st.number_input(
            "PINN 早停耐心",
            min_value=1, max_value=200,
            value=10, step=1,
            key="ms_pinn_pat",
        )
        pinn_max_grad = st.number_input(
            "PINN 梯度裁剪上限",
            min_value=0.0, max_value=100.0,
            value=5.0, step=0.5,
            key="ms_pinn_clip",
        )
        pinn_dropout = st.slider(
            "PINN Dropout",
            min_value=0.0, max_value=0.5,
            value=0.1, step=0.05,
            key="ms_pinn_dropout",
        )

    # 返回状态供主 UI 使用
    return


# ----------------------------------------------------------------------
# 多尺度主 UI
# ----------------------------------------------------------------------
def multiscale_ui() -> None:
    """多尺度 GNN-PINN 建模主 UI。

    使用原子级 GNN -> GAT -> PINN 的演示流水线。
    """
    st.header("多尺度 GNN-PINN 建模")
    st.write("使用原子级 GNN -> GAT -> PINN 的演示流水线。")

    with st.expander("控制面板", expanded=False):
        render_multiscale_sidebar()

    # 显示最近运行结果（如果有）
    scores = st.session_state.get("ms_last_scores")
    if scores:
        # 尝试生成交互式 SVG，若失败回退到静态图像
        svg_html = None
        try:
            svg_html = atom_heatmap_svg(
                st.session_state.get("ms_smiles", "CCO"),
                scores,
                width=int(st.session_state.get("ms_img_width", 640)),
                height=int(st.session_state.get("ms_img_height", 480)),
            )
        except Exception:
            svg_html = None

        if svg_html:
            components.html(svg_html, height=int(st.session_state.get("ms_img_height", 480)))
        else:
            img = atom_heatmap_image(
                st.session_state.get("ms_smiles", "CCO"),
                scores,
                size=(
                    int(st.session_state.get("ms_img_width", 640)),
                    int(st.session_state.get("ms_img_height", 480)),
                ),
            )
            if img is not None:
                st.image(
                    img,
                    caption="原子敏感性热图",
                    use_column_width=bool(st.session_state.get("ms_use_col_width", True)),
                    width=int(st.session_state.get("ms_img_width", 640)),
                )
            else:
                st.write("无法绘制分子图（可能缺少 RDKit 或 2D 坐标）。")

    # 处理 URL 查询参数中可能的 atom 选择
    params = st.query_params
    sel = None
    if "ms_atom" in params:
        try:
            sel = int(params["ms_atom"])
        except Exception:
            pass

    if sel is not None and scores:
        if 0 <= sel < len(scores):
            st.write(f"选中原子 {sel}，分数 = {scores[sel]:.4f}")

    # ------------------------------------------------------------------
    # 运行多尺度分析
    # ------------------------------------------------------------------
    if st.button("运行多尺度分析", key="ms_run_main"):
        smiles = st.session_state.get("ms_smiles", "CCO")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("SMILES 解析失败")
            return

        try:
            import torch
            from src.gnn import mol_to_graph, SimpleGNN, EnhancedGNN
            from src.multiscale import MultiScaleModel
            from src.pinn import pinn_loss

            graph = mol_to_graph(smiles)
            n_atoms = mol.GetNumAtoms()

            hidden_dim = int(st.session_state.get("ms_hidden", 64))
            model_type = str(st.session_state.get("ms_model_type", "EnhancedGNN"))

            if model_type == "SimpleGNN":
                gnn = SimpleGNN(
                    in_dim=graph.num_node_features,
                    hidden_dim=hidden_dim,
                    out_dim=1,
                    dropout=float(st.session_state.get("ms_gnn_dropout", 0.1)),
                )
            else:
                gnn = EnhancedGNN(
                    in_dim=graph.num_node_features,
                    hidden_dim=hidden_dim,
                    out_dim=1,
                    heads=int(st.session_state.get("ms_gat_heads", 4)),
                    dropout=float(st.session_state.get("ms_gnn_dropout", 0.1)),
                    use_physics=bool(st.session_state.get("ms_use_physics", True)),
                )

            msm = MultiScaleModel(
                gnn=gnn,
                readout_type=str(st.session_state.get("ms_readout", "mean")),
                enable_coeff=bool(st.session_state.get("ms_enable_coeff", False)),
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            msm.to(device)
            graph = graph.to(device)

            msm.eval()
            with torch.no_grad():
                node_emb = msm.gnn(graph)
                scores_arr = msm.attn_readout.score(node_emb).squeeze(-1)
                scores = scores_arr.cpu().numpy().tolist()

            st.session_state["ms_last_scores"] = scores

            # 重新绘制热图
            svg_html = atom_heatmap_svg(smiles, scores, width=640, height=480)
            if svg_html:
                components.html(svg_html, height=480)
            else:
                img = atom_heatmap_image(
                    smiles,
                    scores,
                    size=(640, 480),
                )
                if img:
                    st.image(img, caption="原子敏感性热图")

        except Exception as e:
            st.error(f"运行失败：{e}")
