from __future__ import annotations

from typing import Any, Dict, Optional

import base64
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from src.gnn import mol_to_graph
from src.gnn_sensitivity import sensitivity_masking
from src.multiscale import MultiScaleModel
from src.common.training import EarlyStopping, build_scheduler
from src.pinn import pinn_loss
from src.pl_train import train as pl_train_fn

app = FastAPI(title="IGEM Multiscale Worker")


def _build_gnn(
    *,
    in_dim: int,
    steps: int,
    hidden: int,
    model_type: str,
    use_physics: bool,
    gnn_dropout: float,
    lj_eps: float,
    lj_sigma: float,
    dielectric: float,
) -> torch.nn.Module:
    model_type = str(model_type or "SimpleGNN")
    if model_type == "EnhancedGNN":
        from src.gnn import EnhancedGNN

        return EnhancedGNN(in_dim, hidden_dim=hidden, steps=steps, gat_heads=4, use_physics=use_physics, dropout=float(gnn_dropout))
    if model_type == "PhysicsMessageGNN":
        from src.gnn import PhysicsMessageGNN

        return PhysicsMessageGNN(
            in_dim,
            hidden_dim=hidden,
            steps=steps,
            potential_type="auto",
            lj_epsilon=lj_eps,
            lj_sigma=lj_sigma,
            dielectric=dielectric,
            dropout=float(gnn_dropout),
        )
    from src.gnn import SimpleGNN

    return SimpleGNN(in_dim, hidden_dim=hidden, steps=steps, dropout=float(gnn_dropout))


class TaskPayload(BaseModel):
    task: str
    payload: Optional[Dict[str, Any]] = None


def mol_to_png_b64(smiles: str, scores: Dict[int, float], size=(480, 360)) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return base64.b64encode(png).decode('ascii')


@app.post("/task")
async def handle_task(task: TaskPayload):
    t = task.task
    p = task.payload or {}
    try:
        if t == "multiscale":
            smiles = p.get("smiles")
            if not smiles:
                raise HTTPException(status_code=400, detail="missing smiles")
            X, A, mol = mol_to_graph(smiles)
            in_dim = X.shape[1]

            steps = int(p.get("steps", 3))
            hidden = int(p.get("hidden", 64))
            model_type = str(p.get("model_type", "SimpleGNN"))
            use_physics = bool(p.get("use_physics", True))
            gnn_dropout = float(p.get("gnn_dropout", 0.1))
            lj_eps = float(p.get("lj_eps", 0.1))
            lj_sigma = float(p.get("lj_sigma", 3.5))
            dielectric = float(p.get("dielectric", 80.0))

            gnn = _build_gnn(
                in_dim=in_dim,
                steps=steps,
                hidden=hidden,
                model_type=model_type,
                use_physics=use_physics,
                gnn_dropout=gnn_dropout,
                lj_eps=lj_eps,
                lj_sigma=lj_sigma,
                dielectric=dielectric,
            )
            model_fn, _ = None, None
            try:
                from src.gnn_sensitivity import example_model_fn_factory

                model_fn, _ = example_model_fn_factory(64)
            except Exception:
                model_fn = lambda x: x.mean(dim=1)

            scores = sensitivity_masking(smiles, gnn, model_fn)

            # build multiscale, short pinn train
            readout_type = str(p.get("readout_type", "mean"))
            msm = MultiScaleModel(gnn, readout=readout_type)
            mol_emb = msm.encode_molecule(smiles)
            msm.build_pinn(spatial_dim=1, mol_emb_dim=mol_emb.shape[0], hidden=64)

            pinn_losses = []
            try:
                opt = torch.optim.Adam(msm.pinn.parameters(), lr=1e-3)
                for ep in range(8):
                    pts = torch.rand((64, 2))
                    loss = pinn_loss(msm.pinn, pts, mol_emb.detach(), D=0.1, Vmax=0.5, Km=0.1)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    pinn_losses.append(float(loss.item()))
            except Exception:
                pinn_losses = []

            png_b64 = mol_to_png_b64(smiles, scores)
            return {"ok": True, "scores": scores, "pinn_losses": pinn_losses, "png_b64": png_b64}

        elif t == "mask_plot":
            smiles = p.get("smiles")
            atom_idx = int(p.get("atom_idx", 0))
            x_min = float(p.get("x_min", 0.0))
            x_max = float(p.get("x_max", 1.0))
            n_pts = int(p.get("n_pts", 101))
            t_val = float(p.get("t_val", 0.1))

            X, A, mol = mol_to_graph(smiles)
            in_dim = X.shape[1]
            steps = int(p.get("steps", 3))
            hidden = int(p.get("hidden", 64))
            model_type = str(p.get("model_type", "SimpleGNN"))
            use_physics = bool(p.get("use_physics", True))
            gnn_dropout = float(p.get("gnn_dropout", 0.1))
            lj_eps = float(p.get("lj_eps", 0.1))
            lj_sigma = float(p.get("lj_sigma", 3.5))
            dielectric = float(p.get("dielectric", 80.0))

            gnn = _build_gnn(
                in_dim=in_dim,
                steps=steps,
                hidden=hidden,
                model_type=model_type,
                use_physics=use_physics,
                gnn_dropout=gnn_dropout,
                lj_eps=lj_eps,
                lj_sigma=lj_sigma,
                dielectric=dielectric,
            )
            model_fn, _ = None, None
            try:
                from src.gnn_sensitivity import example_model_fn_factory

                model_fn, _ = example_model_fn_factory(64)
            except Exception:
                model_fn = lambda x: x.mean(dim=1)

            scores = sensitivity_masking(smiles, gnn, model_fn)

            readout_type = str(p.get("readout_type", "mean"))
            msm = MultiScaleModel(gnn, readout=readout_type)
            mol_emb = msm.encode_molecule(smiles)
            msm.build_pinn(spatial_dim=1, mol_emb_dim=mol_emb.shape[0], hidden=64)

            xs = torch.linspace(x_min, x_max, n_pts).unsqueeze(1)
            ts = torch.full((xs.shape[0], 1), t_val)
            baseline_x = torch.cat([xs, ts], dim=1)

            with torch.no_grad():
                inp = torch.cat([baseline_x, mol_emb.unsqueeze(0).repeat(baseline_x.shape[0], 1)], dim=1)
                y_orig = msm.pinn(inp).detach().cpu().numpy()

                emb_mask = msm.encode_with_mask(smiles, [atom_idx])
                inp2 = torch.cat([baseline_x, emb_mask.unsqueeze(0).repeat(baseline_x.shape[0], 1)], dim=1)
                y_mask = msm.pinn(inp2).detach().cpu().numpy()

            diff = (y_mask - y_orig).tolist()
            return {"ok": True, "x": xs.squeeze().tolist(), "y_orig": y_orig.tolist(), "y_mask": y_mask.tolist(), "diff": diff}

        elif t == "ms_pinn_train":
            smiles = p.get("smiles")
            if not smiles:
                raise HTTPException(status_code=400, detail="missing smiles")

            steps = int(p.get("steps", 3))
            hidden = int(p.get("hidden", 64))
            model_type = str(p.get("model_type", "SimpleGNN"))
            enable_coeff = bool(p.get("enable_coeff", False))
            coeff_hidden = int(p.get("coeff_hidden", 64))
            readout_type = str(p.get("readout_type", "mean"))
            train_epochs = int(p.get("train_epochs", 8))

            lj_eps = float(p.get("lj_eps", 0.1))
            lj_sigma = float(p.get("lj_sigma", 3.5))
            dielectric = float(p.get("dielectric", 80.0))
            use_physics = bool(p.get("use_physics", True))

            pinn_lr = float(p.get("pinn_lr", 1e-3))
            pinn_weight_decay = float(p.get("pinn_weight_decay", 1e-4))
            pinn_lr_schedule = str(p.get("pinn_lr_schedule", "cosine"))
            pinn_step_size = int(p.get("pinn_step_size", 20))
            pinn_gamma = float(p.get("pinn_gamma", 0.5))
            pinn_min_lr = float(p.get("pinn_min_lr", 1e-6))
            pinn_early_pat = int(p.get("pinn_early_pat", 10))
            pinn_max_grad = float(p.get("pinn_max_grad", 5.0))
            pinn_dropout = float(p.get("pinn_dropout", 0.1))
            D = float(p.get("D", 0.1))
            Vmax = float(p.get("Vmax", 0.5))
            Km = float(p.get("Km", 0.1))

            X, A, mol = mol_to_graph(smiles)
            in_dim = X.shape[1]
            from src.gnn import SimpleGNN, EnhancedGNN, PhysicsMessageGNN

            if model_type == "EnhancedGNN":
                gnn = EnhancedGNN(in_dim, hidden_dim=hidden, steps=steps, gat_heads=4, use_physics=use_physics, dropout=0.0)
            elif model_type == "PhysicsMessageGNN":
                gnn = PhysicsMessageGNN(in_dim, hidden_dim=hidden, steps=steps, potential_type="auto", lj_epsilon=lj_eps, lj_sigma=lj_sigma, dielectric=dielectric, dropout=0.0)
            else:
                gnn = SimpleGNN(in_dim, hidden_dim=hidden, steps=steps, dropout=0.0)

            msm = MultiScaleModel(gnn, readout=readout_type)
            mol_emb = msm.encode_molecule(smiles)
            msm.build_pinn(spatial_dim=1, mol_emb_dim=mol_emb.shape[0], hidden=64, dropout=float(pinn_dropout))
            if enable_coeff:
                msm.build_coeff_net(mol_emb_dim=mol_emb.shape[0], hidden=int(coeff_hidden))

            coeff_fn = msm.coeff_net if (enable_coeff and getattr(msm, "coeff_net", None) is not None) else None
            optimizer = torch.optim.AdamW(msm.pinn.parameters(), lr=pinn_lr, weight_decay=pinn_weight_decay)
            scheduler = build_scheduler(optimizer, pinn_lr_schedule, epochs=int(train_epochs), step_size=int(pinn_step_size), gamma=float(pinn_gamma), min_lr=float(pinn_min_lr))
            stopper = EarlyStopping(patience=int(pinn_early_pat), mode="min")

            losses = []
            for ep in range(int(train_epochs)):
                pts = torch.rand((64, 2))
                optimizer.zero_grad()
                loss = pinn_loss(msm.pinn, pts, mol_emb.detach(), D=D, Vmax=Vmax, Km=Km, coeff_fn=coeff_fn)
                loss.backward()
                if float(pinn_max_grad) > 0:
                    torch.nn.utils.clip_grad_norm_(msm.pinn.parameters(), max_norm=float(pinn_max_grad))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                losses.append(float(loss.item()))
                if stopper.step(float(loss.item())):
                    break

            return {"ok": True, "pinn_losses": losses}
        elif t == "pl_train":
            epochs = int(p.get("epochs", 10))
            batch = int(p.get("batch", 8))
            lr = float(p.get("lr", 1e-3))
            out_name = str(p.get("out_name", "pl_model.pth"))
            lj_eps = float(p.get("lj_eps", 0.1))
            lj_sigma = float(p.get("lj_sigma", 3.5))
            dielectric = float(p.get("dielectric", 80.0))
            use_cuda = bool(p.get("cuda", False))
            weight_decay = float(p.get("weight_decay", 1e-4))
            lr_schedule = str(p.get("lr_schedule", "cosine"))
            step_size = int(p.get("step_size", 20))
            gamma = float(p.get("gamma", 0.5))
            min_lr = float(p.get("min_lr", 1e-6))
            early_patience = int(p.get("early_patience", 10))
            max_grad_norm = float(p.get("max_grad_norm", 5.0))
            dropout = float(p.get("dropout", 0.1))
            use_lstm = bool(p.get("use_lstm", False))
            lstm_hidden = int(p.get("lstm_hidden", 128))
            lstm_layers = int(p.get("lstm_layers", 1))
            lstm_bi = bool(p.get("lstm_bi", True))
            distill_weight = float(p.get("distill_weight", 0.2))

            teacher_path = str(p.get("teacher_path") or "").strip() or None
            teacher_b64 = p.get("teacher_b64")
            if teacher_path is None and isinstance(teacher_b64, str) and teacher_b64.strip():
                try:
                    raw = base64.b64decode(teacher_b64.encode("ascii"))
                    tmp_dir = Path("tmp")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth", dir=str(tmp_dir)) as tf:
                        tf.write(raw)
                        teacher_path = tf.name
                except Exception:
                    teacher_path = None

            # synthetic smiles list (same as pl_train demo)
            smiles_list = [
                "CCO", "CC", "CCC", "CCN", "CCOCC", "c1ccccc1", "CC(=O)O", "CC(C)O", "OCCO", "CCS"
            ] * 10

            out_path = Path("tmp") / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pl_train_fn(
                smiles_list,
                epochs=epochs,
                batch_size=batch,
                lr=lr,
                out=str(out_path),
                lj_epsilon=lj_eps,
                lj_sigma=lj_sigma,
                dielectric=dielectric,
                use_cuda=use_cuda,
                seed=42,
                dropout=dropout,
                use_lstm=use_lstm,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                lstm_bidirectional=lstm_bi,
                weight_decay=weight_decay,
                lr_schedule=lr_schedule,
                step_size=step_size,
                gamma=gamma,
                min_lr=min_lr,
                early_stopping_patience=early_patience,
                max_grad_norm=max_grad_norm,
                teacher_path=teacher_path,
                distill_weight=distill_weight,
            )

            model_b64 = base64.b64encode(out_path.read_bytes()).decode("ascii") if out_path.exists() else ""
            return {"ok": True, "model_b64": model_b64, "out_name": out_name}
        elif t == "literature_autolearn":
            from src.common.literature_autolearn import literature_autolearn

            query = str(p.get("query", "")).strip()
            domain = str(p.get("domain", "custom")).strip()
            keywords = p.get("keywords")
            sources = p.get("sources")
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(";") if k.strip()]
            elif not isinstance(keywords, list):
                keywords = None

            if isinstance(sources, str):
                sources = [s.strip() for s in sources.split(";") if s.strip()]
            elif not isinstance(sources, list):
                sources = None

            year_from = p.get("year_from")
            year_to = p.get("year_to")
            max_results = int(p.get("max_results", 20))
            include_preprints = bool(p.get("include_preprints", False))
            timeout = float(p.get("timeout", 30.0))
            retries = int(p.get("retries", 3))
            backoff_factor = float(p.get("backoff_factor", 0.5))
            user_agent = str(p.get("user_agent", "literature-autolearn/1.0 (research; contact: local)"))
            cache_dir = p.get("cache_dir")
            include_csv = bool(p.get("include_csv", True))

            result = literature_autolearn(
                query=query,
                domain=domain,
                keywords=keywords,
                sources=sources,
                year_from=int(year_from) if year_from else None,
                year_to=int(year_to) if year_to else None,
                include_preprints=include_preprints,
                max_results=max_results,
                timeout=timeout,
                retries=retries,
                backoff_factor=backoff_factor,
                user_agent=user_agent,
                cache_dir=cache_dir,
                include_csv=include_csv,
            )
            result["ok"] = True
            return result
        elif t == "literature_dataset_fetch":
            from src.common.dataset_autofetch import fetch_datasets

            urls = p.get("urls") or []
            if isinstance(urls, str):
                urls = [u.strip() for u in urls.split(";") if u.strip()]
            if not isinstance(urls, list):
                urls = []

            domain = str(p.get("domain", "custom")).strip()
            cache_dir = p.get("cache_dir", "data/cache/http")
            timeout = float(p.get("timeout", 30.0))
            sleep_seconds = float(p.get("sleep", 0.0))
            max_rows = int(p.get("max_rows", 5000))

            result = fetch_datasets(
                urls=urls,
                domain=domain,
                cache_dir=cache_dir,
                timeout=timeout,
                sleep_seconds=sleep_seconds,
                max_rows_per_file=max_rows,
            )
            result["ok"] = True
            return result
        else:
            raise HTTPException(status_code=400, detail=f"unknown task {t}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8765)
