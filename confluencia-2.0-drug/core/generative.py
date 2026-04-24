from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch optional
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, BRICS, Descriptors, Lipinski, QED
except Exception as exc:  # pragma: no cover
    raise ImportError("RDKit is required for molecule generation.") from exc


@dataclass
class GanConfig:
    latent_dim: int = 64
    hidden_dim: int = 256
    epochs: int = 200
    batch_size: int = 128
    lr: float = 2e-4
    device: str = "cpu"


@dataclass
class PropertyFilters:
    min_qed: Optional[float] = None
    min_mw: Optional[float] = None
    max_mw: Optional[float] = None
    min_logp: Optional[float] = None
    max_logp: Optional[float] = None
    max_hbd: Optional[int] = None
    max_hba: Optional[int] = None
    max_tpsa: Optional[float] = None


class _Generator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _ensure_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("torch is required for GAN-based generation. Install with the full profile.")


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    return mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_fp(smiles: str, radius: int, n_bits: int) -> Optional[np.ndarray]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def fingerprints_from_smiles(
    smiles_list: Sequence[str],
    *,
    radius: int,
    n_bits: int,
) -> Tuple[np.ndarray, List[str]]:
    fps = []
    valids: List[str] = []
    for s in smiles_list:
        fp = smiles_to_fp(s, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        fps.append(fp)
        valids.append(s)
    if not fps:
        raise ValueError("No valid SMILES found for fingerprinting.")
    return np.stack(fps, axis=0), valids


def train_fp_gan(fps: np.ndarray, cfg: GanConfig, seed: int = 42) -> _Generator:
    _ensure_torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    device = torch.device(cfg.device)
    data = torch.tensor(fps, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)

    gen = _Generator(cfg.latent_dim, cfg.hidden_dim, fps.shape[1]).to(device)
    disc = _Discriminator(fps.shape[1], cfg.hidden_dim).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=float(cfg.lr), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(cfg.lr), betas=(0.5, 0.999))
    bce = nn.BCELoss()

    for _ in range(int(cfg.epochs)):
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            # train discriminator
            z = torch.randn(batch_size, cfg.latent_dim, device=device)
            fake = gen(z).detach()

            pred_real = disc(real_batch)
            pred_fake = disc(fake)

            loss_d = bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # train generator
            z = torch.randn(batch_size, cfg.latent_dim, device=device)
            gen_samples = gen(z)
            pred = disc(gen_samples)
            loss_g = bce(pred, torch.ones_like(pred))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

    return gen


def sample_gan_fps(gen: _Generator, n_samples: int, cfg: GanConfig) -> np.ndarray:
    _ensure_torch()
    device = torch.device(cfg.device)
    gen.eval()
    with torch.no_grad():
        z = torch.randn(int(n_samples), cfg.latent_dim, device=device)
        out = gen(z).cpu().numpy().astype(np.float32)
    return out


def map_fps_to_smiles(
    fps: np.ndarray,
    ref_fps: np.ndarray,
    ref_smiles: Sequence[str],
    top_k: int = 1,
) -> List[str]:
    ref_norm = np.linalg.norm(ref_fps, axis=1) + 1e-9
    out: List[str] = []
    for fp in fps:
        fp_norm = np.linalg.norm(fp) + 1e-9
        sims = (ref_fps @ fp) / (ref_norm * fp_norm)
        idx = np.argsort(sims)[::-1][: int(top_k)]
        for i in idx:
            out.append(ref_smiles[int(i)])
    return out


def _random_atom_symbol(rng: random.Random) -> str:
    return rng.choice(["C", "N", "O", "S", "P", "F", "Cl", "Br"])


def _try_sanitize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mutate_smiles(smiles: str, rng: random.Random, max_tries: int = 8) -> Optional[str]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    for _ in range(int(max_tries)):
        rw = Chem.RWMol(mol)
        action = rng.choice(["replace", "add", "delete", "bond"])

        if action == "replace" and rw.GetNumAtoms() > 0:
            atom_idx = rng.randrange(rw.GetNumAtoms())
            atom = rw.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(Chem.Atom(_random_atom_symbol(rng)).GetAtomicNum())
        elif action == "add" and rw.GetNumAtoms() > 0:
            atom_idx = rng.randrange(rw.GetNumAtoms())
            new_atom = Chem.Atom(_random_atom_symbol(rng))
            new_idx = rw.AddAtom(new_atom)
            rw.AddBond(atom_idx, new_idx, order=Chem.BondType.SINGLE)
        elif action == "delete" and rw.GetNumAtoms() > 1:
            atom_idx = rng.randrange(rw.GetNumAtoms())
            rw.RemoveAtom(atom_idx)
        elif action == "bond" and rw.GetNumBonds() > 0:
            bond = rw.GetBondWithIdx(rng.randrange(rw.GetNumBonds()))
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            new_order = rng.choice([Chem.BondType.SINGLE, Chem.BondType.DOUBLE])
            rw.RemoveBond(a1, a2)
            rw.AddBond(a1, a2, order=new_order)
        else:
            continue

        new_mol = _try_sanitize(rw.GetMol())
        if new_mol is None:
            continue
        smi = Chem.MolToSmiles(new_mol, canonical=True)
        if smi:
            return smi

    return None


def crossover_smiles(smiles_a: str, smiles_b: str, rng: random.Random) -> Optional[str]:
    mol_a = smiles_to_mol(smiles_a)
    mol_b = smiles_to_mol(smiles_b)
    if mol_a is None or mol_b is None:
        return None

    frags_a = list(BRICS.BRICSDecompose(mol_a))
    frags_b = list(BRICS.BRICSDecompose(mol_b))
    if not frags_a or not frags_b:
        return None

    frags = frags_a + frags_b
    rng.shuffle(frags)
    try:
        for mol in BRICS.BRICSBuild(frags):
            smi = Chem.MolToSmiles(mol, canonical=True)
            if smi:
                return smi
    except Exception:
        return None
    return None


def default_score(smiles: str) -> float:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return float("-inf")
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("-inf")


def calc_props(smiles: str) -> Optional[Dict[str, float]]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        return {
            "qed": float(QED.qed(mol)),
            "mw": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "hbd": float(Lipinski.NumHDonors(mol)),
            "hba": float(Lipinski.NumHAcceptors(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
        }
    except Exception:
        return None


def passes_filters(smiles: str, filters: PropertyFilters) -> bool:
    props = calc_props(smiles)
    if props is None:
        return False
    if filters.min_qed is not None and props["qed"] < float(filters.min_qed):
        return False
    if filters.min_mw is not None and props["mw"] < float(filters.min_mw):
        return False
    if filters.max_mw is not None and props["mw"] > float(filters.max_mw):
        return False
    if filters.min_logp is not None and props["logp"] < float(filters.min_logp):
        return False
    if filters.max_logp is not None and props["logp"] > float(filters.max_logp):
        return False
    if filters.max_hbd is not None and props["hbd"] > float(filters.max_hbd):
        return False
    if filters.max_hba is not None and props["hba"] > float(filters.max_hba):
        return False
    if filters.max_tpsa is not None and props["tpsa"] > float(filters.max_tpsa):
        return False
    return True


def tanimoto_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0.5
    b_bin = b > 0.5
    inter = float(np.logical_and(a_bin, b_bin).sum())
    union = float(np.logical_or(a_bin, b_bin).sum())
    if union == 0:
        return 0.0
    return inter / union


def select_diverse_ranked(
    ranked: Sequence[Tuple[str, float]],
    *,
    radius: int,
    n_bits: int,
    max_sim: float,
) -> List[Tuple[str, float]]:
    if max_sim >= 1.0:
        return list(ranked)

    selected: List[Tuple[str, float]] = []
    selected_fps: List[np.ndarray] = []

    for smi, score in ranked:
        fp = smiles_to_fp(smi, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        if all(tanimoto_from_arrays(fp, prev) <= max_sim for prev in selected_fps):
            selected.append((smi, score))
            selected_fps.append(fp)

    return selected


def evolve_population(
    seed_smiles: Sequence[str],
    score_fn: Callable[[str], float],
    *,
    population_size: int,
    generations: int,
    elite_frac: float,
    mutation_rate: float,
    crossover_rate: float,
    rng: random.Random,
) -> List[Tuple[str, float]]:
    population = [s for s in seed_smiles if canonicalize_smiles(s)]
    if not population:
        raise ValueError("No valid seed SMILES available for evolution.")

    population = list(dict.fromkeys(population))
    if len(population) < population_size:
        population = (population * (population_size // len(population) + 1))[:population_size]

    scores = [score_fn(s) for s in population]

    for _ in range(int(generations)):
        ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        elite_n = max(1, int(math.ceil(population_size * elite_frac)))
        elites = [s for s, _ in ranked[:elite_n]]

        children: List[str] = []
        while len(children) + len(elites) < population_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)

            child = parent_a
            if rng.random() < crossover_rate:
                new_child = crossover_smiles(parent_a, parent_b, rng)
                if new_child:
                    child = new_child

            if rng.random() < mutation_rate:
                mutated = mutate_smiles(child, rng)
                if mutated:
                    child = mutated

            if canonicalize_smiles(child):
                children.append(child)

        population = list(dict.fromkeys(elites + children))
        if len(population) < population_size:
            population = (population * (population_size // len(population) + 1))[:population_size]

        scores = [score_fn(s) for s in population]

    final_ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
    return final_ranked


def generate_molecules(
    *,
    seed_smiles: Sequence[str],
    use_gan: bool,
    gan_cfg: GanConfig,
    radius: int,
    n_bits: int,
    gan_samples: int,
    score_fn: Callable[[str], float],
    population_size: int,
    generations: int,
    elite_frac: float,
    mutation_rate: float,
    crossover_rate: float,
    rng: random.Random,
) -> List[Tuple[str, float]]:
    seed_smiles = [s for s in seed_smiles if canonicalize_smiles(s)]
    if not seed_smiles:
        raise ValueError("seed_smiles is empty or invalid.")

    if use_gan:
        fps, valid_smiles = fingerprints_from_smiles(seed_smiles, radius=radius, n_bits=n_bits)
        gen = train_fp_gan(fps, gan_cfg, seed=int(rng.random() * 1e9))
        gen_fps = sample_gan_fps(gen, int(gan_samples), gan_cfg)
        gan_smiles = map_fps_to_smiles(gen_fps, fps, valid_smiles, top_k=1)
        seed_smiles = list(dict.fromkeys(list(seed_smiles) + gan_smiles))

    ranked = evolve_population(
        seed_smiles,
        score_fn,
        population_size=int(population_size),
        generations=int(generations),
        elite_frac=float(elite_frac),
        mutation_rate=float(mutation_rate),
        crossover_rate=float(crossover_rate),
        rng=rng,
    )
    return ranked
