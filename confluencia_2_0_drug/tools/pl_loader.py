"""Protein-ligand data loader utilities.

Provides:
- `load_complex_from_pdb(pdb_path, ligand_resname=None, pocket_radius=6.0)` -> returns (ligand_smiles, prot_atom_types, prot_coords)
- `iter_dataset_from_csv(csv_path, smiles_col, pdb_col, ligand_resname_col=None)` -> yields tuples for training

Biopython (`Bio.PDB`) is used if available for robust PDB parsing; otherwise functions will raise informative errors.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import numpy as np

try:
    from Bio.PDB import PDBParser, NeighborSearch
    _HAS_BIO = True
except Exception:
    _HAS_BIO = False

from rdkit import Chem


def load_complex_from_pdb(pdb_path: str, ligand_resname: Optional[str] = None, pocket_radius: float = 6.0) -> Tuple[str, List[str], np.ndarray]:
    """Load ligand SMILES and protein pocket atom types + coords from a PDB file.

    - `ligand_resname`: if provided, selects ligand by residue name; otherwise selects the largest hetero residue.
    Returns (ligand_smiles, prot_atom_types, prot_coords)
    """
    if not _HAS_BIO:
        raise RuntimeError("Biopython not available: install biopython to use PDB loader (pip install biopython).")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_path)

    # collect hetero residues as candidate ligands
    lig_residues = []
    prot_atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                hetflag = res.id[0].strip()
                if hetflag != '':
                    lig_residues.append(res)
                else:
                    for atom in res:
                        prot_atoms.append(atom)

    # choose ligand residue
    lig = None
    if ligand_resname is not None:
        for r in lig_residues:
            if r.get_resname().strip() == ligand_resname:
                lig = r
                break
    if lig is None:
        # pick largest ligand by atom count
        if lig_residues:
            lig = max(lig_residues, key=lambda r: len(list(r.get_atoms())))
        else:
            raise RuntimeError('No hetero ligand found in PDB; specify ligand_resname or provide processed files.')

    # build ligand mol via RDKit from PDB coordinates (approximate conversion)
    # write temp PDB string
    pdb_lines = []
    for atom in lig.get_atoms():
        coord = atom.get_coord()
        el = atom.element.strip() or atom.get_name()[0]
        pdb_lines.append(f"HETATM    1  {atom.get_name():>3} {lig.get_resname():>3} A   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {el:>2}")
    pdb_block = "\n".join(pdb_lines)
    rdkit_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
    if rdkit_mol is None:
        # fallback: attempt to build SMILES from residue atoms' element symbols (approximate)
        smi = ''.join([atom.element for atom in lig.get_atoms() if hasattr(atom, 'element')])
        ligand_smiles = smi
    else:
        try:
            ligand_smiles = Chem.MolToSmiles(rdkit_mol)
        except Exception:
            ligand_smiles = Chem.MolToSmiles(rdkit_mol)

    # find pocket atoms within pocket_radius of any ligand atom
    lig_coords = [a.get_coord() for a in lig.get_atoms()]
    ns = NeighborSearch(list(prot_atoms))
    nearby = set()
    for c in lig_coords:
        neighbors = ns.search(c, pocket_radius)
        for a in neighbors:
            nearby.add(a)

    prot_atom_types = [a.element for a in nearby]
    prot_coords = np.array([a.get_coord() for a in nearby], dtype=np.float32)

    return ligand_smiles, prot_atom_types, prot_coords


def iter_dataset_from_csv(csv_path: str, smiles_col: str, pdb_col: str, ligand_resname_col: Optional[str] = None) -> Iterable[Tuple[str, List[str], np.ndarray]]:
    import csv
    with open(csv_path, newline='', encoding='utf-8') as fh:
        r = csv.DictReader(fh)
        for row in r:
            smiles = row[smiles_col]
            pdb_path = row[pdb_col]
            lig_res = row[ligand_resname_col] if ligand_resname_col and ligand_resname_col in row else None
            try:
                s, atypes, coords = load_complex_from_pdb(pdb_path, ligand_resname=lig_res)
            except Exception as e:
                # skip with warning
                print(f"Skipping {pdb_path}: {e}")
                continue
            yield s, atypes, coords
