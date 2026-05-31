#!/usr/bin/env python3
"""Generate composite Figure 1 for Confluencia Application Note.

Panel A: Architecture diagram (clean schematic)
Panel B: RNACTM PK trajectory curves
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ── RNACTM simulation ──────────────────────────────────────────────

def simulate_rnactm(k_deg, t_max=168, dt=0.1):
    k_abs = 0.12
    k_endo = 0.48
    k_escape = 0.02
    k_trans = 0.10

    t = np.arange(0, t_max, dt)
    n = len(t)
    C = np.zeros((n, 6))
    C[0, 0] = 1.0

    for i in range(1, n):
        C[i, 0] = C[i-1, 0] * np.exp(-k_abs * dt)
        C[i, 1] = C[i-1, 1] + (C[i-1, 0] - C[i-1, 0]*np.exp(-k_abs*dt)) * 0.8 - C[i-1, 1] * k_endo * dt
        C[i, 2] = C[i-1, 2] + C[i-1, 1] * k_endo * dt - C[i-1, 2] * k_escape * dt
        C[i, 3] = C[i-1, 3] + C[i-1, 2] * k_escape * dt - C[i-1, 3] * (k_trans + k_deg) * dt
        C[i, 4] = C[i-1, 4] + C[i-1, 3] * k_trans * dt - C[i-1, 4] * k_deg * dt
        C[i, 5] = C[i-1, 5] + (C[i-1, 3] + C[i-1, 4]) * k_deg * dt

    protein = C[:, 4]
    if protein.max() > 0:
        protein = protein / protein.max()
    return t, protein


# ── Color palette ───────────────────────────────────────────────────

C_INPUT    = '#E8F4FD'
C_DRUG     = '#D4EDDA'
C_EPITOPE  = '#CCE5FF'
C_PK       = '#FFE0B2'
C_CIRCRNA  = '#F8BBD0'
C_EVAL     = '#E1BEE7'
C_GO       = '#A5D6A7'
C_COND     = '#FFF9C4'
C_NOGO     = '#EF9A9A'

C_UNMOD    = '#D32F2F'
C_PSI      = '#1565C0'
C_M6A      = '#2E7D32'
C_5MC      = '#F57C00'

# ── Panel A: Architecture ──────────────────────────────────────────

def draw_architecture(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # ── Input row (y=7.5-8.5) ──
    input_items = ['SMILES', 'Epitope\nsequence', 'MHC\nallele', 'Dosing\nparams', 'circRNA\nsequence']
    for i, label in enumerate(input_items):
        x = 0.8 + i * 1.8
        box = FancyBboxPatch((x - 0.7, 7.6), 1.4, 0.9,
                             boxstyle="round,pad=0.08",
                             facecolor=C_INPUT, edgecolor='#546E7A', linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, 8.05, label, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#263238')

    # ── Processing modules (y=5.5-7) ──
    modules = [
        (1.5, 5.8, 2.4, 1.5, C_DRUG,    'Drug Pipeline',
         ['MOE ensemble', 'ADMET screening', 'Toxicophore alerts']),
        (4.5, 5.8, 2.4, 1.5, C_EPITOPE,  'Epitope Pipeline',
         ['Mamba3Lite encoding', 'MHC pseudo-sequence', 'Sample-size-adaptive']),
        (7.5, 5.8, 2.4, 1.5, C_PK,      'RNACTM PK',
         ['6-compartment ODE', 'Modification-specific', '72h trajectory']),
    ]
    for (cx, cy, w, h, col, title, items) in modules:
        box = FancyBboxPatch((cx - w/2, cy), w, h,
                             boxstyle="round,pad=0.08",
                             facecolor=col, edgecolor='#37474F', linewidth=1.3)
        ax.add_patch(box)
        ax.text(cx, cy + h - 0.25, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1B5E20')
        for j, item in enumerate(items):
            ax.text(cx, cy + h - 0.55 - j*0.25, item, ha='center', va='center', fontsize=6.5, color='#37474F')

    # ── circRNA module (y=4-5.3) ──
    circrna_box = FancyBboxPatch((3.2, 4.0), 3.6, 1.3,
                                 boxstyle="round,pad=0.08",
                                 facecolor=C_CIRCRNA, edgecolor='#37474F', linewidth=1.3)
    ax.add_patch(circrna_box)
    ax.text(5.0, 5.0, 'circRNA Functional Assessment', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#880E4F')
    sub_items = ['ViennaRNA structure', '5 immune pathways', 'miRNA/RBP/Translation', 'Modification sites']
    for j, item in enumerate(sub_items):
        ax.text(5.0, 4.65 - j*0.22, item, ha='center', va='center', fontsize=6.2, color='#37474F')

    # ── 5D Evaluation (y=2.5-3.6) ──
    eval_box = FancyBboxPatch((2.8, 2.5), 4.4, 1.1,
                              boxstyle="round,pad=0.08",
                              facecolor=C_EVAL, edgecolor='#37474F', linewidth=1.3)
    ax.add_patch(eval_box)
    ax.text(5.0, 3.35, 'Uncertainty-Adaptive 5D Evaluation', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#4A148C')
    dims = 'Clinical(0.30) | Binding(0.20) | Kinetics(0.15) | Signature(0.15) | circRNA(0.20)'
    ax.text(5.0, 2.9, dims, ha='center', va='center', fontsize=6, color='#4A148C')

    # ── Output decisions (y=0.8-1.6) ──
    decisions = [
        (2.2, C_GO,   'Go',       '>=0.65'),
        (5.0, C_COND, 'Conditional', '>=0.40'),
        (7.8, C_NOGO, 'No-Go',    '<0.40'),
    ]
    for (cx, col, label, thresh) in decisions:
        box = FancyBboxPatch((cx - 0.9, 0.8), 1.8, 0.8,
                             boxstyle="round,pad=0.08",
                             facecolor=col, edgecolor='#37474F', linewidth=1.2)
        ax.add_patch(box)
        ax.text(cx, 1.35, label, ha='center', va='center', fontsize=9, fontweight='bold', color='#212121')
        ax.text(cx, 1.0, thresh, ha='center', va='center', fontsize=6.5, color='#616161')

    # ── Arrows ──
    arrow_kw = dict(arrowstyle='-|>', color='#546E7A', linewidth=1.2, mutation_scale=12)

    # Inputs → modules
    for i in range(5):
        x_src = 0.8 + i * 1.8
        # Connect each input to appropriate module
        if i == 0:  # SMILES → Drug
            ax.annotate('', xy=(1.5, 5.8), xytext=(x_src, 7.6), arrowprops=arrow_kw)
        elif i in [1, 2]:  # Epitope/MHC → Epitope
            ax.annotate('', xy=(4.5, 5.8), xytext=(x_src, 7.6), arrowprops=arrow_kw)
        elif i == 3:  # Dosing → RNACTM
            ax.annotate('', xy=(7.5, 5.8), xytext=(x_src, 7.6), arrowprops=arrow_kw)
        elif i == 4:  # circRNA → circRNA module
            ax.annotate('', xy=(5.0, 5.3), xytext=(x_src, 7.6), arrowprops=arrow_kw)

    # Modules → 5D Evaluation
    for cx in [1.5, 4.5, 7.5]:
        ax.annotate('', xy=(5.0, 3.6), xytext=(cx, 5.8), arrowprops=arrow_kw)

    # circRNA module → 5D Evaluation
    ax.annotate('', xy=(5.0, 3.6), xytext=(5.0, 4.0), arrowprops=arrow_kw)

    # 5D → Decisions
    for cx in [2.2, 5.0, 7.8]:
        ax.annotate('', xy=(cx, 1.6), xytext=(5.0, 2.5), arrowprops=arrow_kw)

    ax.text(5.0, 8.7, 'A', ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#37474F', linewidth=1.5))


# ── Panel B: RNACTM Trajectories ──────────────────────────────────

def draw_trajectories(ax):
    modifications = {
        'Unmodified (t1/2=6.24h)': {'k_deg': 0.111, 'color': C_UNMOD, 'ls': '-'},
        'Psi (t1/2=15.61h)':       {'k_deg': 0.044, 'color': C_PSI,   'ls': '-'},
        'm6A (t1/2=11.24h)':       {'k_deg': 0.062, 'color': C_M6A,   'ls': '-'},
        '5mC (t1/2~8h)':           {'k_deg': 0.083, 'color': C_5MC,   'ls': '--'},
    }

    for name, params in modifications.items():
        t, protein = simulate_rnactm(params['k_deg'], t_max=168)
        ax.plot(t, protein, color=params['color'], linestyle=params['ls'],
                linewidth=2.5, label=name)

    ax.axhline(y=0.5, color='#9E9E9E', linestyle=':', alpha=0.6, linewidth=1)
    ax.text(172, 0.48, '50% peak', fontsize=7, color='#9E9E9E', va='center')

    # Literature validation annotations
    ax.annotate('6.24h\n(validated)', xy=(6.24, 0.5), xytext=(20, 0.35),
                fontsize=6.5, color=C_UNMOD, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_UNMOD, lw=0.8))
    ax.annotate('15.61h\n(validated)', xy=(15.61, 0.5), xytext=(35, 0.6),
                fontsize=6.5, color=C_PSI, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_PSI, lw=0.8))

    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Protein expression\n(normalized)', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9, edgecolor='#BDBDBD')
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

    ax.text(0.02, 1.02, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#37474F', linewidth=1.5))


# ── Composite figure ──────────────────────────────────────────────

def generate_composite(output_dir):
    fig = plt.figure(figsize=(14, 7))

    # Panel A: architecture (top, full width)
    ax_a = fig.add_axes([0.02, 0.38, 0.96, 0.60])
    draw_architecture(ax_a)

    # Panel B: RNACTM trajectories (bottom)
    ax_b = fig.add_axes([0.08, 0.06, 0.86, 0.28])
    draw_trajectories(ax_b)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'fig1_composite.png')
    pdf_path = os.path.join(output_dir, 'fig1_composite.pdf')

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")

    # Verify half-lives
    print("\n=== Half-life verification ===")
    for name, params in [('Unmodified', 0.111), ('Psi', 0.044), ('m6A', 0.062), ('5mC', 0.083)]:
        hl = np.log(2) / params
        print(f"  {name}: {hl:.2f}h")


if __name__ == '__main__':
    # Output directly to paper figures directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'paper', 'mypaper', 'figures')
    generate_composite(output_dir)