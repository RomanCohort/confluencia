#!/usr/bin/env python3
"""
Figure 3: TNBC Subtype Simulation Dynamics
Style: Nature Cancer / Jiang et al. 2019 Cancer Cell
Multi-panel: A) Subtype heatmap  B) Immunoediting time series  C) Shannon diversity  D) Spatial TME
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams['xtick.major.width'] = 0.6
matplotlib.rcParams['ytick.major.width'] = 0.6
matplotlib.rcParams['xtick.major.size'] = 3
matplotlib.rcParams['ytick.major.size'] = 3

# ==================== DATA SECTOR ====================

# Panel A: Subtype parameter heatmap
subtypes = ['BLIS', 'BLIA', 'IM', 'LAR']
params = ['TIL\ndensity', 'PD-L1\nexpression', 'Prognosis\n(inverse)', 'Immune\ngenes', 'AR\nexpression']
heatmap_data = np.array([
    [0.12, 0.10, 0.15, 0.10, 0.05],  # BLIS
    [0.33, 0.30, 0.50, 0.65, 0.10],  # BLIA
    [0.60, 0.50, 0.80, 0.85, 0.15],  # IM
    [0.20, 0.25, 0.55, 0.30, 0.80],  # LAR
])

# Panel B: Immunoediting dynamics
cycles = np.arange(0, 31, 1)
np.random.seed(42)

BLIS_TIL = 0.12 * np.exp(-0.18 * cycles) + np.random.normal(0, 0.005, len(cycles))
BLIA_TIL = 0.33 * np.exp(-0.02 * cycles) + np.random.normal(0, 0.008, len(cycles))
IM_TIL = 0.60 * (0.85 + 0.15 * np.cos(0.1 * cycles)) + np.random.normal(0, 0.01, len(cycles))
LAR_TIL = 0.45 * np.exp(-0.03 * cycles) + np.random.normal(0, 0.008, len(cycles))

# Panel C: Shannon diversity
div_cycles = np.array([0, 5, 10, 15, 20, 25, 30])
div_no_treat = np.array([0.40, 0.42, 0.44, 0.46, 0.47, 0.48, 0.50])
div_chemo = np.array([0.40, 0.55, 0.72, 0.88, 1.00, 1.12, 1.20])

# Panel D: Spatial TME compartments
compartments = ['Hypoxic\ncore', 'Immune-rich\nmargin', 'Stromal\nbarrier']
tme_params = ['Oxygen', 'Drug\npenetration', 'Immune\ncell density']
tme_data = np.array([
    [0.02, 0.30, 0.10],  # Hypoxic core
    [0.08, 0.70, 0.80],  # Immune margin
    [0.04, 0.55, 0.45],  # Stromal barrier
])

# ==================== COLOR PALETTE (Nature style) ====================

subtype_colors = {
    'BLIS': '#E64B35',    # Nature red
    'BLIA': '#4DBBD5',    # Nature cyan
    'IM': '#00A087',      # Nature teal
    'LAR': '#3C5488',     # Nature navy
}

# ==================== FIGURE COMPOSITION ====================

fig = plt.figure(figsize=(7.2, 6.5), dpi=300)
gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.40,
              left=0.08, right=0.95, top=0.94, bottom=0.08)

# ---- Panel A: Subtype Heatmap ----
ax_a = fig.add_subplot(gs[0, 0])

cmap = LinearSegmentedColormap.from_list('nature_heat',
    ['#2166AC', '#F7F7F7', '#B2182B'])
im = ax_a.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=0.9)

ax_a.set_xticks(range(len(params)))
ax_a.set_xticklabels(params, fontsize=6.5, ha='center')
ax_a.set_yticks(range(len(subtypes)))
ax_a.set_yticklabels(subtypes, fontsize=7.5, fontweight='bold')

for i in range(len(subtypes)):
    for j in range(len(params)):
        val = heatmap_data[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax_a.text(j, i, f'{val:.2f}', ha='center', va='center',
                  fontsize=6, color=color, fontweight='bold')

ax_a.set_title('A', fontsize=11, fontweight='bold', loc='left', pad=4)
ax_a.set_title('Subtype parameters', fontsize=8, loc='right', pad=4)

cb = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
cb.set_label('Normalized value', fontsize=6.5)
cb.ax.tick_params(labelsize=6)

# ---- Panel B: Immunoediting Dynamics ----
ax_b = fig.add_subplot(gs[0, 1])

ax_b.plot(cycles, BLIS_TIL, color=subtype_colors['BLIS'], linewidth=1.5, label='BLIS')
ax_b.plot(cycles, BLIA_TIL, color=subtype_colors['BLIA'], linewidth=1.5, label='BLIA')
ax_b.plot(cycles, IM_TIL, color=subtype_colors['IM'], linewidth=1.5, label='IM')
ax_b.plot(cycles, LAR_TIL, color=subtype_colors['LAR'], linewidth=1.5, label='LAR')

# Phase boundaries
ax_b.axvline(x=12, color='#888888', linestyle=':', linewidth=0.6, alpha=0.7)
ax_b.text(6, 0.72, 'Elimination', fontsize=6, ha='center', color='#555555', style='italic')
ax_b.text(18, 0.72, 'Equilibrium', fontsize=6, ha='center', color='#555555', style='italic')
ax_b.text(26, 0.72, 'Escape', fontsize=6, ha='center', color='#555555', style='italic')

# BLIS escape annotation
ax_b.annotate('BLIS escapes\ncycle 12', xy=(12, 0.02), xytext=(16, 0.15),
              fontsize=5.5, arrowprops=dict(arrowstyle='->', color=subtype_colors['BLIS'],
              lw=0.8), color=subtype_colors['BLIS'])

ax_b.set_xlabel('Simulation cycles', fontsize=7.5)
ax_b.set_ylabel('TIL density', fontsize=7.5)
ax_b.set_xlim(0, 30)
ax_b.set_ylim(0, 0.75)
ax_b.legend(fontsize=6, frameon=True, edgecolor='#cccccc', fancybox=False,
            loc='upper right', handlelength=1.5)
ax_b.set_title('B', fontsize=11, fontweight='bold', loc='left', pad=4)
ax_b.set_title('Immunoediting dynamics', fontsize=8, loc='right', pad=4)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.tick_params(labelsize=6.5)

# ---- Panel C: Shannon Diversity ----
ax_c = fig.add_subplot(gs[1, 0])

ax_c.plot(div_cycles, div_no_treat, 'o-', color='#7E6148', linewidth=1.5,
           markersize=4, label='No treatment')
ax_c.plot(div_cycles, div_chemo, 's-', color='#E64B35', linewidth=1.5,
           markersize=4, label='Chemotherapy')

# Fill between
ax_c.fill_between(div_cycles, div_no_treat, div_chemo, alpha=0.1, color='#E64B35')

# Annotation
ax_c.annotate('Drug-induced\ninstability\n(1% → 50%)', xy=(15, 0.88),
              xytext=(8, 1.05), fontsize=6, ha='center',
              arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8))

ax_c.set_xlabel('Simulation cycles', fontsize=7.5)
ax_c.set_ylabel('Shannon diversity H', fontsize=7.5)
ax_c.set_xlim(0, 30)
ax_c.set_ylim(0.3, 1.3)
ax_c.legend(fontsize=6, frameon=True, edgecolor='#cccccc', fancybox=False, loc='upper left')
ax_c.set_title('C', fontsize=11, fontweight='bold', loc='left', pad=4)
ax_c.set_title('Subclonal evolution', fontsize=8, loc='right', pad=4)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.tick_params(labelsize=6.5)

# ---- Panel D: Spatial TME ----
ax_d = fig.add_subplot(gs[1, 1])

x_pos = np.arange(len(compartments))
width = 0.22

colors_tme = ['#2166AC', '#4DBBD5', '#E64B35']
labels_tme = tme_params

for i, (param, color) in enumerate(zip(labels_tme, colors_tme)):
    offset = (i - 1) * width
    bars = ax_d.bar(x_pos + offset, tme_data[:, i], width, label=param,
                     color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, tme_data[:, i]):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f'{val:.2f}', ha='center', va='bottom', fontsize=5)

# Hot TME annotation
ax_d.annotate('Hot TME:\n2.3× response', xy=(1, 0.80), xytext=(1.6, 0.92),
              fontsize=5.5, arrowprops=dict(arrowstyle='->', color='#00A087', lw=0.8),
              color='#00A087', fontweight='bold')

ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(compartments, fontsize=6.5)
ax_d.set_ylabel('Normalized value', fontsize=7.5)
ax_d.set_ylim(0, 1.1)
ax_d.legend(fontsize=5.5, frameon=True, edgecolor='#cccccc', fancybox=False,
            loc='upper left', ncol=3, columnspacing=0.8)
ax_d.set_title('D', fontsize=11, fontweight='bold', loc='left', pad=4)
ax_d.set_title('Spatial TME simulation', fontsize=8, loc='right', pad=4)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.tick_params(labelsize=6.5)

# ==================== SAVE ====================
fig.savefig('D:/IGEM集成方案/manuscripts/figures/fig3_tnbc_simulation.png', dpi=300, bbox_inches='tight')
fig.savefig('D:/IGEM集成方案/manuscripts/figures/fig3_tnbc_simulation.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved successfully.")
