#!/usr/bin/env python3
"""
Figure 6: TorusFold Architecture - 3D Visualization
Style: AlphaFold2 Nature 2021 architecture diagrams
Multi-panel: A) Torus S1 topology  B) TPE periodicity  C) Circular distance  D) 3D circRNA fold
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['axes.linewidth'] = 0.6

# ==================== DATA SECTOR ====================

L = 100  # circRNA length
H = 16   # TPE harmonics

# TPE basis functions
positions = np.arange(0, 2*L, 1)
tpe_sin = np.zeros((H, len(positions)))
tpe_cos = np.zeros((H, len(positions)))
for h in range(1, H+1):
    tpe_sin[h-1] = np.sin(2 * np.pi * h * positions / L)
    tpe_cos[h-1] = np.cos(2 * np.pi * h * positions / L)

# Verify periodicity: TPE(i) = TPE(i+L)
# |TPE(i) - TPE(i+L)| < 1e-6

# Circular distance examples
d_circ_0_99 = min(abs(0 - 99), L - abs(0 - 99))  # = 1
d_circ_10_50 = min(abs(10 - 50), L - abs(10 - 50))  # = 40
d_linear_0_99 = abs(0 - 99)  # = 99

# 3D circRNA theoretical structure parameters
theta = np.linspace(0, 2*np.pi, 200)

# ==================== COLOR PALETTE ====================

torus_color = '#3C5488'    # Navy
bsj_color = '#E64B35'      # Nature red
ires_color = '#00A087'     # Teal
dsrna_color = '#4DBBD5'    # Cyan
text_color = '#555555'
accent_color = '#7E6148'   # Brown
pathway_color = '#00A087'  # Teal (same as ires_color, used in Panel C)

# ==================== FIGURE COMPOSITION ====================

fig = plt.figure(figsize=(7.2, 7.5), dpi=300)
gs = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.35,
              left=0.06, right=0.96, top=0.94, bottom=0.06)

# ---- Panel A: Torus S1 Topology (3D) ----
ax_a = fig.add_subplot(gs[0, 0], projection='3d')

# Draw circRNA as a torus (S1 topology)
R_major = 3.0  # Major radius
r_minor = 0.6  # Minor radius

u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, 2*np.pi, 60)
U, V = np.meshgrid(u, v)

X = (R_major + r_minor * np.cos(V)) * np.cos(U)
Y = (R_major + r_minor * np.cos(V)) * np.sin(U)
Z = r_minor * np.sin(V)

# Color the torus with position encoding
# Position i maps to angle U, position i+L maps to same angle
position_norm = U / (2*np.pi)  # 0 to 1
colors = plt.cm.viridis(position_norm)

ax_a.plot_surface(X, Y, Z, facecolors=colors, alpha=0.35, linewidth=0, antialiased=True)

# Mark position i=0 (BSJ)
i_theta = 0
x_bsj = R_major * np.cos(i_theta)
y_bsj = R_major * np.sin(i_theta)
z_bsj = 0
ax_a.scatter([x_bsj], [y_bsj], [z_bsj], s=80, color=bsj_color, zorder=10, edgecolors='white', linewidths=1)

# Mark position i+L (same as i=0 for circRNA)
# Arrow showing periodicity
iL_theta = 2*np.pi
x_il = R_major * np.cos(iL_theta - 0.3)
y_il = R_major * np.sin(iL_theta - 0.3)
z_il = 0
ax_a.scatter([x_il], [y_il], [z_il], s=80, color=bsj_color, zorder=10, marker='D', edgecolors='white', linewidths=1)

# Draw connecting arc showing periodicity
arc_theta = np.linspace(-0.3, 0, 20)
arc_x = R_major * np.cos(arc_theta)
arc_y = R_major * np.sin(arc_theta)
arc_z = np.zeros_like(arc_theta) + r_minor * 1.5
ax_a.plot(arc_x, arc_y, arc_z, color=bsj_color, linewidth=1.5, linestyle='--')

# Label
ax_a.text2D(0.05, 0.95, 'Position i\n= Position i+L', fontsize=6.5,
            transform=ax_a.transAxes, va='top', color=bsj_color, fontweight='bold')

ax_a.set_xlim(-4, 4)
ax_a.set_ylim(-4, 4)
ax_a.set_zlim(-2, 2)
ax_a.view_init(elev=25, azim=45)
ax_a.set_axis_off()
ax_a.set_title('A', fontsize=11, fontweight='bold', loc='left', pad=-10)
ax_a.text2D(0.5, 1.02, 'S$^1$ torus topology', fontsize=8,
            transform=ax_a.transAxes, ha='center', va='bottom')

# ---- Panel B: TPE Periodicity Verification ----
ax_b = fig.add_subplot(gs[0, 1])

# Plot first few TPE harmonics
x_pos = np.arange(0, L+1)
for h in range(1, 5):
    y_h = np.sin(2 * np.pi * h * x_pos / L)
    alpha = 0.8 if h <= 2 else 0.4
    lw = 1.5 if h <= 2 else 0.8
    ax_b.plot(x_pos, y_h, linewidth=lw, alpha=alpha,
              label=f'h={h}')

# Mark periodicity at L
ax_b.axvline(x=L, color=bsj_color, linestyle=':', linewidth=1, alpha=0.7)
ax_b.annotate('TPE(0) = TPE(L)\n|Δ| < 10$^{-6}$', xy=(L, 0), xytext=(L-20, 0.8),
              fontsize=6, arrowprops=dict(arrowstyle='->', color=bsj_color, lw=0.8),
              color=bsj_color, fontweight='bold')

# Formula inset
formula_text = ('TPE(i,2k) = $\\sum_{h=1}^{H}$ $w_{h,k}$ sin(2$\\pi$hi/L)\n'
                'TPE(i,2k+1) = $\\sum_{h=1}^{H}$ $w_{h,k}$ cos(2$\\pi$hi/L)')
ax_b.text(0.02, 0.98, formula_text, fontsize=5.5,
          transform=ax_b.transAxes, va='top', ha='left',
          bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.9, boxstyle='round,pad=0.4'))

ax_b.set_xlabel('Position i', fontsize=7.5)
ax_b.set_ylabel('TPE value', fontsize=7.5)
ax_b.set_xlim(0, L+2)
ax_b.set_ylim(-1.3, 1.3)
ax_b.legend(fontsize=5.5, frameon=True, edgecolor='#cccccc', fancybox=False,
            loc='lower right', ncol=2, columnspacing=0.8)
ax_b.set_title('B', fontsize=11, fontweight='bold', loc='left', pad=4)
ax_b.set_title('Torus Positional Encoding', fontsize=8, loc='right', pad=4)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.tick_params(labelsize=6.5)

# ---- Panel C: Circular vs Linear Distance ----
ax_c = fig.add_subplot(gs[1, 0])

# Linear distance
i_range = np.arange(0, L)
j = 0  # reference position
d_linear = np.abs(i_range - j)
d_circular = np.minimum(np.abs(i_range - j), L - np.abs(i_range - j))

ax_c.plot(i_range, d_linear, color=text_color, linewidth=1.5, linestyle='--', label='Linear d(i,j)')
ax_c.plot(i_range, d_circular, color=pathway_color, linewidth=1.5, label='Circular $d_{circ}$(i,j)')

# Highlight key examples
ax_c.scatter([99], [d_circ_0_99], s=60, color=bsj_color, zorder=5, edgecolors='white')
ax_c.annotate(f'$d_{{circ}}$(0,99) = 1\n(neighbors!)', xy=(99, 1), xytext=(75, 30),
              fontsize=6, arrowprops=dict(arrowstyle='->', color=bsj_color, lw=0.8),
              color=bsj_color, fontweight='bold')

ax_c.scatter([99], [99], s=40, color=text_color, zorder=5, marker='x')
ax_c.annotate(f'd(0,99) = 99\n(distant)', xy=(99, 99), xytext=(70, 85),
              fontsize=6, arrowprops=dict(arrowstyle='->', color=text_color, lw=0.8),
              color=text_color)

# Formula
ax_c.text(0.02, 0.98, '$d_{circ}$(i,j) = min(|i-j|, L-|i-j|)', fontsize=6.5,
          transform=ax_c.transAxes, va='top', ha='left',
          bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.9, boxstyle='round,pad=0.4'))

ax_c.set_xlabel('Position i (j=0)', fontsize=7.5)
ax_c.set_ylabel('Distance', fontsize=7.5)
ax_c.set_xlim(0, L)
ax_c.set_ylim(0, 105)
ax_c.legend(fontsize=6, frameon=True, edgecolor='#cccccc', fancybox=False, loc='upper left')
ax_c.set_title('C', fontsize=11, fontweight='bold', loc='left', pad=4)
ax_c.set_title('Circular distance metric', fontsize=8, loc='right', pad=4)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.tick_params(labelsize=6.5)

# ---- Panel D: 3D circRNA Theoretical Fold ----
ax_d = fig.add_subplot(gs[1, 1], projection='3d')

# Generate theoretical circRNA 3D structure
# Main circle backbone
t = np.linspace(0, 2*np.pi, 300)
R_backbone = 5.0
x_circle = R_backbone * np.cos(t)
y_circle = R_backbone * np.sin(t)
z_circle = np.zeros_like(t)

# Add stem-loop structures
# Stem 1: positions 10-30 (dsRNA region for MDA5)
stem1_mask = (t > 0.2*np.pi) & (t < 0.6*np.pi)
stem1_height = 3.0 * np.sin(np.pi * (t[stem1_mask] - 0.2*np.pi) / (0.4*np.pi))
x_stem1 = x_circle[stem1_mask]
y_stem1 = y_circle[stem1_mask]
z_stem1 = stem1_height

# Stem 2: positions 60-85 (dsRNA region for PKR)
stem2_mask = (t > 1.2*np.pi) & (t < 1.7*np.pi)
stem2_height = 4.0 * np.sin(np.pi * (t[stem2_mask] - 1.2*np.pi) / (0.5*np.pi))
x_stem2 = x_circle[stem2_mask]
y_stem2 = y_circle[stem2_mask]
z_stem2 = stem2_height

# IRES region (small loop)
ires_mask = (t > 0.5*np.pi) & (t < 0.7*np.pi)
ires_height = 1.5 * np.sin(np.pi * (t[ires_mask] - 0.5*np.pi) / (0.2*np.pi))
x_ires = x_circle[ires_mask]
y_ires = y_circle[ires_mask]
z_ires = ires_height

# Plot backbone
ax_d.plot(x_circle, y_circle, z_circle, color='#888888', linewidth=1.2, alpha=0.6)

# Plot stem-loops
ax_d.plot(x_stem1, y_stem1, z_stem1, color=dsrna_color, linewidth=2.5, alpha=0.85)
ax_d.plot(x_stem2, y_stem2, z_stem2, color='#3C5488', linewidth=2.5, alpha=0.85)
ax_d.plot(x_ires, y_ires, z_ires, color=ires_color, linewidth=2.5, alpha=0.85)

# Mark BSJ
bsj_idx = 0
ax_d.scatter([x_circle[bsj_idx]], [y_circle[bsj_idx]], [z_circle[bsj_idx]],
             s=100, color=bsj_color, zorder=10, edgecolors='white', linewidths=1.5)

# Mark m6A sites
m6a_positions = [0.15*np.pi, 0.35*np.pi, 1.3*np.pi, 1.5*np.pi]
for m6a_t in m6a_positions:
    idx = int(m6a_t / (2*np.pi) * 300) % 300
    ax_d.scatter([x_circle[idx]], [y_circle[idx]], [0],
                 s=40, color='#F39B7F', marker='o', zorder=10, edgecolors='white', linewidths=0.5)

# Legend
ax_d.text2D(0.02, 0.92, 'BSJ junction', fontsize=5.5, color=bsj_color,
            transform=ax_d.transAxes, fontweight='bold')
ax_d.text2D(0.02, 0.86, 'MDA5 dsRNA (>16bp)', fontsize=5.5, color=dsrna_color,
            transform=ax_d.transAxes)
ax_d.text2D(0.02, 0.80, 'PKR dsRNA (>33bp)', fontsize=5.5, color='#3C5488',
            transform=ax_d.transAxes)
ax_d.text2D(0.02, 0.74, 'IRES region', fontsize=5.5, color=ires_color,
            transform=ax_d.transAxes)
ax_d.text2D(0.02, 0.68, 'm6A sites', fontsize=5.5, color='#F39B7F',
            transform=ax_d.transAxes)

# Warning annotation
ax_d.text2D(0.98, 0.02, 'Theoretical structure\nNo circRNA 3D data\nexists in PDB', fontsize=5,
            transform=ax_d.transAxes, va='bottom', ha='right', color='#999999',
            style='italic',
            bbox=dict(facecolor='#fff8f0', edgecolor='#E64B35', alpha=0.8, boxstyle='round,pad=0.3', linewidth=0.8))

ax_d.set_xlim(-8, 8)
ax_d.set_ylim(-8, 8)
ax_d.set_zlim(-6, 6)
ax_d.view_init(elev=20, azim=30)
ax_d.set_axis_off()
ax_d.set_title('D', fontsize=11, fontweight='bold', loc='left', pad=-10)
ax_d.text2D(0.5, 1.02, '3D circRNA fold (theoretical)', fontsize=8,
            transform=ax_d.transAxes, ha='center', va='bottom')

# ==================== SAVE ====================
fig.savefig('D:/IGEM集成方案/manuscripts/figures/fig6_torusfold_architecture.png', dpi=300, bbox_inches='tight')
fig.savefig('D:/IGEM集成方案/manuscripts/figures/fig6_torusfold_architecture.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 6 saved successfully.")