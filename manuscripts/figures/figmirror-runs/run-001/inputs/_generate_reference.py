"""
Generate a synthetic AlphaFold2-style reference image for TorusFold Architecture (Figure 6).

Style: Nature architecture diagrams (Jumper et al. 2021) - mathematical diagrams with
3D structure visualization. Clean, serif typography, muted color palette, tight layout.

Panels:
  A) S1 torus topology - circRNA as closed loop on a torus
  B) TPE formula and periodicity verification
  C) Circular distance metric visualization
  D) Rotation equivariance demonstration
  E) CircPairformer architecture block diagram
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle, Wedge
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

# --- Style: Nature/AlphaFold2 architecture diagram ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
})

# Color palette - muted Nature-style
C_TORUS = '#4A7FB5'
C_TORUS_LIGHT = '#B8D4E8'
C_PAIR = '#D4765A'
C_PAIR_LIGHT = '#F0C8B0'
C_EQUIV = '#6B8E6B'
C_EQUIV_LIGHT = '#B8D8B8'
C_DIST = '#8B6BAE'
C_DIST_LIGHT = '#D0B8E8'
C_TEXT = '#1A1A1A'
C_ACCENT = '#C44E52'
C_GRID = '#E0E0E0'

fig = plt.figure(figsize=(7.5, 9.0))

# ============================================================
# Panel A: S1 Torus Topology (top-left, large)
# ============================================================
ax_a = fig.add_axes([0.06, 0.62, 0.40, 0.33])
ax_a.set_xlim(-1.6, 1.6)
ax_a.set_ylim(-1.4, 1.4)
ax_a.set_aspect('equal')
ax_a.axis('off')

# Draw torus wireframe
theta = np.linspace(0, 2*np.pi, 200)
R, r = 1.0, 0.35

# Torus surface - draw meridian circles
for t in np.linspace(0, 2*np.pi, 24, endpoint=False):
    x_line = (R + r*np.cos(theta)) * np.cos(t)
    y_line = (R + r*np.cos(theta)) * np.sin(t)
    z_line = r * np.sin(theta)
    proj_y = y_line * 0.35 + z_line * 0.65
    alpha_val = 0.15 + 0.1 * np.cos(t)
    ax_a.plot(x_line, proj_y, color=C_TORUS, alpha=max(0.05, alpha_val), linewidth=0.3)

# Draw latitude circles
for phi in np.linspace(0, 2*np.pi, 12, endpoint=False):
    x_line = (R + r*np.cos(phi)) * np.cos(theta)
    y_line = (R + r*np.cos(phi)) * np.sin(theta)
    z_line = r * np.sin(phi) * np.ones_like(theta)
    proj_y = y_line * 0.35 + z_line * 0.65
    ax_a.plot(x_line, proj_y, color=C_TORUS, alpha=0.2, linewidth=0.3)

# Highlight the S1 loop on the torus surface
n_pos = 100
positions = np.linspace(0, 2*np.pi, n_pos, endpoint=False)
x_loop = (R + r*np.cos(positions)) * np.cos(positions)
y_loop = (R + r*np.cos(positions)) * np.sin(positions)
z_loop = r * np.sin(positions)
proj_y_loop = y_loop * 0.35 + z_loop * 0.65

for i in range(len(positions)-1):
    color_val = plt.cm.coolwarm(i / len(positions))
    ax_a.plot([x_loop[i], x_loop[i+1]], [proj_y_loop[i], proj_y_loop[i+1]],
              color=color_val, linewidth=2.5, solid_capstyle='round')

ax_a.plot([x_loop[-1], x_loop[0]], [proj_y_loop[-1], proj_y_loop[0]],
          color=plt.cm.coolwarm(0.99), linewidth=2.5, solid_capstyle='round')

# Mark BSJ
bsj_idx = 0
ax_a.plot(x_loop[bsj_idx], proj_y_loop[bsj_idx], 'o', color=C_ACCENT,
          markersize=8, zorder=10, markeredgecolor='white', markeredgewidth=1.5)
ax_a.annotate('BSJ', (x_loop[bsj_idx], proj_y_loop[bsj_idx]),
              xytext=(15, 10), textcoords='offset points',
              fontsize=7, fontweight='bold', color=C_ACCENT,
              arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=0.8))

ax_a.annotate(r'$\mathbf{x}_i \equiv \mathbf{x}_{i+L}$',
              (0.0, -1.15), fontsize=8, ha='center', color=C_TORUS,
              fontstyle='italic')

ax_a.text(-0.02, 1.08, 'a', transform=ax_a.transAxes, fontsize=11,
          fontweight='bold', va='top', ha='right')
ax_a.text(0.5, 1.05, r'$S^1$ Torus Topology', transform=ax_a.transAxes,
          fontsize=9, ha='center', va='bottom', fontweight='semibold')

# ============================================================
# Panel B: TPE Formula (top-right)
# ============================================================
ax_b = fig.add_axes([0.54, 0.72, 0.42, 0.23])
ax_b.set_xlim(0, 10)
ax_b.set_ylim(0, 6)
ax_b.axis('off')

ax_b.text(5, 5.2, r'Torus Positional Encoding', fontsize=9, ha='center',
          fontweight='semibold', color=C_TEXT)
ax_b.text(5, 4.2,
          r'$\mathrm{TPE}(i) = \sum_{h=0}^{H-1} \left[\sin\!\left(\frac{2\pi h \cdot i}{L}\right),\; \cos\!\left(\frac{2\pi h \cdot i}{L}\right)\right]$',
          fontsize=8, ha='center', color=C_TORUS,
          bbox=dict(boxstyle='round,pad=0.4', facecolor=C_TORUS_LIGHT, edgecolor=C_TORUS, alpha=0.3, linewidth=0.5))

ax_b.text(5, 3.0, r'Periodicity: $|\mathrm{TPE}(i) - \mathrm{TPE}(i+L)| < 10^{-6}$',
          fontsize=7.5, ha='center', color=C_TEXT)

# Small verification plot
ax_b_inset = fig.add_axes([0.60, 0.725, 0.12, 0.08])
L_val = 100
H_val = 16
i_vals = np.arange(200)
tpe_diff = np.array([abs(np.sin(2*np.pi*3*i/L_val) - np.sin(2*np.pi*3*(i+L_val)/L_val)) for i in i_vals])
ax_b_inset.plot(i_vals, tpe_diff + 1e-8, color=C_TORUS, linewidth=0.8)
ax_b_inset.axvline(x=L_val, color=C_ACCENT, linewidth=0.6, linestyle='--')
ax_b_inset.set_xlim(0, 200)
ax_b_inset.set_ylim(-0.05, 0.15)
ax_b_inset.set_xticks([0, 100, 200])
ax_b_inset.set_xticklabels(['0', 'L', '2L'], fontsize=5)
ax_b_inset.set_yticks([])
ax_b_inset.set_title(r'$|\Delta|$', fontsize=5, pad=1)
ax_b_inset.spines['top'].set_visible(False)
ax_b_inset.spines['right'].set_visible(False)
ax_b_inset.spines['left'].set_visible(False)
ax_b_inset.tick_params(length=2, pad=1)

ax_b.text(5, 1.8, f'L = {L_val},  H = {H_val} harmonics', fontsize=7,
          ha='center', color='#555555')

ax_b.text(-0.02, 1.08, 'b', transform=ax_b.transAxes, fontsize=11,
          fontweight='bold', va='top', ha='right')

# ============================================================
# Panel C: Circular Distance (middle-left)
# ============================================================
ax_c = fig.add_axes([0.06, 0.35, 0.40, 0.22])
ax_c.set_xlim(-1.5, 1.5)
ax_c.set_ylim(-1.5, 1.5)
ax_c.set_aspect('equal')
ax_c.axis('off')

circle_theta = np.linspace(0, 2*np.pi, 200)
ax_c.plot(np.cos(circle_theta), np.sin(circle_theta), color=C_DIST, linewidth=1.5)

n_marks = 20
mark_theta = np.linspace(0, 2*np.pi, n_marks, endpoint=False)
ax_c.scatter(np.cos(mark_theta), np.sin(mark_theta), s=15, color=C_DIST, zorder=5)

pos_0_theta = 0
pos_99_theta = 2*np.pi * 99/100

ax_c.plot(np.cos(pos_0_theta), np.sin(pos_0_theta), 'o', color=C_ACCENT,
          markersize=10, zorder=10, markeredgecolor='white', markeredgewidth=1.5)
ax_c.plot(np.cos(pos_99_theta), np.sin(pos_99_theta), 's', color=C_ACCENT,
          markersize=9, zorder=10, markeredgecolor='white', markeredgewidth=1.5)

arc_theta = np.linspace(pos_99_theta, pos_0_theta + 2*np.pi, 30)
ax_c.plot(np.cos(arc_theta)*0.85, np.sin(arc_theta)*0.85, color=C_ACCENT,
          linewidth=2.5, solid_capstyle='round')

ax_c.annotate('0', (np.cos(pos_0_theta)*1.15, np.sin(pos_0_theta)*1.15),
              fontsize=7, ha='center', color=C_ACCENT, fontweight='bold')
ax_c.annotate('99', (np.cos(pos_99_theta)*1.15, np.sin(pos_99_theta)*1.15),
              fontsize=7, ha='center', color=C_ACCENT, fontweight='bold')

ax_c.text(0, -1.35,
          r'$d_{\mathrm{circ}}(i,j) = \min(|i-j|,\; L - |i-j|)$',
          fontsize=7.5, ha='center', color=C_DIST,
          bbox=dict(boxstyle='round,pad=0.3', facecolor=C_DIST_LIGHT, edgecolor=C_DIST, alpha=0.3, linewidth=0.5))

ax_c.text(0, 0.0,
          r'$d_{\mathrm{circ}}(0,99) = \min(99, 1) = 1$',
          fontsize=7, ha='center', color=C_TEXT)

ax_c.text(-0.02, 1.08, 'c', transform=ax_c.transAxes, fontsize=11,
          fontweight='bold', va='top', ha='right')
ax_c.text(0.5, 1.05, 'Circular Distance', transform=ax_c.transAxes,
          fontsize=9, ha='center', va='bottom', fontweight='semibold')

# ============================================================
# Panel D: Rotation Equivariance (middle-right)
# ============================================================
ax_d = fig.add_axes([0.54, 0.35, 0.42, 0.30])
ax_d.set_xlim(-1.5, 1.5)
ax_d.set_ylim(-0.8, 1.2)
ax_d.set_aspect('equal')
ax_d.axis('off')

ax_d.text(-0.75, 1.1, r'$f(\mathbf{x})$', fontsize=8, ha='center',
          color=C_EQUIV, fontweight='semibold')

n_seg = 8
seg_theta = np.linspace(0, 2*np.pi, n_seg+1)
for i in range(n_seg):
    t = np.linspace(seg_theta[i], seg_theta[i+1], 20)
    color = plt.cm.Set2(i / n_seg)
    ax_d.plot(np.cos(t)*0.5 - 0.75, np.sin(t)*0.5 + 0.3, color=color, linewidth=4)

ax_d.annotate('', xy=(0.15, 0.3), xytext=(-0.15, 0.3),
              arrowprops=dict(arrowstyle='->', color=C_TEXT, lw=1.2))
ax_d.text(0.0, 0.55, r'$R_s$', fontsize=9, ha='center', color=C_TEXT)

ax_d.text(0.75, 1.1, r'$R_s \cdot f(\mathbf{x})$', fontsize=8, ha='center',
          color=C_EQUIV, fontweight='semibold')

shift = 2
for i in range(n_seg):
    t = np.linspace(seg_theta[(i+shift) % n_seg], seg_theta[(i+shift+1) % n_seg], 20)
    color = plt.cm.Set2(i / n_seg)
    ax_d.plot(np.cos(t)*0.5 + 0.75, np.sin(t)*0.5 + 0.3, color=color, linewidth=4)

ax_d.text(0.0, -0.55,
          r'$f(R_s \cdot \mathbf{x}) = R_s \cdot f(\mathbf{x})$',
          fontsize=9, ha='center', color=C_EQUIV,
          bbox=dict(boxstyle='round,pad=0.4', facecolor=C_EQUIV_LIGHT, edgecolor=C_EQUIV, alpha=0.3, linewidth=0.5))

ax_d.text(-0.02, 1.08, 'd', transform=ax_d.transAxes, fontsize=11,
          fontweight='bold', va='top', ha='right')
ax_d.text(0.5, 1.05, 'Rotation Equivariance', transform=ax_d.transAxes,
          fontsize=9, ha='center', va='bottom', fontweight='semibold')

# ============================================================
# Panel E: CircPairformer Architecture (bottom, full width)
# ============================================================
ax_e = fig.add_axes([0.06, 0.04, 0.88, 0.26])
ax_e.set_xlim(0, 20)
ax_e.set_ylim(0, 5)
ax_e.axis('off')

blocks = [
    (1.0, 2.0, 3.0, 2.5, 'Input\nEmbedding', C_TORUS_LIGHT, C_TORUS),
    (4.5, 2.0, 3.0, 2.5, 'Torus\nPos. Enc.', C_TORUS_LIGHT, C_TORUS),
    (8.0, 1.0, 3.5, 3.5, 'CircPair-\nformer\n$\\times N$', C_PAIR_LIGHT, C_PAIR),
    (12.0, 2.0, 3.0, 2.5, 'Circular\nAttention', C_EQUIV_LIGHT, C_EQUIV),
    (15.5, 2.0, 3.0, 2.5, 'Structure\nModule', C_DIST_LIGHT, C_DIST),
]

for (x, y, w, h, label, fc, ec) in blocks:
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                          facecolor=fc, edgecolor=ec, linewidth=1.2, alpha=0.85)
    ax_e.add_patch(rect)
    ax_e.text(x + w/2, y + h/2, label, fontsize=7, ha='center', va='center',
              color=C_TEXT, fontweight='medium', linespacing=1.3)

arrow_pairs = [(4.0, 3.25, 4.5, 3.25), (7.5, 3.25, 8.0, 2.75),
               (11.5, 2.75, 12.0, 3.25), (15.0, 3.25, 15.5, 3.25)]
for (x1, y1, x2, y2) in arrow_pairs:
    ax_e.annotate('', xy=(x2, y2), xytext=(x1, y1),
                  arrowprops=dict(arrowstyle='->', color=C_TEXT, lw=1.0))

ax_e.annotate('', xy=(11.5, 1.2), xytext=(8.0, 1.2),
              arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8,
                             linestyle='dashed', connectionstyle='arc3,rad=0.0'))
ax_e.text(9.75, 0.6, 'residual', fontsize=5.5, ha='center', color='#888888',
          fontstyle='italic')

ax_e.text(19.0, 3.25, r'$\hat{Y}$', fontsize=10, ha='center', va='center',
          color=C_TEXT, fontweight='bold')
ax_e.annotate('', xy=(18.7, 3.25), xytext=(18.5, 3.25),
              arrowprops=dict(arrowstyle='->', color=C_TEXT, lw=1.0))

ax_e.text(0.3, 3.25, r'$\mathbf{X}$', fontsize=10, ha='center', va='center',
          color=C_TEXT, fontweight='bold')

ax_e.text(-0.01, 1.05, 'e', transform=ax_e.transAxes, fontsize=11,
          fontweight='bold', va='top', ha='right')
ax_e.text(0.5, 1.05, 'CircPairformer Architecture', transform=ax_e.transAxes,
          fontsize=9, ha='center', va='bottom', fontweight='semibold')

# ============================================================
# Save
# ============================================================
fig.savefig('D:/IGEM集成方案/manuscripts/figures/figmirror-runs/run-001/inputs/reference_raw.png',
            dpi=300, facecolor='white', edgecolor='none',
            bbox_inches='tight', pad_inches=0.04)
plt.close()
print("reference_raw.png generated successfully")
