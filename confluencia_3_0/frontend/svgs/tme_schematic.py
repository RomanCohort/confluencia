"""TME SVG Schematic Generator

Generates an inline SVG visualization of the tumor microenvironment,
showing tumor mass, immune cells, vasculature, CAF/ECM, and immune pressure.
"""
from typing import Dict, Any

# Tokyo Night palette (matches app_core)
TN = {
    "bg": "#1a1b26", "surface": "#24283b", "border": "#414d68",
    "text": "#c0caf5", "muted": "#a9b1d6",
    "blue": "#7aa2f7", "green": "#9ece6a", "yellow": "#e0af68",
    "red": "#f7768e", "purple": "#bb9af7", "cyan": "#7dcfff",
    "orange": "#ff9e64", "teal": "#73daca",
}


def render_tme_svg(state: Dict[str, Any]) -> str:
    """Generate inline SVG of tumor microenvironment.

    Layout:
      - Center: tumor mass circle (radius ~ volume)
      - Surrounding: vasculature arcs (density ~ microvessel_density)
      - Inner dots: CD8+ (green), NK (cyan), Tregs (red), MDSCs (orange)
      - Outer ring: CAF/ECM (opacity ~ ecm_density)
      - PD-L1 shield arc (opacity ~ pd_l1_expression)
      - Pressure arrows: immune (blue inward) vs evasion (red outward)
      - Phase label at top
    """
    # Extract state values with defaults
    volume = state.get("tum_volume", 50.0)
    cd8 = state.get("imm_cd8_count", 100)
    nk = state.get("imm_nk_count", 50)
    treg = state.get("imm_treg_count", 20)
    mdsc = state.get("imm_mdsc_count", 30)
    mvd = state.get("vasc_microvessel_density", 0.5)
    ecm = state.get("caf_ecm_density", 0.3)
    pd_l1 = state.get("evs_pd_l1_expression", 0.15)
    phase = state.get("ied_phase", "elimination")
    immune_pressure = state.get("ied_immune_pressure", 0.5)
    evasion_pressure = state.get("ied_evasion_pressure", 0.3)
    m1_frac = state.get("imm_m1_fraction", 0.5)
    csc_frac = state.get("csc_fraction", 0.02)

    # SVG dimensions
    W, H = 400, 400
    cx, cy = W // 2, H // 2

    # Tumor radius (log scale, 20-80px)
    import math
    tumor_r = max(20, min(80, 20 + 15 * math.log10(max(1, volume))))

    # Build SVG elements
    elements = []

    # Background
    elements.append(f'<rect width="{W}" height="{H}" fill="{TN["bg"]}" rx="12"/>')

    # CAF/ECM outer ring
    ecm_opacity = min(0.8, ecm * 1.5)
    ecm_r = tumor_r + 40
    elements.append(f'<circle cx="{cx}" cy="{cy}" r="{ecm_r}" '
                    f'fill="none" stroke="{TN["yellow"]}" stroke-width="8" '
                    f'opacity="{ecm_opacity:.2f}" stroke-dasharray="4,6"/>')
    if ecm > 0.3:
        elements.append(f'<text x="{cx}" y="{cy - ecm_r - 6}" text-anchor="middle" '
                        f'fill="{TN["yellow"]}" font-size="9" opacity="0.8">ECM</text>')

    # Vasculature arcs
    n_vessels = max(3, int(mvd * 12))
    vasc_r = tumor_r + 25
    for i in range(n_vessels):
        angle = i * 360 / n_vessels
        import math as _m
        rad = _m.radians(angle)
        x1 = cx + vasc_r * _m.cos(rad)
        y1 = cy + vasc_r * _m.sin(rad)
        x2 = cx + (tumor_r + 5) * _m.cos(rad)
        y2 = cy + (tumor_r + 5) * _m.sin(rad)
        elements.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{TN["red"]}" stroke-width="2" opacity="0.6"/>')

    # Tumor mass
    # Color based on necrosis
    necrosis = state.get("tum_necrosis_fraction", 0.0)
    tumor_color = TN["purple"] if necrosis < 0.3 else TN["orange"]
    elements.append(f'<circle cx="{cx}" cy="{cy}" r="{tumor_r:.1f}" '
                    f'fill="{tumor_color}" opacity="0.7"/>')
    elements.append(f'<circle cx="{cx}" cy="{cy}" r="{tumor_r:.1f}" '
                    f'fill="none" stroke="{TN["text"]}" stroke-width="1.5" opacity="0.5"/>')

    # CSC core
    csc_r = max(3, tumor_r * csc_frac * 5)
    elements.append(f'<circle cx="{cx}" cy="{cy}" r="{csc_r:.1f}" '
                    f'fill="{TN["yellow"]}" opacity="0.9"/>')

    # PD-L1 shield arc
    if pd_l1 > 0.05:
        shield_r = tumor_r + 8
        shield_extent = min(180, pd_l1 * 300)
        # Draw as a thick arc at top
        import math as _m2
        start_angle = _m2.radians(-shield_extent / 2 + 90)
        end_angle = _m2.radians(shield_extent / 2 + 90)
        x1 = cx + shield_r * _m2.cos(start_angle)
        y1 = cy - shield_r * _m2.sin(start_angle)
        x2 = cx + shield_r * _m2.cos(end_angle)
        y2 = cy - shield_r * _m2.sin(end_angle)
        pd_l1_opacity = min(0.9, pd_l1 * 3)
        elements.append(f'<path d="M {x1:.1f} {y1:.1f} A {shield_r} {shield_r} 0 0 1 {x2:.1f} {y2:.1f}" '
                        f'fill="none" stroke="{TN["red"]}" stroke-width="4" '
                        f'opacity="{pd_l1_opacity:.2f}"/>')
        elements.append(f'<text x="{cx}" y="{cy - shield_r - 8}" text-anchor="middle" '
                        f'fill="{TN["red"]}" font-size="8" opacity="0.8">PD-L1</text>')

    # Immune cells as dots around tumor
    import random
    rng = random.Random(42)

    # CD8+ T cells (green)
    n_cd8 = max(2, min(15, int(cd8 / 50)))
    for _ in range(n_cd8):
        angle = rng.uniform(0, 2 * math.pi)
        dist = tumor_r + rng.uniform(10, 35)
        x = cx + dist * math.cos(angle)
        y = cy + dist * math.sin(angle)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{TN["green"]}" opacity="0.8"/>')

    # NK cells (cyan)
    n_nk = max(1, min(8, int(nk / 30)))
    for _ in range(n_nk):
        angle = rng.uniform(0, 2 * math.pi)
        dist = tumor_r + rng.uniform(12, 40)
        x = cx + dist * math.cos(angle)
        y = cy + dist * math.sin(angle)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{TN["cyan"]}" opacity="0.8"/>')

    # Tregs (red)
    n_treg = max(0, min(6, int(treg / 10)))
    for _ in range(n_treg):
        angle = rng.uniform(0, 2 * math.pi)
        dist = tumor_r + rng.uniform(8, 30)
        x = cx + dist * math.cos(angle)
        y = cy + dist * math.sin(angle)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{TN["red"]}" opacity="0.7"/>')

    # MDSCs (orange)
    n_mdsc = max(0, min(6, int(mdsc / 15)))
    for _ in range(n_mdsc):
        angle = rng.uniform(0, 2 * math.pi)
        dist = tumor_r + rng.uniform(10, 32)
        x = cx + dist * math.cos(angle)
        y = cy + dist * math.sin(angle)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{TN["orange"]}" opacity="0.7"/>')

    # Immune pressure arrow (blue, inward)
    arrow_len = min(40, immune_pressure * 60)
    if arrow_len > 5:
        ax = cx + tumor_r + 55
        ay = cy
        elements.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{ax - arrow_len:.1f}" y2="{ay:.1f}" '
                        f'stroke="{TN["blue"]}" stroke-width="3" marker-end="url(#arrowBlue)"/>')

    # Evasion pressure arrow (red, outward)
    evasion_len = min(40, evasion_pressure * 60)
    if evasion_len > 5:
        ax = cx - tumor_r - 55
        ay = cy
        elements.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{ax + evasion_len:.1f}" y2="{ay:.1f}" '
                        f'stroke="{TN["red"]}" stroke-width="3" marker-end="url(#arrowRed)"/>')

    # Arrow markers
    elements.insert(2, f'''
    <defs>
        <marker id="arrowBlue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="{TN["blue"]}"/>
        </marker>
        <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="{TN["red"]}"/>
        </marker>
    </defs>
    ''')

    # Phase label
    phase_colors = {"elimination": TN["blue"], "equilibrium": TN["yellow"], "escape": TN["red"]}
    phase_color = phase_colors.get(phase, TN["muted"])
    elements.append(f'<text x="{cx}" y="20" text-anchor="middle" '
                    f'fill="{phase_color}" font-size="12" font-weight="bold">{phase.upper()}</text>')

    # Legend
    legend_y = H - 50
    legend_items = [
        (TN["green"], "CD8+"), (TN["cyan"], "NK"), (TN["red"], "Treg"),
        (TN["orange"], "MDSC"), (TN["yellow"], "CSC"), (TN["purple"], "Tumor"),
    ]
    for i, (color, label) in enumerate(legend_items):
        lx = 15 + i * 62
        elements.append(f'<circle cx="{lx}" cy="{legend_y}" r="4" fill="{color}"/>')
        elements.append(f'<text x="{lx + 8}" y="{legend_y + 4}" fill="{TN["muted"]}" font-size="8">{label}</text>')

    # Volume label
    elements.append(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
                    f'fill="{TN["text"]}" font-size="10" font-weight="bold">{volume:.1f} mm3</text>')

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%">' + ''.join(elements) + '</svg>'
    return svg
