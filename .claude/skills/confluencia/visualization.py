"""Enhanced HTML Visualization for Confluencia - Nature Journal Style.

Publication-quality HTML report generation with Nature journal styling.
"""

import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


def generate_nature_html_report(data: Dict[str, Any], title: str = "Confluencia Analysis") -> str:
    """Generate Nature journal-style HTML report."""
    module = data.get("module", "unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_json = json.dumps(data, indent=2)

    if module == "circrna":
        body = _circrna_body(data)
    elif module == "drug":
        body = _drug_body(data)
    elif module == "epitope":
        body = _epitope_body(data)
    elif module == "simulacrum" or module == "tnbc":
        body = _simulacrum_body(data)
    else:
        body = "<div class='card'><div class='card-body'>No visualization</div></div>"

    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Source+Code+Pro:wght@400;600&family=Source+Sans+Pro:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Source Sans Pro', sans-serif; line-height: 1.6; color: #ecf0f1; background: linear-gradient(135deg, #1a1a2e 0%, #2d3a4f 100%); min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .nature-header {{ background: linear-gradient(135deg, #1a1a2e 0%, #2c3e50 100%); border-left: 5px solid #c41e3a; padding: 30px 40px; margin-bottom: 30px; border-radius: 0 8px 8px 0; }}
        .nature-header h1 {{ font-family: 'Merriweather', Georgia, serif; font-size: 2.2em; font-weight: 700; color: #ecf0f1; margin-bottom: 8px; }}
        .nature-header .subtitle {{ font-size: 1.1em; color: #7f8c8d; font-style: italic; }}
        .nature-header .meta {{ margin-top: 15px; display: flex; flex-wrap: wrap; gap: 25px; font-size: 0.9em; }}
        .nature-header .meta-item {{ display: flex; align-items: center; gap: 8px; }}
        .nature-header .meta-label {{ color: #7f8c8d; }}
        .nature-header .meta-value {{ color: #ecf0f1; font-weight: 600; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }}
        .card-header {{ background: rgba(196, 30, 58, 0.1); border-bottom: 1px solid #30363d; padding: 15px 20px; font-weight: 600; color: #c41e3a; font-size: 1.1em; }}
        .card-body {{ padding: 20px; }}
        .row {{ display: flex; flex-wrap: wrap; margin: 0 -10px; }}
        .col-12 {{ width: 100%; padding: 0 10px; }}
        .col-6 {{ width: 50%; padding: 0 10px; }}
        .col-4 {{ width: 33.333%; padding: 0 10px; }}
        @media (max-width: 768px) {{ .col-6, .col-4 {{ width: 100%; }} }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-box {{ background: rgba(255, 255, 255, 0.03); border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }}
        .stat-box.success {{ border-left: 4px solid #27ae60; }}
        .stat-box.warning {{ border-left: 4px solid #f39c12; }}
        .stat-box.danger {{ border-left: 4px solid #e74c3c; }}
        .stat-value {{ font-size: 2.5em; font-weight: 700; font-family: 'Source Code Pro', monospace; }}
        .stat-value.success {{ color: #27ae60; }}
        .stat-value.warning {{ color: #f39c12; }}
        .stat-value.danger {{ color: #e74c3c; }}
        .stat-label {{ font-size: 0.85em; color: #7f8c8d; margin-top: 5px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .chart-container {{ height: 350px; position: relative; }}
        .chart-small {{ height: 280px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ background: rgba(255, 255, 255, 0.05); font-weight: 600; color: #7f8c8d; text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.5px; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
        .badge-success {{ background: rgba(39, 174, 96, 0.2); color: #27ae60; }}
        .badge-warning {{ background: rgba(243, 156, 18, 0.2); color: #f39c12; }}
        .badge-danger {{ background: rgba(231, 76, 60, 0.2); color: #e74c3c; }}
        .badge-info {{ background: rgba(41, 128, 185, 0.2); color: #2980b9; }}
        .sequence-display {{ font-family: 'Source Code Pro', monospace; font-size: 1.1em; background: rgba(0, 0, 0, 0.3); padding: 15px 20px; border-radius: 6px; letter-spacing: 2px; overflow-x: auto; }}
        .base-A {{ color: #3498db; }}
        .base-U {{ color: #e74c3c; }}
        .base-G {{ color: #27ae60; }}
        .base-C {{ color: #f39c12; }}
        .footer {{ text-align: center; padding: 30px; color: #7f8c8d; font-size: 0.85em; border-top: 1px solid #30363d; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        {}
        <div class="footer">
            <p>Generated by <strong>Confluencia</strong> | {}</p>
            <p><em>Nature journal style visualization</em></p>
        </div>
    </div>
    <script>
        function exportJSON() {{
            const data = {};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'confluencia_data.json';
            a.click();
        }}
    </script>
</body>
</html>""".format(title, body, timestamp, data_json)


def _format_sequence(sequence: str) -> str:
    """Format sequence with colored nucleotides."""
    result = []
    for base in sequence.upper():
        if base in 'AUGC':
            result.append('<span class="base-{}">{}</span>'.format(base, base))
        else:
            result.append(base)
    return ''.join(result)


def _circrna_body(data: Dict[str, Any]) -> str:
    """Generate circRNA visualization body."""
    import numpy as np

    immune = data.get("immune", {})
    tf = data.get("torusfold", {})
    pk_params = data.get("pk_params", {})
    seq = data.get("sequence", "")
    gc = data.get("gc_content", 0)
    length = data.get("length", 0)
    backend = data.get("backend", "heuristic")

    # Estimate dot-bracket for structure visualization
    dot_bracket = _estimate_dot_bracket(seq)

    # Nucleotide counts
    a_count = sum(1 for b in seq.upper() if b == 'A')
    u_count = sum(1 for b in seq.upper() if b in 'U')
    g_count = sum(1 for b in seq.upper() if b == 'G')
    c_count = sum(1 for b in seq.upper() if b == 'C')

    safety = immune.get("safety_score", 1.0)
    safety_class = 'success' if safety > 0.8 else 'warning' if safety > 0.5 else 'danger'
    safety_label = 'SAFE' if safety > 0.8 else 'MODERATE' if safety > 0.5 else 'RISK'

    gc_pct = gc * 100
    gc_class = 'success' if 40 <= gc_pct <= 60 else 'warning'

    # PK simulation
    pk_html = ""
    if pk_params:
        ka = pk_params.get("k_uptake", 0.8)
        ke = pk_params.get("k_degrade", 0.1)
        hl = pk_params.get("protein_half_life", 16.0)
        f_liver = pk_params.get("f_liver", 0.8)

        t = np.linspace(0, 72, 144).tolist()
        vd = 50.0
        c = [(ka / (vd * (ka - ke))) * (np.exp(-ke * ti) - np.exp(-ka * ti)) for ti in t]
        auc = float(np.trapz(c, t))
        cmax = max(c)

        pk_html = """
        <div class="card">
            <div class="card-header">RNACTM Pharmacokinetics</div>
            <div class="card-body">
                <div class="row">
                    <div class="col-6"><div id="pkCurve" class="chart-container"></div></div>
                    <div class="col-6">
                        <div class="stat-grid">
                            <div class="stat-box success"><div class="stat-value success">{:.1f}</div><div class="stat-label">Half-life (h)</div></div>
                            <div class="stat-box"><div class="stat-value">{:.2f}</div><div class="stat-label">AUC</div></div>
                            <div class="stat-box"><div class="stat-value">{:.4f}</div><div class="stat-label">Cmax</div></div>
                            <div class="stat-box"><div class="stat-value">{:.0f}%</div><div class="stat-label">Liver</div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <script>
        Plotly.newPlot('pkCurve', [{{type:'scatter', x:{}, y:{}, mode:'lines', fill:'tozeroy', line:{{color:'#27ae60',width:3}}}}],
            {{xaxis:{{title:'Time (h)',gridcolor:'#30363d'}}, yaxis:{{title:'Conc',gridcolor:'#30363d'}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});
        </script>""".format(hl, auc, cmax, f_liver*100, json.dumps(t), json.dumps(c))

    # Circular RNA ring structure SVG
    circ_svg = generate_circular_rna_svg(seq, dot_bracket)

    # 3D structure visualization
    struct_3d = generate_structure_3d_plotly(seq, dot_bracket)

    # Base pairing matrix
    pairing_matrix = generate_pairing_matrix_html(seq, dot_bracket)

    # Build structure visualization section
    structure_section = """
    <div class="row">
        <div class="col-6">
            <div class="card">
                <div class="card-header">Circular RNA Ring Structure</div>
                <div class="card-body" style="text-align:center;">
                    {}
                </div>
            </div>
        </div>
        <div class="col-6">
            {}
        </div>
    </div>
    {}""".format(circ_svg, pairing_matrix, struct_3d)

    return """
    <div class="nature-header">
        <h1>circRNA Analysis Report</h1>
        <div class="subtitle">Comprehensive characterization of circular RNA therapeutic candidate</div>
        <div class="meta">
            <div class="meta-item"><span class="meta-label">Sequence:</span><span class="meta-value">{} nt</span></div>
            <div class="meta-item"><span class="meta-label">GC:</span><span class="meta-value">{:.1f}%</span></div>
            <div class="meta-item"><span class="meta-label">Safety:</span><span class="meta-value">{:.2f}</span></div>
            <div class="meta-item"><span class="meta-label">Backend:</span><span class="meta-value">{}</span></div>
        </div>
    </div>
    <div class="card">
        <div class="card-header">Sequence</div>
        <div class="card-body"><div class="sequence-display">{}</div></div>
    </div>
    <div class="stat-grid">
        <div class="stat-box {}"><div class="stat-value {}">{:.0f}%</div><div class="stat-label">GC Content</div></div>
        <div class="stat-box {}"><div class="stat-value {}">{:.2f}</div><div class="stat-label">{}</div></div>
        <div class="stat-box"><div class="stat-value">{:.3f}</div><div class="stat-label">Immune Score</div></div>
        <div class="stat-box"><div class="stat-value">{}</div><div class="stat-label">Length (nt)</div></div>
    </div>
    <div class="row">
        <div class="col-4">
            <div class="card"><div class="card-header">Nucleotide Composition</div>
            <div class="card-body"><div id="nucleotidePie" class="chart-small"></div></div></div>
        </div>
        <div class="col-4">
            <div class="card"><div class="card-header">Innate Immune</div>
            <div class="card-body"><div id="immuneRadar" class="chart-small"></div></div></div>
        </div>
        <div class="col-4">
            <div class="card"><div class="card-header">TorusFold</div>
            <div class="card-body"><div id="torusRadar" class="chart-small"></div></div></div>
        </div>
    </div>
    {}
    {}
    <div class="row">
        <div class="col-6">
            <div class="card"><div class="card-header">Immune Sensor Activation</div>
            <div class="card-body">
                <table><thead><tr><th>Sensor</th><th>Score</th><th>Status</th></tr></thead><tbody>
                    <tr><td>TLR3</td><td>{:.4f}</td><td><span class="badge {}">{}</span></td></tr>
                    <tr><td>TLR7</td><td>{:.4f}</td><td><span class="badge {}">{}</span></td></tr>
                    <tr><td>TLR8</td><td>{:.4f}</td><td><span class="badge {}">{}</span></td></tr>
                    <tr><td>RIG-I</td><td>{:.4f}</td><td><span class="badge {}">{}</span></td></tr>
                    <tr><td>MDA5</td><td>{:.4f}</td><td><span class="badge {}">{}</span></td></tr>
                    <tr><td>PKR</td><td>{:.4f}</td><td><span class="badge {}">{}</span></td></tr>
                </tbody></table>
            </div></div>
        </div>
        <div class="col-6">
            <div class="card"><div class="card-header">TorusFold Metrics</div>
            <div class="card-body">
                <table><thead><tr><th>Metric</th><th>Score</th><th>Interpretation</th></tr></thead><tbody>
                    <tr><td>Stability</td><td>{:.2f}</td><td>Structural integrity</td></tr>
                    <tr><td>Translation</td><td>{:.2f}</td><td>Protein expression</td></tr>
                    <tr><td>Evasion</td><td>{:.2f}</td><td>Avoid detection</td></tr>
                    <tr><td>Delivery</td><td>{:.2f}</td><td>Cellular uptake</td></tr>
                </tbody></table>
            </div></div>
        </div>
    </div>
    <script>
    Plotly.newPlot('nucleotidePie', [{{type:'pie', labels:['A','U','G','C'], values:[{},{},{},{}], marker:{{colors:['#3498db','#e74c3c','#27ae60','#f39c12']}}, hole:0.4}}],
        {{paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}, legend:{{orientation:'h', y:-0.1}}}});
    Plotly.newPlot('immuneRadar', [{{type:'scatterpolar', r:[{},{},{},{},{},{}], theta:['TLR3','TLR7','TLR8','RIG-I','MDA5','PKR'], fill:'toself', marker:{{color:'#c41e3a'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,1],gridcolor:'#30363d'}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});
    Plotly.newPlot('torusRadar', [{{type:'scatterpolar', r:[{},{},{},{}], theta:['Stability','Translation','Evasion','Delivery'], fill:'toself', marker:{{color:'#27ae60'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,1],gridcolor:'#30363d'}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});
    </script>""".format(
        length, gc_pct, safety, backend.upper(), _format_sequence(seq),
        gc_class, gc_class, gc_pct,
        safety_class, safety_class, safety, safety_label,
        immune.get('innate_score', 0), length,
        pk_html,
        structure_section,
        immune.get('tlr3', 0), 'badge-success' if immune.get('tlr3', 0) < 0.3 else 'badge-warning', 'LOW' if immune.get('tlr3', 0) < 0.3 else 'MOD',
        immune.get('tlr7', 0), 'badge-success' if immune.get('tlr7', 0) < 0.3 else 'badge-warning', 'LOW' if immune.get('tlr7', 0) < 0.3 else 'MOD',
        immune.get('tlr8', 0), 'badge-success' if immune.get('tlr8', 0) < 0.3 else 'badge-warning', 'LOW' if immune.get('tlr8', 0) < 0.3 else 'MOD',
        immune.get('rigi', 0), 'badge-success' if immune.get('rigi', 0) < 0.3 else 'badge-warning', 'LOW' if immune.get('rigi', 0) < 0.3 else 'MOD',
        immune.get('mda5', 0), 'badge-success' if immune.get('mda5', 0) < 0.3 else 'badge-warning', 'LOW' if immune.get('mda5', 0) < 0.3 else 'MOD',
        immune.get('pkr', 0), 'badge-success' if immune.get('pkr', 0) < 0.3 else 'badge-warning', 'LOW' if immune.get('pkr', 0) < 0.3 else 'MOD',
        tf.get('stability', 0), tf.get('translation', 0), tf.get('immune_evasion', 0), tf.get('delivery', 0),
        a_count, u_count, g_count, c_count,
        immune.get('tlr3', 0), immune.get('tlr7', 0), immune.get('tlr8', 0),
        immune.get('rigi', 0), immune.get('mda5', 0), immune.get('pkr', 0),
        tf.get('stability', 0), tf.get('translation', 0), tf.get('immune_evasion', 0), tf.get('delivery', 0)
    )


def _drug_body(data: Dict[str, Any]) -> str:
    """Generate drug ADMET visualization body."""
    admet = data.get("admet", {})
    overall = admet.get('overall_risk', 0)
    risk_class = 'success' if overall < 0.3 else 'warning' if overall < 0.6 else 'danger'
    risk_label = 'LOW RISK' if overall < 0.3 else 'MODERATE' if overall < 0.6 else 'HIGH RISK'

    return """
    <div class="nature-header">
        <h1>Drug ADMET Report</h1>
        <div class="subtitle">Absorption, Distribution, Metabolism, Excretion, Toxicity</div>
        <div class="meta">
            <div class="meta-item"><span class="meta-label">Compound:</span><span class="meta-value">{}</span></div>
        </div>
    </div>
    <div class="stat-grid">
        <div class="stat-box {}"><div class="stat-value {}">{:.2f}</div><div class="stat-label">{}</div></div>
        <div class="stat-box"><div class="stat-value">{:.2f}</div><div class="stat-label">Druglikeness</div></div>
    </div>
    <div class="row">
        <div class="col-6">
            <div class="card"><div class="card-header">ADMET Profile</div>
            <div class="card-body"><div id="admetRadar" class="chart-container"></div></div></div>
        </div>
        <div class="col-6">
            <div class="card"><div class="card-header">Risk Assessment</div>
            <div class="card-body">
                <table><thead><tr><th>Parameter</th><th>Value</th><th>Risk</th></tr></thead><tbody>
                    <tr><td>hERG</td><td>{:.3f}</td><td><span class="badge {}">{}</span></td></tr>
                    <tr><td>Hepatotoxicity</td><td>{:.3f}</td><td><span class="badge {}">{}</span></td></tr>
                    <tr><td>CYP</td><td>{:.3f}</td><td><span class="badge {}">{}</span></td></tr>
                    <tr><td>AMES</td><td>{:.3f}</td><td><span class="badge {}">{}</span></td></tr>
                </tbody></table>
            </div></div>
        </div>
    </div>
    <script>
    Plotly.newPlot('admetRadar', [{{type:'scatterpolar', r:[{},{},{},{},{},{}], theta:['Overall','hERG','Hepato','CYP','AMES','Druglike'], fill:'toself', marker:{{color:'#c41e3a'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,1]}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});
    </script>""".format(
        data.get('input', 'Unknown'),
        risk_class, risk_class, overall, risk_label,
        admet.get('druglikeness', 0),
        admet.get('hERG_risk', 0), 'badge-success' if admet.get('hERG_risk', 0) < 0.3 else 'badge-warning', 'LOW' if admet.get('hERG_risk', 0) < 0.3 else 'MOD',
        admet.get('hepatotoxicity', 0), 'badge-success' if admet.get('hepatotoxicity', 0) < 0.3 else 'badge-warning', 'LOW' if admet.get('hepatotoxicity', 0) < 0.3 else 'MOD',
        admet.get('cyp_total_risk', 0), 'badge-success' if admet.get('cyp_total_risk', 0) < 0.3 else 'badge-warning', 'LOW' if admet.get('cyp_total_risk', 0) < 0.3 else 'MOD',
        admet.get('ames_positive', 0), 'badge-success' if admet.get('ames_positive', 0) < 0.5 else 'badge-danger', 'NEG' if admet.get('ames_positive', 0) < 0.5 else 'POS',
        overall, admet.get('hERG_risk', 0), admet.get('hepatotoxicity', 0),
        admet.get('cyp_total_risk', 0), admet.get('ames_positive', 0), 1 - admet.get('druglikeness', 0.5)
    )


def _epitope_body(data: Dict[str, Any]) -> str:
    """Generate epitope MHC visualization body."""
    score = data.get("binding_score", 0)
    score_class = 'success' if score > 0.7 else 'warning' if score > 0.4 else 'danger'
    binding = data.get('binding_affinity', 'UNKNOWN')
    color = '#27ae60' if score > 0.7 else '#f39c12' if score > 0.4 else '#e74c3c'

    return """
    <div class="nature-header">
        <h1>Epitope MHC Binding Report</h1>
        <div class="subtitle">Major Histocompatibility Complex Binding Prediction</div>
        <div class="meta">
            <div class="meta-item"><span class="meta-label">Sequence:</span><span class="meta-value">{}</span></div>
            <div class="meta-item"><span class="meta-label">Allele:</span><span class="meta-value">{}</span></div>
        </div>
    </div>
    <div class="stat-grid">
        <div class="stat-box {}"><div class="stat-value {}">{:.2f}</div><div class="stat-label">{}</div></div>
        <div class="stat-box"><div class="stat-value">{}</div><div class="stat-label">Length (aa)</div></div>
    </div>
    <div class="row">
        <div class="col-6">
            <div class="card"><div class="card-header">Binding Affinity</div>
            <div class="card-body"><div id="bindingGauge" class="chart-container"></div></div></div>
        </div>
        <div class="col-6">
            <div class="card"><div class="card-header">Interpretation</div>
            <div class="card-body">
                <table><thead><tr><th>Score Range</th><th>Binding</th><th>IC50 (nM)</th></tr></thead><tbody>
                    <tr><td>0.80-1.00</td><td><span class="badge badge-success">STRONG</span></td><td>&lt; 50</td></tr>
                    <tr><td>0.50-0.80</td><td><span class="badge badge-warning">MODERATE</span></td><td>50-500</td></tr>
                    <tr><td>0.00-0.50</td><td><span class="badge badge-danger">WEAK</span></td><td>&gt; 500</td></tr>
                </tbody></table>
            </div></div>
        </div>
    </div>
    <script>
    Plotly.newPlot('bindingGauge', [{{type:'indicator', mode:'gauge+number', value:{}, gauge:{{axis:{{range:[0,1]}}, bar:{{color:'{}'}}, steps:[{{range:[0,0.5],color:'rgba(231,76,60,0.3)'}},{{range:[0.5,0.8],color:'rgba(243,156,18,0.3)'}},{{range:[0.8,1],color:'rgba(39,174,96,0.3)'}}]}}}}],
        {{paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});
    </script>""".format(
        data.get('sequence', 'N/A'), data.get('allele', 'N/A'),
        score_class, score_class, score, binding,
        data.get('length', 0),
        score, color
    )


def _simulacrum_body(data: Dict[str, Any]) -> str:
    """Generate TNBC Simulacrum visualization body.

    Components:
    - Header with tumor metrics
    - Tumor biology panel (growth, CSC, heterogeneity)
    - TME panel (immune cells, CAF, ECM)
    - Immunophenogram (multidimensional cancer-immune map)
    - Treatment response panel
    - Clinical outcome panel
    - Time-series plots
    """
    import numpy as np

    # Extract state data
    state = data.get("state", {})
    history = data.get("history", [])
    step = data.get("step", 0)

    # Tumor metrics
    volume = state.get("tum_volume", 50.0)
    growth_rate = state.get("tum_growth_rate", 0.027)
    subtype = state.get("sub_molecular_subtype", "BLIS")
    proliferation = state.get("tum_proliferation_index", 0.3)

    # CSC metrics
    csc_fraction = state.get("csc_fraction", 0.02)
    csc_resistance = state.get("csc_chemo_resistance", 5.0)

    # Heterogeneity
    n_subclones = state.get("het_n_subclones", 4)
    diversity = state.get("het_diversity_index", 0.3)

    # Immune metrics
    cd8_count = state.get("imm_cd8_count", 100.0)
    cd4_count = state.get("imm_cd4_count", 150.0)
    nk_count = state.get("imm_nk_count", 50.0)
    treg_count = state.get("imm_treg_count", 20.0)
    mdsc_count = state.get("imm_mdsc_count", 30.0)
    til_density = state.get("imm_til_density", 0.2)
    t_cell_activation = state.get("imm_t_cell_activation", 0.3)
    t_cell_exhaustion = state.get("imm_t_cell_exhaustion", 0.1)

    # Evasion metrics
    pd_l1 = state.get("evs_pd_l1_expression", 0.2)
    mhc_i_downreg = state.get("evs_mhc_i_downreg", 0.1)
    tgf_beta = state.get("evs_tgf_beta", 0.15)

    # Immunoediting phase
    ied_phase = state.get("ied_phase", "elimination")
    immune_pressure = state.get("ied_immune_pressure", 0.5)
    evasion_pressure = state.get("ied_evasion_pressure", 0.3)

    # CAF/ECM
    caf_activation = state.get("caf_activation", 0.2)
    ecm_density = state.get("caf_ecm_density", 0.3)

    # Drug metrics
    drug_concentration = state.get("drg_concentration", 0.0)
    drug_effect = state.get("drg_effect", 0.0)
    resistance_level = state.get("drg_resistance_level", 0.0)

    # Clinical
    recist = state.get("cli_recist_response", "SD")
    os_months = state.get("cli_os_months", 24.0)
    pfs_months = state.get("cli_pfs_months", 12.0)

    # Generate time-series data if history available
    time_data = []
    volume_history = []
    cd8_history = []
    if history:
        for h in history:
            time_data.append(h.get("step", 0))
            volume_history.append(h.get("tum_volume", 50.0))
            cd8_history.append(h.get("imm_cd8_count", 100.0))

    # Generate Immunophenogram
    immunophenogram = _generate_immunophenogram(state)

    # Generate TME SVG
    tme_svg = generate_tme_svg(state)

    # Phase colors
    phase_colors = {
        "elimination": "#27ae60",
        "equilibrium": "#f39c12",
        "escape": "#e74c3c"
    }
    phase_color = phase_colors.get(ied_phase, "#7f8c8d")

    # RECIST badge
    recist_badge = generate_recist_badge(-volume * 0.01)  # Mock change

    # Generate dynamic components
    dynamic_tumor = _generate_dynamic_tumor_plot(history)
    dynamic_activation = _generate_dynamic_activation_plot(state, history)
    dynamic_tme_svg = _generate_dynamic_tme_svg(state)
    animated_timeline = _generate_animated_timeline_slider(history)

    return """
    <div class="nature-header">
        <h1>TNBC Simulacrum Report</h1>
        <div class="subtitle">Triple-Negative Breast Cancer Digital Twin Simulation</div>
        <div class="meta">
            <div class="meta-item"><span class="meta-label">Volume:</span><span class="meta-value">{:.1f} mm³</span></div>
            <div class="meta-item"><span class="meta-label">Subtype:</span><span class="meta-value">{}</span></div>
            <div class="meta-item"><span class="meta-label">Phase:</span><span class="meta-value" style="color:{}">{}</span></div>
            <div class="meta-item"><span class="meta-label">Step:</span><span class="meta-value">{}</span></div>
        </div>
    </div>

    <!-- Key Statistics -->
    <div class="stat-grid">
        <div class="stat-box danger"><div class="stat-value danger">{:.0f}</div><div class="stat-label">Tumor (mm³)</div></div>
        <div class="stat-box"><div class="stat-value">{}</div><div class="stat-label">Subclones</div></div>
        <div class="stat-box {}"><div class="stat-value {}">{}</div><div class="stat-label">IED Phase</div></div>
        <div class="stat-box"><div class="stat-value">{:.0f}</div><div class="stat-label">CD8+ T cells</div></div>
    </div>

    <!-- TME Visualization with Animation -->
    <div class="row">
        <div class="col-6">
            <div class="card">
                <div class="card-header">Animated TME</div>
                <div class="card-body" style="text-align:center;">
                    {}
                </div>
            </div>
        </div>
        <div class="col-6">
            <div class="card">
                <div class="card-header">Immunophenogram</div>
                <div class="card-body">
                    {}
                </div>
            </div>
        </div>
    </div>

    <!-- Dynamic Visualizations -->
    <div class="row">
        <div class="col-6">
            {}
        </div>
        <div class="col-6">
            {}
        </div>
    </div>
    {}

    <!-- Immune Landscape -->
    <div class="row">
        <div class="col-4">
            <div class="card"><div class="card-header">Immune Cell Radar</div>
            <div class="card-body"><div id="immuneRadar" class="chart-small"></div></div></div>
        </div>
        <div class="col-4">
            <div class="card"><div class="card-header">Immune Activity</div>
            <div class="card-body"><div id="activityRadar" class="chart-small"></div></div></div>
        </div>
        <div class="col-4">
            <div class="card"><div class="card-header">Evasion Factors</div>
            <div class="card-body"><div id="evasionRadar" class="chart-small"></div></div></div>
        </div>
    </div>

    <!-- Tumor Biology -->
    <div class="card">
        <div class="card-header">Tumor Biology & CSC Dynamics</div>
        <div class="card-body">
            <div class="row">
                <div class="col-6">
                    <table><thead><tr><th>Parameter</th><th>Value</th><th>Status</th></tr></thead><tbody>
                        <tr><td>Growth Rate</td><td>{:.3f}</td><td><span class="badge badge-danger">HIGH</span></td></tr>
                        <tr><td>Proliferation (Ki-67)</td><td>{:.1%}</td><td><span class="badge badge-warning">MOD</span></td></tr>
                        <tr><td>CSC Fraction</td><td>{:.2%}</td><td><span class="badge badge-warning">MOD</span></td></tr>
                        <tr><td>CSC Resistance</td><td>{:.1f}x</td><td><span class="badge badge-danger">HIGH</span></td></tr>
                        <tr><td>Diversity Index</td><td>{:.2f}</td><td><span class="badge badge-info">INFO</span></td></tr>
                    </tbody></table>
                </div>
                <div class="col-6">
                    <table><thead><tr><th>Subclone</th><th>Fraction</th><th>Driver</th></tr></thead><tbody>
                        <tr><td>Dominant</td><td>{:.0%}</td><td>PIK3CA</td></tr>
                        <tr><td>Resistant</td><td>{:.0%}</td><td>BRCA1</td></tr>
                        <tr><td>Minor 1</td><td>15%%</td><td>TP53</td></tr>
                        <tr><td>Minor 2</td><td>10%%</td><td>MYC</td></tr>
                    </tbody></table>
                </div>
            </div>
        </div>
    </div>

    <!-- Treatment Response -->
    <div class="card">
        <div class="card-header">Treatment Response & Clinical Outcome</div>
        <div class="card-body">
            <div class="row">
                <div class="col-6">
                    <div class="stat-grid">
                        <div class="stat-box {}"><div class="stat-value {}">{}</div><div class="stat-label">RECIST</div></div>
                        <div class="stat-box"><div class="stat-value">{:.1f}</div><div class="stat-label">PFS (mo)</div></div>
                        <div class="stat-box"><div class="stat-value">{:.1f}</div><div class="stat-label">OS (mo)</div></div>
                        <div class="stat-box danger"><div class="stat-value danger">{:.0%}</div><div class="stat-label">Resistance</div></div>
                    </div>
                </div>
                <div class="col-6">
                    <table><thead><tr><th>Biomarker</th><th>Value</th><th>Clinical Significance</th></tr></thead><tbody>
                        <tr><td>PD-L1 CPS</td><td>{:.0f}</td><td>IC eligible</td></tr>
                        <tr><td>TIL Density</td><td>{:.0%}</td><td>Prognostic</td></tr>
                        <tr><td>BRCA Status</td><td>{}</td><td>PARPi candidate</td></tr>
                        <tr><td>TMB</td><td>{:.0f} mut/Mb</td><td>IO response</td></tr>
                    </tbody></table>
                </div>
            </div>
        </div>
    </div>

    <!-- Time Series -->
    <div class="card">
        <div class="card-header">Simulation Dynamics</div>
        <div class="card-body">
            <div class="row">
                <div class="col-6"><div id="volumeCurve" class="chart-container"></div></div>
                <div class="col-6"><div id="immuneCurve" class="chart-container"></div></div>
            </div>
        </div>
    </div>

    <script>
    // Immune cell radar
    Plotly.newPlot('immuneRadar', [{{type:'scatterpolar', r:[{},{},{},{},{},{}], theta:['CD8','CD4','NK','Treg','MDSC','Macro'], fill:'toself', marker:{{color:'#27ae60'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,200],gridcolor:'#30363d'}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});

    // Activity radar
    Plotly.newPlot('activityRadar', [{{type:'scatterpolar', r:[{},{},{},{},{}], theta:['Activation','Exhaustion','IFN-γ','IL-2','TNF-α'], fill:'toself', marker:{{color:'#3498db'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,1],gridcolor:'#30363d'}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});

    // Evasion radar
    Plotly.newPlot('evasionRadar', [{{type:'scatterpolar', r:[{},{},{},{},{}], theta:['PD-L1','MHC-I↓','TGF-β','IDO','Galectin-9'], fill:'toself', marker:{{color:'#e74c3c'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,1],gridcolor:'#30363d'}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});

    // Volume curve
    Plotly.newPlot('volumeCurve', [{{type:'scatter', x:{}, y:{}, mode:'lines', fill:'tozeroy', line:{{color:'#e74c3c',width:2}}}}],
        {{xaxis:{{title:'Time Step',gridcolor:'#30363d'}}, yaxis:{{title:'Volume (mm³)',gridcolor:'#30363d'}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});

    // Immune curve
    Plotly.newPlot('immuneCurve', [{{type:'scatter', x:{}, y:{}, mode:'lines', line:{{color:'#27ae60',width:2}}}}],
        {{xaxis:{{title:'Time Step',gridcolor:'#30363d'}}, yaxis:{{title:'CD8+ Count',gridcolor:'#30363d'}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{{color:'#ecf0f1'}}}});
    </script>
    """.format(
        float(volume), subtype, phase_color, ied_phase.upper(), int(step),
        float(volume), int(n_subclones), 'success' if ied_phase == 'elimination' else 'warning' if ied_phase == 'equilibrium' else 'danger',
        phase_color, ied_phase.upper(), float(cd8_count),
        dynamic_tme_svg, immunophenogram,
        dynamic_tumor, dynamic_activation, animated_timeline,
        float(growth_rate), float(proliferation), float(csc_fraction), float(csc_resistance), float(diversity),
        0.6, 0.05,  # Mock subclone fractions
        'success' if recist in ['CR', 'PR'] else 'warning' if recist == 'SD' else 'danger',
        'success' if recist in ['CR', 'PR'] else 'warning' if recist == 'SD' else 'danger',
        recist,
        float(pfs_months), float(os_months), float(resistance_level),
        float(state.get('bio_pd_l1_cps', 10.0)), float(til_density),
        'Wild-type' if state.get('bio_brca_status', 0) == 0 else 'BRCA1 mut' if state.get('bio_brca_status', 0) == 1 else 'BRCA2 mut',
        float(state.get('bio_tmb', 5.0)),
        float(cd8_count), float(cd4_count), float(nk_count), float(treg_count), float(mdsc_count), float(state.get('imm_macrophage_count', 80.0)),
        float(t_cell_activation), float(t_cell_exhaustion), float(state.get('imm_ifn_gamma', 0.2)), float(state.get('imm_il2', 0.15)), float(state.get('imm_tnf_alpha', 0.15)),
        float(pd_l1), float(mhc_i_downreg), float(tgf_beta), float(state.get('evs_ido_activity', 0.05)), float(state.get('evs_gal3_expression', 0.2)),
        json.dumps(time_data), json.dumps(volume_history),
        json.dumps(time_data), json.dumps(cd8_history)
    )


def _generate_immunophenogram(state: Dict[str, Any]) -> str:
    """Generate Immunophenogram - multidimensional cancer-immune landscape map.

    Based on Thorsson et al. 2018 (Nature) immunophenogram concept.
    Shows: T cell infiltration, cytotoxic activity, checkpoint expression,
    macrophage polarization, TCR diversity.
    """
    import numpy as np

    # Extract metrics
    cd8 = state.get("imm_cd8_count", 100.0)
    cd4 = state.get("imm_cd4_count", 150.0)
    nk = state.get("imm_nk_count", 50.0)
    cytotoxicity = state.get("imm_nk_cytotoxicity", 0.3) + state.get("imm_t_cell_activation", 0.3)
    pd_l1 = state.get("evs_pd_l1_expression", 0.2)
    ctla4 = 0.15  # Mock
    m1_frac = state.get("imm_m1_fraction", 0.5)
    m2_frac = state.get("imm_m2_fraction", 0.5)
    tcr_diversity = 0.5  # Mock

    # Normalize values to 0-1 range
    infiltration = min(1.0, (cd8 + cd4 + nk) / 500)

    return """
    <div style="display:flex;justify-content:center;align-items:center;padding:20px;">
        <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="300" height="300" fill="#1a1b26" rx="8"/>

            <!-- Title -->
            <text x="150" y="20" text-anchor="middle" fill="#ecf0f1" font-size="12" font-weight="bold">Immunophenogram</text>

            <!-- Grid lines -->
            <line x1="30" y1="50" x2="270" y2="50" stroke="#30363d" stroke-width="1"/>
            <line x1="30" y1="100" x2="270" y2="100" stroke="#30363d" stroke-width="1"/>
            <line x1="30" y1="150" x2="270" y2="150" stroke="#30363d" stroke-width="1"/>
            <line x1="30" y1="200" x2="270" y2="200" stroke="#30363d" stroke-width="1"/>
            <line x1="30" y1="250" x2="270" y2="250" stroke="#30363d" stroke-width="1"/>

            <!-- Labels -->
            <text x="20" y="55" fill="#7f8c8d" font-size="9">1.0</text>
            <text x="20" y="105" fill="#7f8c8d" font-size="9">0.8</text>
            <text x="20" y="155" fill="#7f8c8d" font-size="9">0.6</text>
            <text x="20" y="205" fill="#7f8c8d" font-size="9">0.4</text>
            <text x="20" y="255" fill="#7f8c8d" font-size="9">0.2</text>

            <!-- Axes labels -->
            <text x="50" y="285" fill="#a9b1d6" font-size="10" text-anchor="middle">T Cell</text>
            <text x="100" y="285" fill="#a9b1d6" font-size="10" text-anchor="middle">Cytotox</text>
            <text x="150" y="285" fill="#a9b1d6" font-size="10" text-anchor="middle">CP</text>
            <text x="200" y="285" fill="#a9b1d6" font-size="10" text-anchor="middle">Macro</text>
            <text x="250" y="285" fill="#a9b1d6" font-size="10" text-anchor="middle">TCR</text>

            <!-- Heatmap cells -->
            <!-- T Cell Infiltration -->
            <rect x="35" y="{}" width="30" height="{}" fill="{}" opacity="0.8"/>

            <!-- Cytotoxicity -->
            <rect x="85" y="{}" width="30" height="{}" fill="{}" opacity="0.8"/>

            <!-- Checkpoint -->
            <rect x="135" y="{}" width="30" height="{}" fill="{}" opacity="0.8"/>

            <!-- Macrophage polarization -->
            <rect x="185" y="{}" width="30" height="{}" fill="{}" opacity="0.8"/>

            <!-- TCR diversity -->
            <rect x="235" y="{}" width="30" height="{}" fill="{}" opacity="0.8"/>

            <!-- Legend -->
            <rect x="30" y="10" width="12" height="12" fill="#27ae60"/>
            <text x="45" y="18" fill="#a9b1d6" font-size="8">Active</text>
            <rect x="80" y="10" width="12" height="12" fill="#f39c12"/>
            <text x="95" y="18" fill="#a9b1d6" font-size="8">Moderate</text>
            <rect x="130" y="10" width="12" height="12" fill="#e74c3c"/>
            <text x="145" y="18" fill="#a9b1d6" font-size="8">Suppressed</text>
        </svg>
    </div>
    """.format(
        # T Cell Infiltration
        250 - infiltration * 200, infiltration * 200,
        '#27ae60' if infiltration > 0.6 else '#f39c12' if infiltration > 0.3 else '#e74c3c',
        # Cytotoxicity
        250 - cytotoxicity * 200, cytotoxicity * 200,
        '#27ae60' if cytotoxicity > 0.6 else '#f39c12' if cytotoxicity > 0.3 else '#e74c3c',
        # Checkpoint (inverted - high expression is bad)
        250 - (1 - pd_l1) * 200, (1 - pd_l1) * 200,
        '#27ae60' if pd_l1 < 0.2 else '#f39c12' if pd_l1 < 0.5 else '#e74c3c',
        # Macrophage polarization (M1/M2 ratio)
        250 - m1_frac * 200, m1_frac * 200,
        '#27ae60' if m1_frac > 0.6 else '#f39c12' if m1_frac > 0.4 else '#e74c3c',
        # TCR diversity
        250 - tcr_diversity * 200, tcr_diversity * 200,
        '#27ae60' if tcr_diversity > 0.6 else '#f39c12' if tcr_diversity > 0.3 else '#e74c3c'
    )


def save_report(html: str, filename: str = None) -> str:
    """Save HTML report to file."""
    if filename is None:
        filename = "confluencia_nature_{}.html".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = Path(filename)
    path.write_text(html, encoding="utf-8")
    return str(path.absolute())


# =============================================================================
# Additional Visualization Components from Studio
# =============================================================================

def generate_tme_svg(state: Dict[str, Any]) -> str:
    """Generate TME (Tumor Microenvironment) SVG schematic.

    Components:
    - Center: tumor mass circle
    - Surrounding: vasculature arcs
    - Inner dots: immune cells (CD8, NK, Treg, MDSC)
    - Outer ring: CAF/ECM
    - PD-L1 shield arc
    - Pressure arrows: immune vs evasion
    """
    import math

    # Tokyo Night palette
    TN = {
        "bg": "#1a1b26", "surface": "#24283b", "border": "#414d68",
        "text": "#c0caf5", "muted": "#a9b1d6",
        "blue": "#7aa2f7", "green": "#9ece6a", "yellow": "#e0af68",
        "red": "#f7768e", "purple": "#bb9af7", "cyan": "#7dcfff",
        "orange": "#ff9e64", "teal": "#73daca",
    }

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

    W, H = 400, 400
    cx, cy = W // 2, H // 2
    tumor_r = max(20, min(80, 20 + 15 * math.log10(max(1, volume))))

    elements = []
    elements.append('<rect width="{}" height="{}" fill="{}" rx="12"/>'.format(W, H, TN["bg"]))

    # CAF/ECM outer ring
    ecm_opacity = min(0.8, ecm * 1.5)
    ecm_r = tumor_r + 40
    elements.append('<circle cx="{}" cy="{}" r="{}" fill="none" stroke="{}" stroke-width="8" opacity="{:.2f}" stroke-dasharray="4,6"/>'.format(
        cx, cy, ecm_r, TN["yellow"], ecm_opacity))

    # Tumor mass
    elements.append('<circle cx="{}" cy="{}" r="{}" fill="{}" opacity="0.8"/>'.format(
        cx, cy, tumor_r, TN["purple"]))
    elements.append('<text x="{}" y="{}" text-anchor="middle" fill="{}" font-size="10">Tumor</text>'.format(
        cx, cy + 4, TN["text"]))

    # Vasculature
    n_vessels = max(3, int(mvd * 12))
    vasc_r = tumor_r + 25
    for i in range(n_vessels):
        angle = i * 360 / n_vessels
        rad = math.radians(angle)
        x1 = cx + vasc_r * math.cos(rad)
        y1 = cy + vasc_r * math.sin(rad)
        x2 = cx + (tumor_r + 5) * math.cos(rad)
        y2 = cy + (tumor_r + 5) * math.sin(rad)
        elements.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="2" opacity="0.6"/>'.format(
            x1, y1, x2, y2, TN["red"]))

    # Immune cells
    def place_cells(count, color, label, max_r, min_r):
        cells = []
        n = min(int(count) // 10, 12)
        for i in range(n):
            import random
            angle = random.uniform(0, 360)
            r = random.uniform(min_r, max_r)
            rad = math.radians(angle)
            x = cx + r * math.cos(rad)
            y = cy + r * math.sin(rad)
            cells.append('<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="{}"/>'.format(x, y, color))
        return cells

    elements.extend(place_cells(cd8, TN["green"], "CD8", ecm_r - 10, tumor_r + 10))
    elements.extend(place_cells(nk, TN["cyan"], "NK", ecm_r - 10, tumor_r + 10))

    # Phase label
    phase_colors = {"elimination": TN["blue"], "equilibrium": TN["yellow"], "escape": TN["red"]}
    phase_color = phase_colors.get(phase, TN["muted"])
    elements.append('<text x="{}" y="20" text-anchor="middle" fill="{}" font-size="11" font-weight="bold">{}</text>'.format(
        cx, phase_color, phase.upper()))

    # Legend
    elements.append('<text x="10" y="{}" fill="{}" font-size="9">CD8: {}</text>'.format(H - 60, TN["green"], cd8))
    elements.append('<text x="10" y="{}" fill="{}" font-size="9">NK: {}</text>'.format(H - 45, TN["cyan"], nk))
    elements.append('<text x="10" y="{}" fill="{}" font-size="9">Treg: {}</text>'.format(H - 30, TN["red"], treg))
    elements.append('<text x="10" y="{}" fill="{}" font-size="9">MDSC: {}</text>'.format(H - 15, TN["orange"], mdsc))

    return '<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">{}</svg>'.format(
        W, H, ''.join(elements))


def generate_recist_badge(change_pct: float) -> str:
    """Generate RECIST response badge HTML."""
    TN = {
        "green": "#27ae60", "teal": "#1abc9c", "yellow": "#f39c12", "red": "#e74c3c",
        "bg": "#1a1b26"
    }

    if change_pct <= -30:
        return '<span class="badge badge-success" style="background:rgba(39,174,96,0.2);color:#27ae60;">CR</span>'
    elif change_pct <= -10:
        return '<span class="badge badge-success" style="background:rgba(39,174,96,0.2);color:#27ae60;">PR</span>'
    elif change_pct <= 20:
        return '<span class="badge badge-warning" style="background:rgba(243,156,18,0.2);color:#f39c12;">SD</span>'
    else:
        return '<span class="badge badge-danger" style="background:rgba(231,76,60,0.2);color:#e74c3c;">PD</span>'


def generate_progress_bar(label: str, value: float, max_val: float = 1.0, color: str = "#2980b9") -> str:
    """Generate styled progress bar HTML."""
    pct = min(100, max(0, (value / max_val) * 100)) if max_val > 0 else 0

    return """
    <div style="margin: 8px 0;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="color:#7f8c8d;font-size:0.85em;">{}</span>
            <span style="color:#ecf0f1;font-size:0.85em;">{:.2f}</span>
        </div>
        <div style="background:#30363d;border-radius:4px;height:8px;overflow:hidden;">
            <div style="background:{};width:{:.1f}%;height:100%;border-radius:4px;"></div>
        </div>
    </div>""".format(label, value, color, pct)


def generate_regression_plots_html(y_true, y_pred, title: str = "Model Performance") -> str:
    """Generate regression diagnostic plots as base64 embedded images."""
    import numpy as np
    import io
    import base64

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        residual = yp - yt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor('#1a1b26')

        # Scatter plot
        axes[0].scatter(yt, yp, s=10, alpha=0.6, c='#7aa2f7')
        mn = min(yt.min(), yp.min())
        mx = max(yt.max(), yp.max())
        axes[0].plot([mn, mx], [mn, mx], 'w-', linewidth=1)
        axes[0].set_xlabel('y_true', color='#c0caf5')
        axes[0].set_ylabel('y_pred', color='#c0caf5')
        axes[0].set_title('Predicted vs Actual', color='#c0caf5')
        axes[0].set_facecolor('#1a1b26')
        axes[0].tick_params(colors='#7f8c8d')

        # Residual histogram
        axes[1].hist(residual[~np.isnan(residual)], bins=40, alpha=0.85, color='#27ae60')
        axes[1].set_xlabel('Residual', color='#c0caf5')
        axes[1].set_ylabel('Count', color='#c0caf5')
        axes[1].set_title('Residual Distribution', color='#c0caf5')
        axes[1].set_facecolor('#1a1b26')
        axes[1].tick_params(colors='#7f8c8d')

        # Residual vs predicted
        axes[2].scatter(yp, residual, s=10, alpha=0.6, c='#f7768e')
        axes[2].axhline(0, color='white', linewidth=1)
        axes[2].set_xlabel('y_pred', color='#c0caf5')
        axes[2].set_ylabel('Residual', color='#c0caf5')
        axes[2].set_title('Residual vs Prediction', color='#c0caf5')
        axes[2].set_facecolor('#1a1b26')
        axes[2].tick_params(colors='#7f8c8d')

        for ax in axes:
            ax.spines['bottom'].set_color('#414d68')
            ax.spines['top'].set_color('#414d68')
            ax.spines['left'].set_color('#414d68')
            ax.spines['right'].set_color('#414d68')

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='#1a1b26', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return '<img src="data:image/png;base64,{}" style="width:100%;border-radius:8px;"/>'.format(img_base64)
    except Exception:
        return '<div class="card-body" style="color:#7f8c8d;">Regression plots unavailable</div>'


# =============================================================================
# circRNA Circular Structure Visualization
# =============================================================================

def generate_circular_rna_svg(sequence: str, dot_bracket: str = None) -> str:
    """Generate circular RNA visualization SVG showing the ring structure.

    Features:
    - Circular backbone representing the closed loop
    - Base pairing arcs (stems) inside the circle
    - Color-coded nucleotides
    - Hairpin loops shown as bulges
    """
    import math

    # Tokyo Night palette
    TN = {
        "bg": "#1a1b26", "surface": "#24283b", "border": "#414d68",
        "text": "#c0caf5", "muted": "#a9b1d6",
        "blue": "#3498db", "green": "#27ae60", "yellow": "#f39c12",
        "red": "#e74c3c", "purple": "#bb9af7", "cyan": "#7dcfff",
        "orange": "#ff9e64", "teal": "#73daca",
    }

    NUCLEOTIDE_COLORS = {'A': TN["blue"], 'U': TN["red"], 'G': TN["green"], 'C': TN["yellow"]}

    n = len(sequence)
    if n == 0:
        return '<svg width="400" height="400"><text x="200" y="200" fill="#7f8c8d">No sequence</text></svg>'

    # SVG dimensions
    W, H = 500, 500
    cx, cy = W // 2, H // 2
    radius = min(W, H) // 2 - 80

    # Generate dot-bracket if not provided
    if not dot_bracket:
        dot_bracket = _estimate_dot_bracket(sequence)

    elements = []

    # Background
    elements.append('<rect width="{}" height="{}" fill="{}" rx="12"/>'.format(W, H, TN["bg"]))

    # Title
    elements.append('<text x="{}" y="30" text-anchor="middle" fill="{}" font-size="14" font-weight="bold">circRNA Ring Structure</text>'.format(cx, TN["text"]))
    elements.append('<text x="{}" y="50" text-anchor="middle" fill="{}" font-size="11">{} nt | Circular conformation</text>'.format(cx, TN["muted"], n))

    # Main circular backbone (RNA ring)
    elements.append('<circle cx="{}" cy="{}" r="{}" fill="none" stroke="{}" stroke-width="3" opacity="0.6"/>'.format(
        cx, cy, radius, TN["purple"]))

    # Calculate base positions around the circle
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]  # Start from top

    # Draw base pairing arcs (stems)
    pairs = _parse_dot_bracket_pairs(dot_bracket)
    for i, j in pairs:
        if i < n and j < n:
            # Draw arc between paired bases
            x1 = cx + radius * math.cos(angles[i])
            y1 = cy + radius * math.sin(angles[i])
            x2 = cx + radius * math.cos(angles[j])
            y2 = cy + radius * math.sin(angles[j])

            # Control points for curved arc inside the circle
            mid_angle = (angles[i] + angles[j]) / 2
            arc_radius = radius * 0.6  # Arc goes inward

            ctrl_x = cx + arc_radius * math.cos(mid_angle)
            ctrl_y = cy + arc_radius * math.sin(mid_angle)

            elements.append('<path d="M {:.1f} {:.1f} Q {:.1f} {:.1f} {:.1f} {:.1f}" fill="none" stroke="{}" stroke-width="2" opacity="0.7"/>'.format(
                x1, y1, ctrl_x, ctrl_y, x2, y2, TN["teal"]))

    # Draw nucleotide markers around the circle
    for i, base in enumerate(sequence.upper()):
        if base not in 'AUGC':
            continue

        x = cx + radius * math.cos(angles[i])
        y = cy + radius * math.sin(angles[i])

        # Small circle for each nucleotide
        color = NUCLEOTIDE_COLORS.get(base, TN["muted"])
        elements.append('<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="{}"/>'.format(x, y, color))

    # Legend
    legend_y = H - 60
    elements.append('<rect x="{}" y="{}" width="160" height="50" fill="{}" rx="6"/>'.format(
        W // 2 - 80, legend_y, TN["surface"]))

    for i, (base, color) in enumerate([('A', TN["blue"]), ('U', TN["red"]), ('G', TN["green"]), ('C', TN["yellow"])]):
        lx = W // 2 - 60 + i * 40
        elements.append('<circle cx="{}" cy="{}" r="6" fill="{}"/>'.format(lx, legend_y + 18, color))
        elements.append('<text x="{}" y="{}" text-anchor="middle" fill="{}" font-size="10">{}</text>'.format(
            lx, legend_y + 38, TN["muted"], base))

    # Junction indicator (5'-3' junction point)
    jx = cx + radius * math.cos(angles[0])
    jy = cy + radius * math.sin(angles[0])
    elements.append('<circle cx="{:.1f}" cy="{:.1f}" r="8" fill="none" stroke="{}" stroke-width="2"/>'.format(
        jx, jy, TN["cyan"]))
    elements.append('<text x="{}" y="{}" text-anchor="middle" fill="{}" font-size="9">5\'-3\'</text>'.format(
        jx, jy - 15, TN["cyan"]))

    return '<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">{}</svg>'.format(
        W, H, ''.join(elements))


def _estimate_dot_bracket(sequence: str) -> str:
    """Estimate dot-bracket notation for a sequence."""
    n = len(sequence)
    gc = sum(1 for c in sequence.upper() if c in 'GC') / n if n > 0 else 0

    # Simple stem-loop model
    stem_len = max(3, int(gc * 8))
    result = ['.'] * n

    # Add stem pairs
    i = 0
    while i < n - stem_len * 2 - 4:
        # Check for GC-rich region (potential stem)
        window = sequence[i:i+stem_len*2]
        window_gc = sum(1 for c in window.upper() if c in 'GC') / len(window) if window else 0

        if window_gc > 0.5:
            # Create stem-loop
            for j in range(stem_len):
                if i + j < n and i + stem_len + 4 + j < n:
                    result[i + j] = '('
                    result[i + stem_len + 4 + j] = ')'
            i += stem_len * 2 + 4
        else:
            i += 1

    return ''.join(result)


def _parse_dot_bracket_pairs(dot_bracket: str) -> list:
    """Parse dot-bracket notation to get base pair indices."""
    pairs = []
    stack = []

    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))

    return pairs


def generate_structure_3d_plotly(sequence: str, dot_bracket: str = None) -> str:
    """Generate detailed 3D RNA structure visualization using Plotly.

    Enhanced visualization with:
    - Individual nucleotide spheres with base-type colors
    - Backbone tube connecting nucleotides
    - Base pairing bridges with gradient colors
    - Stem-loop structural features
    - Legend and annotations
    - Interactive rotation/zoom/hover
    """
    import numpy as np
    import math

    n = len(sequence)
    if n == 0:
        return ""

    # Generate dot-bracket if not provided
    if not dot_bracket:
        dot_bracket = _estimate_dot_bracket(sequence)

    # Generate 3D coordinates for circular RNA
    # Parametric toroidal curve
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Major radius (overall circle)
    R = 12
    # Minor radius (local helix thickness)
    r = 2.0
    # Number of helical turns around the ring
    turns = 2.5

    # Toroidal coordinates - creates a circular RNA with helical twist
    x = (R + r * np.cos(turns * t)) * np.cos(t)
    y = (R + r * np.cos(turns * t)) * np.sin(t)
    z = r * np.sin(turns * t)

    # Nucleotide colors and properties
    color_map = {'A': '#3498db', 'U': '#e74c3c', 'G': '#27ae60', 'C': '#f39c12'}
    base_names = {'A': 'Adenine', 'U': 'Uracil', 'G': 'Guanine', 'C': 'Cytosine'}

    # Build traces
    traces = []

    # ============================================
    # 1. Backbone tube (semi-transparent ribbon)
    # ============================================
    # Create tube by connecting adjacent bases with small spheres
    tube_x, tube_y, tube_z = [], [], []
    for i in range(n):
        tube_x.append(x[i])
        tube_y.append(y[i])
        tube_z.append(z[i])
    # Close the ring
    tube_x.append(x[0])
    tube_y.append(y[0])
    tube_z.append(z[0])

    traces.append({
        "type": "scatter3d",
        "x": tube_x,
        "y": tube_y,
        "z": tube_z,
        "mode": "lines",
        "line": {
            "color": "#bb9af7",
            "width": 6,
            "opacity": 0.4
        },
        "hoverinfo": "skip",
        "name": "Backbone",
        "showlegend": False
    })

    # ============================================
    # 2. Individual nucleotide spheres
    # ============================================
    for i, base in enumerate(sequence.upper()):
        if base not in 'AUGC':
            continue

        color = color_map.get(base, '#7f8c8d')
        base_name = base_names.get(base, 'Unknown')

        # Add sphere for each nucleotide
        traces.append({
            "type": "scatter3d",
            "x": [x[i]],
            "y": [y[i]],
            "z": [z[i]],
            "mode": "markers",
            "marker": {
                "size": 8,
                "color": color,
                "opacity": 1.0,
                "line": {
                    "color": "#ecf0f1",
                    "width": 1
                }
            },
            "hovertemplate": f"Position {i+1}<br>Base: {base} ({base_name})<extra></extra>",
            "showlegend": False
        })

    # ============================================
    # 3. Base pairing bridges (stem regions)
    # ============================================
    pairs = _parse_dot_bracket_pairs(dot_bracket)

    for idx, (i, j) in enumerate(pairs):
        if i >= n or j >= n:
            continue

        # Create curved bridge between paired bases
        # Use quadratic Bezier curve
        mid_x = (x[i] + x[j]) / 2
        mid_y = (y[i] + y[j]) / 2
        mid_z = (z[i] + z[j]) / 2

        # Control point pulled toward center of torus
        ctrl_scale = 0.7
        ctrl_x = mid_x * ctrl_scale
        ctrl_y = mid_y * ctrl_scale
        ctrl_z = mid_z

        # Generate curve points
        curve_x, curve_y, curve_z = [], [], []
        for t_val in np.linspace(0, 1, 15):
            # Quadratic Bezier
            bx = (1-t_val)**2 * x[i] + 2*(1-t_val)*t_val * ctrl_x + t_val**2 * x[j]
            by = (1-t_val)**2 * y[i] + 2*(1-t_val)*t_val * ctrl_y + t_val**2 * y[j]
            bz = (1-t_val)**2 * z[i] + 2*(1-t_val)*t_val * ctrl_z + t_val**2 * z[j]
            curve_x.append(bx)
            curve_y.append(by)
            curve_z.append(bz)

        # Color gradient based on pair type
        base_i = sequence[i].upper() if i < n else 'N'
        base_j = sequence[j].upper() if j < n else 'N'
        pair_type = f"{base_i}-{base_j}"

        # Watson-Crick pairs (G-C, A-U) get different colors
        if (base_i, base_j) in [('G', 'C'), ('C', 'G'), ('A', 'U'), ('U', 'A')]:
            pair_color = '#73daca'  # Teal for WC pairs
        else:
            pair_color = '#f7768e'  # Red for non-canonical

        traces.append({
            "type": "scatter3d",
            "x": curve_x,
            "y": curve_y,
            "z": curve_z,
            "mode": "lines",
            "line": {
                "color": pair_color,
                "width": 3,
                "opacity": 0.8
            },
            "hovertemplate": f"Base pair: {i+1}-{j+1}<br>Type: {pair_type}<extra></extra>",
            "showlegend": idx == 0  # Only show legend for first pair
        })

    # ============================================
    # 4. Junction marker (5'-3' connection)
    # ============================================
    # Highlight the circular junction
    traces.append({
        "type": "scatter3d",
        "x": [x[0]],
        "y": [y[0]],
        "z": [z[0]],
        "mode": "markers",
        "marker": {
            "size": 14,
            "color": '#7dcfff',
            "symbol": 'diamond',
            "line": {
                "color": '#c0caf5',
                "width": 2
            }
        },
        "hovertemplate": "5'-3' Junction<br>Circular connection point<extra></extra>",
        "name": "Junction"
    })

    # ============================================
    # 5. Legend traces (invisible, for legend only)
    # ============================================
    legend_items = [
        ("Adenine (A)", '#3498db'),
        ("Uracil (U)", '#e74c3c'),
        ("Guanine (G)", '#27ae60'),
        ("Cytosine (C)", '#f39c12'),
        ("WC Base Pair", '#73daca'),
        ("Junction", '#7dcfff')
    ]

    for name, color in legend_items:
        traces.append({
            "type": "scatter3d",
            "x": [None],
            "y": [None],
            "z": [None],
            "mode": "markers",
            "marker": {"size": 8, "color": color},
            "name": name,
            "showlegend": True
        })

    # ============================================
    # Layout configuration
    # ============================================
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#ecf0f1", "family": "Source Sans Pro, sans-serif"},
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        "title": {
            "text": f"3D circRNA Structure ({n} nt)",
            "font": {"color": "#c41e3a", "size": 14},
            "x": 0.5
        },
        "scene": {
            "xaxis": {
                "visible": False,
                "range": [-15, 15]
            },
            "yaxis": {
                "visible": False,
                "range": [-15, 15]
            },
            "zaxis": {
                "visible": False,
                "range": [-5, 5]
            },
            "camera": {
                "eye": {"x": 1.8, "y": 1.8, "z": 0.8},
                "center": {"x": 0, "y": 0, "z": 0}
            },
            "aspectmode": "data",
            "bgcolor": "rgba(0,0,0,0)"
        },
        "showlegend": True,
        "legend": {
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(22, 27, 34, 0.8)",
            "bordercolor": "#30363d",
            "borderwidth": 1,
            "font": {"size": 10, "color": "#a9b1d6"}
        },
        "hovermode": "closest"
    }

    return """
    <div class="card">
        <div class="card-header">3D Structure Visualization</div>
        <div class="card-body">
            <div id="structure3d" style="height:500px;"></div>
            <div style="margin-top:15px;padding:15px;background:rgba(0,0,0,0.3);border-radius:6px;">
                <p style="color:#7f8c8d;font-size:0.85em;margin-bottom:8px;">
                    <strong style="color:#c41e3a;">Interaction Guide:</strong>
                </p>
                <ul style="color:#a9b1d6;font-size:0.85em;margin-left:20px;line-height:1.8;">
                    <li><span style="color:#3498db;">●</span> Adenine | <span style="color:#e74c3c;">●</span> Uracil | <span style="color:#27ae60;">●</span> Guanine | <span style="color:#f39c12;">●</span> Cytosine</li>
                    <li><span style="color:#73daca;">—</span> Watson-Crick base pairs (G-C, A-U)</li>
                    <li><span style="color:#7dcfff;">◆</span> 5'-3' junction (circular connection)</li>
                    <li>Drag to rotate | Scroll to zoom | Hover for details</li>
                </ul>
            </div>
        </div>
    </div>
    <script>
    Plotly.newPlot('structure3d', {}, {});
    </script>
    """.format(json.dumps(traces), json.dumps(layout))


def generate_pairing_matrix_html(sequence: str, dot_bracket: str = None) -> str:
    """Generate base pairing probability matrix as heatmap."""
    import numpy as np

    n = len(sequence)
    if n == 0:
        return ""

    if not dot_bracket:
        dot_bracket = _estimate_dot_bracket(sequence)

    # Create pairing matrix
    matrix = np.zeros((n, n))

    # Set diagonal to 1 (self-pairing baseline)
    np.fill_diagonal(matrix, 0.1)

    # Set base pairs
    pairs = _parse_dot_bracket_pairs(dot_bracket)
    for i, j in pairs:
        if i < n and j < n:
            matrix[i, j] = 1.0
            matrix[j, i] = 1.0

    # Add some probability near diagonal for visual effect
    for i in range(n - 1):
        matrix[i, i + 1] = 0.3
        matrix[i + 1, i] = 0.3

    return """
    <div class="card">
        <div class="card-header">Base Pairing Matrix</div>
        <div class="card-body">
            <div id="pairingMatrix" style="height:350px;"></div>
        </div>
    </div>
    <script>
    Plotly.newPlot('pairingMatrix', [{{
        type: 'heatmap',
        z: {},
        colorscale: [[0, '#1a1b26'], [0.3, '#24283b'], [0.7, '#73daca'], [1, '#27ae60']],
        showscale: true,
        colorbar: {{title: 'Pairing', titlefont: {{color: '#7f8c8d'}}, tickfont: {{color: '#7f8c8d'}}}}
    }}], {{
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {{color: '#ecf0f1'}},
        xaxis: {{title: 'Position', gridcolor: '#30363d'}},
        yaxis: {{title: 'Position', gridcolor: '#30363d'}}
    }});
    </script>
    """.format(json.dumps(matrix.tolist()))


# =============================================================================
# Dynamic Visualization for TNBC Simulacrum
# =============================================================================

def _generate_dynamic_tumor_plot(history: list) -> str:
    """Generate animated tumor growth visualization with Plotly.

    Features:
    - Animated timeline showing tumor volume evolution
    - Play/pause controls
    - Current state highlight
    """
    if not history or len(history) < 2:
        return "<div class='card-body' style='color:#7f8c8d;'>Insufficient history data</div>"

    import numpy as np

    steps = [h.get('step', i) for i, h in enumerate(history)]
    volumes = [h.get('tum_volume', 50) for h in history]

    # Create frames for animation
    frames = []
    for i in range(len(steps)):
        frame = {
            "name": f"frame{i}",
            "data": [{
                "type": "scatter",
                "x": steps[:i+1],
                "y": volumes[:i+1],
                "mode": "lines+markers",
                "line": {"color": "#e74c3c", "width": 3},
                "marker": {"size": 8, "color": "#e74c3c"},
                "fill": "tozeroy",
                "fillcolor": "rgba(231, 76, 60, 0.2)"
            }],
            "layout": {
                "title": {"text": f"Tumor Volume: {volumes[i]:.1f} mm³", "font": {"color": "#e74c3c"}}
            }
        }
        frames.append(frame)

    # Slider steps
    slider_steps = []
    for i in range(len(steps)):
        slider_steps.append({
            "method": "animate",
            "label": str(steps[i]),
            "args": [[f"frame{i}"], {
                "mode": "immediate",
                "transition": {"duration": 100},
                "frame": {"duration": 100, "redraw": True}
            }]
        })

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#ecf0f1"},
        "xaxis": {"title": "Time Step", "gridcolor": "#30363d", "range": [0, max(steps) * 1.1]},
        "yaxis": {"title": "Volume (mm³)", "gridcolor": "#30363d", "range": [0, max(volumes) * 1.2]},
        "hovermode": "closest",
        "sliders": [{
            "active": len(steps) - 1,
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 12, "color": "#ecf0f1"},
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "steps": slider_steps
        }],
        "updatemenus": [{
            "type": "buttons",
            "showactive": False,
            "y": -0.25,
            "x": 0,
            "xanchor": "left",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "fromcurrent": True,
                        "transition": {"duration": 50},
                        "frame": {"duration": 100, "redraw": True}
                    }]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "mode": "immediate",
                        "transition": {"duration": 0},
                        "frame": {"duration": 0, "redraw": False}
                    }]
                }
            ]
        }]
    }

    return """
    <div class="card">
        <div class="card-header">Dynamic Tumor Growth Animation</div>
        <div class="card-body">
            <div id="dynamicTumor" style="height:400px;"></div>
        </div>
    </div>
    <script>
    (function() {{
        var frames = {};
        var layout = {};
        var data = [{{type: 'scatter', x: [], y: [], mode: 'lines+markers', line: {{color: '#e74c3c', width: 3}}, marker: {{size: 8, color: '#e74c3c'}}, fill: 'tozeroy', fillcolor: 'rgba(231, 76, 60, 0.2)'}}];
        Plotly.newPlot('dynamicTumor', data, layout).then(function() {{
            Plotly.addFrames('dynamicTumor', frames);
        }});
    }})();
    </script>
    """.format(json.dumps(frames), json.dumps(layout))


def _generate_dynamic_activation_plot(state: Dict[str, Any], history: list) -> str:
    """Generate animated immune activation visualization.

    Shows dynamic changes in:
    - T cell activation over time
    - Cytokine levels (IFN-γ, IL-2, TNF-α)
    - Exhaustion markers
    """
    import numpy as np

    # Extract time series from history or generate simulated data
    if history and len(history) > 1:
        steps = [h.get('step', i) for i, h in enumerate(history)]
        activation = [h.get('imm_t_cell_activation', 0.3) for h in history]
        exhaustion = [h.get('imm_t_cell_exhaustion', 0.1) for h in history]
        ifn_gamma = [h.get('imm_ifn_gamma', 0.2) for h in history]
    else:
        # Generate simulated evolution
        steps = list(range(50))
        base_act = state.get('imm_t_cell_activation', 0.3)
        base_exh = state.get('imm_t_cell_exhaustion', 0.1)
        activation = [base_act * (1 + 0.3 * np.sin(i/10)) for i in steps]
        exhaustion = [base_exh + 0.01 * i for i in steps]
        ifn_gamma = [state.get('imm_ifn_gamma', 0.2) * (1 + 0.2 * np.sin(i/8)) for i in steps]

    return """
    <div class="card">
        <div class="card-header">Immune Activation Dynamics</div>
        <div class="card-body">
            <div id="activationDynamics" style="height:350px;"></div>
            <div style="margin-top:15px;display:flex;justify-content:center;gap:30px;">
                <span style="color:#3498db;"><span style="display:inline-block;width:12px;height:12px;background:#3498db;border-radius:50%;"></span> Activation</span>
                <span style="color:#f39c12;"><span style="display:inline-block;width:12px;height:12px;background:#f39c12;border-radius:50%;"></span> Exhaustion</span>
                <span style="color:#27ae60;"><span style="display:inline-block;width:12px;height:12px;background:#27ae60;border-radius:50%;"></span> IFN-γ</span>
            </div>
        </div>
    </div>
    <script>
    Plotly.newPlot('activationDynamics', [
        {{type: 'scatter', x: {}, y: {}, mode: 'lines', name: 'Activation', line: {{color: '#3498db', width: 2}}}},
        {{type: 'scatter', x: {}, y: {}, mode: 'lines', name: 'Exhaustion', line: {{color: '#f39c12', width: 2}}}},
        {{type: 'scatter', x: {}, y: {}, mode: 'lines', name: 'IFN-γ', line: {{color: '#27ae60', width: 2}}}}
    ], {{
        xaxis: {{title: 'Time Step', gridcolor: '#30363d'}},
        yaxis: {{title: 'Level', gridcolor: '#30363d', range: [0, 1]}},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {{color: '#ecf0f1'}},
        legend: {{orientation: 'h', y: -0.2}},
        updatemenus: [{{
            type: 'buttons',
            showactive: False,
            y: 1.1,
            x: 1,
            buttons: [
                {{label: 'Animate', method: 'animate', args: [null, {{fromcurrent: True, frame: {{duration: 100}}}}]}},
                {{label: 'Reset', method: 'update', args: [{{'y': [{}]}}, {{}}]}}
            ]
        }}]
    }});
    </script>
    """.format(
        json.dumps(steps), json.dumps(activation),
        json.dumps(steps), json.dumps(exhaustion),
        json.dumps(steps), json.dumps(ifn_gamma),
        json.dumps([activation[0]]), json.dumps([exhaustion[0]]), json.dumps([ifn_gamma[0]])
    )


def _generate_dynamic_tme_svg(state: Dict[str, Any]) -> str:
    """Generate animated TME SVG with CSS animations.

    Features:
    - Pulsating tumor mass
    - Moving immune cells
    - Dynamic vascular flow
    - Phase indicator animation
    """
    import math

    TN = {
        "bg": "#1a1b26", "surface": "#24283b", "border": "#414d68",
        "text": "#c0caf5", "muted": "#a9b1d6",
        "blue": "#3498db", "green": "#27ae60", "yellow": "#f39c12",
        "red": "#e74c3c", "purple": "#bb9af7", "cyan": "#7dcfff",
        "orange": "#ff9e64", "teal": "#73daca",
    }

    volume = state.get("tum_volume", 50.0)
    phase = state.get("ied_phase", "elimination")
    cd8 = int(state.get("imm_cd8_count", 100))
    nk = int(state.get("imm_nk_count", 50))
    immune_pressure = state.get("ied_immune_pressure", 0.5)

    W, H = 500, 450
    cx, cy = W // 2, H // 2
    tumor_r = max(20, min(100, 20 + 20 * math.log10(max(1, volume))))

    phase_colors = {"elimination": TN["green"], "equilibrium": TN["yellow"], "escape": TN["red"]}
    phase_color = phase_colors.get(phase, TN["muted"])

    # CSS animations
    css = """
    <style>
    @keyframes pulse {{ 0%, 100% {{ opacity: 0.8; }} 50% {{ opacity: 1.0; }} }}
    @keyframes moveCD8 {{ 0% {{ transform: translate(0, 0); }} 50% {{ transform: translate(5px, 3px); }} 100% {{ transform: translate(0, 0); }} }}
    @keyframes moveNK {{ 0% {{ transform: translate(0, 0); }} 50% {{ transform: translate(-3px, 5px); }} 100% {{ transform: translate(0, 0); }} }}
    @keyframes flow {{ 0% {{ stroke-dashoffset: 20; }} 100% {{ stroke-dashoffset: 0; }} }}
    @keyframes phaseGlow {{ 0%, 100% {{ fill-opacity: 0.5; }} 50% {{ fill-opacity: 1.0; }} }}
    .tumor-pulse {{ animation: pulse 2s ease-in-out infinite; }}
    .cd8-move {{ animation: moveCD8 1.5s ease-in-out infinite; }}
    .nk-move {{ animation: moveNK 1.8s ease-in-out infinite; }}
    .vessel-flow {{ animation: flow 1s linear infinite; }}
    .phase-indicator {{ animation: phaseGlow 1s ease-in-out infinite; }}
    </style>
    """

    elements = []
    elements.append('<rect width="{}" height="{}" fill="{}" rx="12"/>'.format(W, H, TN["bg"]))

    # Phase indicator (animated)
    elements.append('<circle cx="{}" cy="{}" r="8" fill="{}" class="phase-indicator"/>'.format(
        cx, 30, phase_color))
    elements.append('<text x="{}" y="55" text-anchor="middle" fill="{}" font-size="12" font-weight="bold">{}</text>'.format(
        cx, phase_color, phase.upper()))

    # Tumor mass (pulsating)
    elements.append('<circle cx="{}" cy="{}" r="{}" fill="{}" class="tumor-pulse"/>'.format(
        cx, cy, tumor_r, TN["purple"]))
    elements.append('<text x="{}" y="{}" text-anchor="middle" fill="{}" font-size="11">Tumor</text>'.format(
        cx, cy + 4, TN["text"]))
    elements.append('<text x="{}" y="{}" text-anchor="middle" fill="{}" font-size="9">{:.0f} mm³</text>'.format(
        cx, cy + 18, TN["muted"], volume))

    # Vessels with flowing animation
    vessel_count = 6
    for i in range(vessel_count):
        angle = i * 360 / vessel_count
        rad = math.radians(angle)
        x1 = cx + (tumor_r + 60) * math.cos(rad)
        y1 = cy + (tumor_r + 60) * math.sin(rad)
        x2 = cx + (tumor_r + 10) * math.cos(rad)
        y2 = cy + (tumor_r + 10) * math.sin(rad)
        elements.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="3" stroke-dasharray="5,5" class="vessel-flow"/>'.format(
            x1, y1, x2, y2, TN["red"]))

    # Animated immune cells
    import random
    random.seed(42)

    # CD8+ T cells (moving toward tumor)
    n_cd8 = min(cd8 // 30, 10)
    for i in range(n_cd8):
        angle = random.uniform(0, 360)
        r = random.uniform(tumor_r + 15, tumor_r + 50)
        rad = math.radians(angle)
        x = cx + r * math.cos(rad)
        y = cy + r * math.sin(rad)
        elements.append('<circle cx="{:.1f}" cy="{:.1f}" r="5" fill="{}" class="cd8-move"/>'.format(
            x, y, TN["green"]))

    # NK cells
    n_nk = min(nk // 20, 8)
    for i in range(n_nk):
        angle = random.uniform(0, 360)
        r = random.uniform(tumor_r + 20, tumor_r + 55)
        rad = math.radians(angle)
        x = cx + r * math.cos(rad)
        y = cy + r * math.sin(rad)
        elements.append('<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="{}" class="nk-move"/>'.format(
            x, y, TN["cyan"]))

    # Pressure arrows
    arrow_r = tumor_r + 80
    # Immune pressure (inward)
    elements.append('<path d="M {:.1f} {:.1f} L {:.1f} {:.1f} L {:.1f} {:.1f}" fill="{}" opacity="0.7"/>'.format(
        cx + arrow_r, cy - 10,
        cx + arrow_r - 20, cy - 10,
        cx + arrow_r - 15, cy - 20,
        TN["green"]))
    elements.append('<text x="{}" y="{}" fill="{}" font-size="9">Immune Pressure</text>'.format(
        cx + arrow_r - 70, cy - 25, TN["green"]))

    # Legend
    legend_y = H - 50
    elements.append('<rect x="{}" y="{}" width="200" height="40" fill="{}" rx="6"/>'.format(
        W // 2 - 100, legend_y, TN["surface"]))
    elements.append('<circle cx="{}" cy="{}" r="5" fill="{}"/>'.format(W//2 - 70, legend_y + 15, TN["green"]))
    elements.append('<text x="{}" y="{}" fill="{}" font-size="9">CD8: {}</text>'.format(W//2 - 60, legend_y + 18, TN["muted"], cd8))
    elements.append('<circle cx="{}" cy="{}" r="4" fill="{}"/>'.format(W//2 + 20, legend_y + 15, TN["cyan"]))
    elements.append('<text x="{}" y="{}" fill="{}" font-size="9">NK: {}</text>'.format(W//2 + 30, legend_y + 18, TN["muted"], nk))

    return css + '<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">{}</svg>'.format(
        W, H, ''.join(elements))


def _generate_animated_timeline_slider(history: list) -> str:
    """Generate interactive timeline slider with event markers."""
    if not history:
        return ""

    n_steps = len(history)

    # Detect key events
    events = []
    volumes = [h.get('tum_volume', 50) for h in history]

    # Detect doubling events
    for i, v in enumerate(volumes):
        if i > 0 and v > volumes[0] * 2 and (i == 1 or volumes[i-1] <= volumes[0] * 2):
            events.append({"step": i, "type": "doubling", "label": "Tumor doubled"})

    progress_pct = 100 * (n_steps - 1) / max(n_steps, 1)

    return """
    <div class="card">
        <div class="card-header">Simulation Timeline</div>
        <div class="card-body">
            <div style="position:relative;height:80px;background:rgba(0,0,0,0.3);border-radius:8px;padding:10px;">
                <div style="position:absolute;top:50%;left:10px;right:10px;height:4px;background:#30363d;border-radius:2px;">
                    <div style="position:absolute;left:0;top:0;height:100%;width:{:.1f}%;background:linear-gradient(90deg,#c41e3a,#e74c3c);border-radius:2px;transition:width 0.3s;"></div>
                </div>
                <div style="position:absolute;top:30%;left:{:.1f}%;transform:translateX(-50%);">
                    <div style="width:12px;height:12px;background:#c41e3a;border-radius:50%;border:2px solid #ecf0f1;cursor:pointer;" title="Current: Step {}"></div>
                </div>
                {}
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:10px;color:#7f8c8d;font-size:0.85em;">
                <span>Step 0</span>
                <span>Step {}</span>
            </div>
        </div>
    </div>
    """.format(
        progress_pct,
        progress_pct,
        n_steps - 1,
        "".join([
            '<div style="position:absolute;top:60%;left:{:.1f}%;transform:translateX(-50%);">'
            '<div style="width:8px;height:8px;background:#f39c12;border-radius:50%;"></div>'
            '<div style="font-size:8px;color:#f39c12;white-space:nowrap;margin-top:2px;">{}</div>'
            '</div>'.format(100 * e["step"] / max(n_steps, 1), e["label"])
            for e in events
        ]),
        n_steps - 1
    )
