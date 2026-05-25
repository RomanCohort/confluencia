# Figure 1: Architecture Diagram

## SVG/PNG版本需要用绘图软件制作，这里是文本描述版本

```
┌────────────────────────────────────────────────────────────────────┐
│                     Confluencia circRNA Platform                    │
│                         (Version 2.5.0)                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  INPUT: circRNA Sequence (A/U/G/C)                                │
│         + Modification Type (m6A/Psi/5mC)                          │
│         + Gene Expression (optional)                               │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    CORE MODULES (10)                          │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │                                                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │ │
│  │  │   Immune    │  │  Structure  │  │  Kinetics   │          │ │
│  │  │  Sensing    │  │ Prediction  │  │  Analysis   │          │ │
│  │  │             │  │             │  │             │          │ │
│  │  │ • RIG-I     │  │ • ViennaRNA │  │ • Rate      │          │ │
│  │  │ • TLR7/8    │  │ • MFE       │  │ • Barrier   │          │ │
│  │  │ • PKR       │  │ • dsRNA     │  │ • Metastable│          │ │
│  │  │ (0.35/0.25/ │  │ • Hairpin   │  │             │          │ │
│  │  │  0.20/0.20) │  │             │  │             │          │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │ │
│  │                                                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │ │
│  │  │ Cotrans     │  │  Folding    │  │ Modifications│          │ │
│  │  │ Folding     │  │  Pathways   │  │             │          │ │
│  │  │             │  │             │  │ • m6A sites │          │ │
│  │  │ • Intermed. │  │ • Transitions│ │ • IRES      │          │ │
│  │  │ • Kinetic   │  │ • Landscape │  │ • miRNA     │          │ │
│  │  │   Trap      │  │ • Pareto    │  │ • RBP       │          │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │ │
│  │                                                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │ │
│  │  │  Drug       │  │  Clinical   │  │   RNA       │          │ │
│  │  │  Response   │  │ Prediction  │  │  Docking    │          │ │
│  │  │             │  │             │  │             │          │ │
│  │  │ • IPS       │  │ • Survival  │  │ • Binding   │          │ │
│  │  │ • TIDE      │  │ • Biomarker │  │ • Drug      │          │ │
│  │  │ • Treatment │  │ • Adverse   │  │   design    │          │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              EVOLUTIONARY OPTIMIZATION                        │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │                                                              │ │
│  │  Operators:                                                  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │ │
│  │  │ Backbone │ │   IRES   │ │   UTR    │ │ Modification │    │ │
│  │  │  Mutation│ │ Optimize │ │  Shuffle │ │   Selection  │    │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │ │
│  │                                                              │ │
│  │  Multi-Objective:                                            │ │
│  │  ┌──────────────────────────────────────────────────────┐   │ │
│  │  │  Stability │ Translation │ Immune │ Delivery │        │   │ │
│  │  │   (0.35)   │    (0.30)   │ (0.25) │  (0.10)  │        │   │ │
│  │  └──────────────────────────────────────────────────────┘   │ │
│  │                                                              │ │
│  │  Selection: Pareto Front + REINFORCE Policy Learning         │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  OUTPUT:                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │ Optimized   │  │   Score     │  │   Clinical  │               │
│  │  Sequence   │  │   Report    │  │   Report    │               │
│  │             │  │             │  │             │               │
│  │ • Stability │  │ • Radar     │  │ • Survival  │               │
│  │ • Mods      │  │   chart     │  │ • Biomarker │               │
│  │ • Objectives│  │ • Pareto    │  │ • AE risk   │               │
│  │             │  │   front     │  │ • Treatment │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  LITERATURE REFERENCES:                                            │
│  • RIG-I: Schlee et al., Nature 2009 (weight=0.35)               │
│  • PKR:  Nallagatla et al., RNA 2007 (weight=0.20)               │
│  • TLR:  Forsbach et al., J Immunol 2008                         │
│  • m6A: Liu et al., Nature 2022                                   │
│  • IPS: Cristescu et al., Nature Genetics 2018                   │
│  • TIDE: Jiang et al., Nature Medicine 2018                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Figure 1 Caption

**Figure 1. Architecture of Confluencia circRNA platform.** The platform integrates 10 core modules for circRNA vaccine design: immunogenicity scoring (RIG-I, TLR7/8, PKR with literature-backed weights), structure prediction (ViennaRNA), folding kinetics, cotranscriptional folding, pathway analysis, modification prediction (m6A, IRES, miRNA/RBP), drug response (IPS/TIDE), clinical outcome prediction, and RNA-drug docking. Evolutionary optimization employs four operators (backbone mutation, IRES optimization, UTR shuffling, modification selection) with Pareto multi-objective selection for stability, translation potential, immune evasion, and delivery compatibility. Literature weights are derived from quantitative experimental data (Schlee et al., 2009; Nallagatla et al., 2007; Forsbach et al., 2008).

---

## 建议用专业工具制作

推荐使用以下工具制作正式Figure：

| 工具 | 优势 | 链接 |
|------|------|------|
| **draw.io** | 免费、在线、导出SVG | https://app.diagrams.net |
| **BioRender** | 生物专业风格 | https://biorender.com |
| **Inkscape** | 矢量图专业工具 | https://inkscape.org |
| **PowerPoint** | 简单快速 | 转PDF/SVG即可 |

---

## 制作步骤

1. 打开 draw.io
2. 按上面的文本架构绘制
3. 导出为 SVG 或 PNG (300dpi)
4. 在文章中引用