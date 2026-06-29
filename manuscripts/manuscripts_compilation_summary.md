# Manuscripts Compilation Summary
**Date:** 2026-06-27
**Status:** ✅ Both papers compiled successfully

---

## 📄 Generated Documents

### 1. TorusFold Paper (Nature Methods)

| Format | File | Size | Location |
|--------|------|------|----------|
| **LaTeX** | torusfold.tex | ~20KB | manuscripts/torusfold_paper/ |
| **Word** | torusfold_manuscript.docx | 26KB | manuscripts/torusfold_paper/ |
| **Markdown** | torusfold_for_word.md | 40KB | manuscripts/torusfold_paper/ |

**Key Features:**
- 8 architectural schemes with detailed descriptions
- Design rationale, Strengths, Limitations for each scheme
- Methods section with complete implementation details
- TBD placeholders cleaned (0 TBD in main manuscript)
- 3 external baseline experiments planned
- Supplementary experiments plan included

---

### 2. Confluencia 3.0 Paper (Bioinformatics)

| Format | File | Size | Location |
|--------|------|------|----------|
| **LaTeX** | confluencia_3.0_research_paper.tex | ~40KB | manuscripts/ |
| **Word** | confluencia_3.0_manuscript.docx | 328KB | manuscripts/ |
| **Markdown** | confluencia_3.0_plain.md | 37KB | manuscripts/ |

**Key Features:**
- EventBus-first architecture with 34+ event types
- 6 subsystems (Tumor, TME, Treatment, CircRNA, Biomarker, Clinical)
- 3 figures properly referenced:
  - fig1_system_architecture.png (123KB)
  - fig2_torusfold_flow.png (109KB)
  - fig6_validation.png (101KB)
- 2 tables with experimental data
- BibTeX references (7 citations)
- Wet-lab validation ongoing section

---

## 🔬 Compilation Details

### TorusFold Compilation Process
```bash
cd D:/IGEM集成方案/manuscripts/torusfold_paper
python: pylatexenc.latex2text conversion
pandoc: torusfold_for_word.md → torusfold_manuscript.docx
Result: 26KB Word document (26,297 bytes)
```

### Confluencia 3.0 Compilation Process
```bash
cd D:/IGEM集成方案/manuscripts
pandoc: confluencia_3.0_research_paper.tex → confluencia_3.0_manuscript.docx
Result: 328KB Word document (335,038 bytes)
```

**Note:** Confluencia DOCX is larger (328KB vs 26KB) because:
- Contains embedded figures (PNG images ~330KB total)
- More content (37KB text vs 40KB)
- BibTeX references included

---

## 🖼️ Figures Status

### TorusFold Paper
- Figures located in: manuscripts/torusfold_paper/figures_png/
- Currently referenced but not embedded in DOCX
- 8 PNG files available for manual insertion

### Confluencia 3.0 Paper
- Figures located in: confluencia_3_0/docs/paper/figures/
- **✅ Properly embedded in DOCX** (pandoc auto-embeds)
- 3 PNG files: fig1, fig2, fig6

---

## 📊 Content Verification

### TorusFold Key Sections Present:
✅ Abstract (N=7 correctly stated)
✅ Introduction
✅ Results (8 schemes, design rationales)
✅ Methods (architectural details)
✅ Discussion
✅ Limitations
✅ Supplementary experiments plan

### Confluencia 3.0 Key Sections Present:
✅ Abstract
✅ Introduction (4 gaps, 4 innovations)
✅ Methods (EventBus, 6 modules)
✅ Results (PK, immunogenicity, subtype comparison)
✅ Discussion (4 contributions)
✅ Limitations
✅ Data Availability
✅ Code Availability

---

## 📝 References Status

### TorusFold
- references.bib created
- Need manual insertion in Word version

### Confluencia 3.0
- references.bib created (7 citations)
- Pandoc auto-included from .bib file

---

## 🎯 Submission Readiness

### TorusFold (Nature Methods)
| Item | Status | Notes |
|------|--------|-------|
| Main manuscript | ✅ Ready | DOCX available |
| Supplementary plan | ✅ Ready | critical_experiments_plan.md |
| Figures | ⏳ Manual insertion | 8 PNGs available |
| References | ⏳ Manual insertion | BibTeX created |
| TBD placeholders | ✅ 0 remaining | All cleaned |

### Confluencia 3.0 (Bioinformatics)
| Item | Status | Notes |
|------|--------|-------|
| Main manuscript | ✅ Ready | DOCX with figures |
| Figures | ✅ Embedded | 3 PNGs auto-embedded |
| References | ✅ Included | BibTeX auto-included |
| Tables | ✅ Included | 2 tables |
| Wet-lab validation | ⏳ Ongoing | Section included |

---

## 📂 File Locations Summary

```
D:/IGEM集成方案/manuscripts/
├── torusfold_paper/
│   ├── torusfold.tex (LaTeX source)
│   ├── torusfold_manuscript.docx (Word output, 26KB)
│   ├── torusfold_for_word.md (Markdown intermediate)
│   ├── references.bib
│   ├── critical_experiments_plan.md
│   ├── supplementary_experiments_plan.md
│   └── figures_png/ (8 PNG files)
│
├── confluencia_3.0_research_paper.tex (LaTeX source)
├── confluencia_3.0_manuscript.docx (Word output, 328KB)
├── confluencia_3.0_plain.md (Markdown intermediate)
├── references.bib (7 citations)
│
└── confluencia_3_0/docs/paper/figures/
    ├── fig1_system_architecture.png
    ├── fig2_torusfold_flow.png
    └── fig6_validation.png
```

---

## 🎉 Compilation Success

Both manuscripts compiled successfully:
- **TorusFold:** 26KB Word document (pure text, figures need manual insertion)
- **Confluencia 3.0:** 328KB Word document (figures auto-embedded)

Ready for submission after:
1. Manual figure insertion for TorusFold
2. Reference formatting verification
3. Author information addition
4. Conflict of interest statement

