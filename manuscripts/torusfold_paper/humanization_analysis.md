# TorusFold Humanization Analysis

## Draft Analysis - AI Patterns Found

Based on the humanizer skill guidelines, here are the AI writing patterns detected in TorusFold:

### Critical Issues to Fix:

1. **Em dashes (Pattern #14)** - Hard constraint to eliminate ALL em dashes
   - Multiple instances of `---` throughout
   - Must replace with: commas, periods, colons, or parentheses

2. **Overused "AI Vocabulary" (Pattern #7)**
   - "Key" (appears ~15+ times)
   - "Crucial" 
   - "Testament" in overclaiming statements
   - "Highlighting/underscoring/emphasizing"

3. **Undue Significance (Pattern #1)**
   - "marking a pivotal moment"
   - "stands as a testament"
   - "underscores the importance"
   - Last paragraph: "TorusFold provides a starting point for circRNA structure prediction where none existed before" - overclaiming

4. **Superficial -ing Phrases (Pattern #3)**
   - "highlighting the importance"
   - "ensuring that"
   - "reflecting the"

5. **Rule of Three Overuse (Pattern #10)**
   - "equivariance, diffusion, iterative refinement, and attention mechanisms" (4 items)
   - "high-quality data first, then confidence-weighted mixed data, then long sequences"

6. **Boldface Overuse (Pattern #15)**
   - Excessive use of `\textbf{}` for emphasis
   - Bold headers followed by colons (Pattern #16)

7. **Copula Avoidance (Pattern #8)**
   - "serves as" "stands as" instead of simple "is"

8. **Generic Positive Conclusions (Pattern #25)**
   - Abstract closing: "TorusFold provides a starting point for circRNA structure prediction where none existed before"
   - Should be specific, not promotional

### What's Already Good:
- No curly quotes (uses straight quotes)
- No emojis in academic text
- Title case is appropriate for academic paper
- No "Great question!" chatbot artifacts
- No knowledge cutoff disclaimers
- No promotional language like "vibrant tapestry"

## Humanization Plan:

### 1. Abstract Rewrite

**Before (AI-sounding):**
> "TorusFold provides a starting point for circRNA structure prediction where none existed before."

**After (Human):**
> "TorusFold implements eight architectures for circRNA 3D structure prediction. On a test set of N=7 sequences (lengths 20-27 nt), the best scheme achieved 13.91Å RMSD with 0.02Å closure error. We introduce Torus Positional Encoding (TPE) which guarantees circular periodicity, and release benchmark datasets for comparison with existing methods."

### 2. Introduction Simplification

**Before:**
> "We built TorusFold to address this. Our contributions: (1) Torus Positional Encoding (TPE), which guarantees circular periodicity; (2) comparison of eight architectures spanning equivariance, diffusion, iterative refinement, and attention mechanisms..."

**After:**
> "TorusFold compares eight architectures for this task. TPE guarantees circular periodicity. We tested equivariant networks, diffusion models, and attention mechanisms on N=7 sequences. We also define evaluation metrics specific to circRNA (closure error, circular distance) and release benchmarks."

### 3. Remove Em Dashes (Critical)

Replace all `---` with:
- Commas for tight asides
- Periods for separate sentences
- Colons for explanations
- Parentheses for true asides

### 4. Simplify Boldface Usage

Keep bold only for:
- Section headers
- Truly necessary emphasis (rare)

Remove bold from:
- Inline text like "Design Philosophy"
- List items that don't need emphasis

### 5. Replace "Key" Overuse

Instead of "key role", "key insight", "key contribution":
- Use specific descriptions
- Or just remove the puffery

### 6. Fix Rule of Three

Reduce 3-4 item lists to natural variety:
- Sometimes 2 items
- Sometimes 1 item
- Sometimes 5 items (varied)

### 7. Remove Significance Puffery

Cut phrases like:
- "marking a pivotal moment"
- "stands as a testament"
- "underscores the importance"
- "reflects broader trends"

Replace with plain facts.

## Next Steps:

1. Apply all fixes systematically
2. Read aloud to check rhythm
3. Ensure varied sentence length
4. Keep technical accuracy
5. Preserve academic tone (don't over-casualize)
6. Final em dash scan (zero tolerance)