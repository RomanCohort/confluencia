#!/usr/bin/env python3
import os
import re
import sys
import json

workspace = r"d:\IGEM集成方案"
readme_path = os.path.join(workspace, "readme", "TOTALREADME.md")
out_checked = os.path.join(workspace, "readme", "TOTALREADME_katex_checked.md")

if not os.path.exists(readme_path):
    print(json.dumps({"error": "not_found", "path": readme_path}, ensure_ascii=False))
    sys.exit(0)


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

text = read_text(readme_path)
has_replacement_char = "\ufffd" in text

# Mask fenced code blocks
code_blocks = []

def mask_codeblocks(s):
    def repl(m):
        code_blocks.append(m.group(0))
        return f"__CODEBLOCK_PLACEHOLDER_{len(code_blocks)-1}__"
    return re.sub(r"```.*?```", repl, s, flags=re.DOTALL)

# Mask inline code
inline_codes = []

def mask_inline_code(s):
    def repl(m):
        inline_codes.append(m.group(0))
        return f"__INLINECODE_PLACEHOLDER_{len(inline_codes)-1}__"
    return re.sub(r"`[^`\n]+`", repl, s)

# Mask $$...$$ blocks
dollar_blocks = []

def mask_dollar_blocks(s):
    def repl(m):
        dollar_blocks.append(m.group(0))
        return f"__DOLLARBLOCK_PLACEHOLDER_{len(dollar_blocks)-1}__"
    return re.sub(r"\$\$.*?\$\$", repl, s, flags=re.DOTALL)

masked = mask_codeblocks(text)
masked = mask_inline_code(masked)
masked = mask_dollar_blocks(masked)

# Replace single-dollar inline math (no newlines inside)
single_inline_pat = re.compile(r"(?<!\$)\$(?!\$)([^\n\$]+?)(?<!\$)\$(?!\$)")


def replace_inline(m):
    inner = m.group(1)
    return "\\(" + inner + "\\)"

new_masked, n_inline_repl = single_inline_pat.subn(replace_inline, masked)

# Restore masked blocks
for i, b in enumerate(dollar_blocks):
    new_masked = new_masked.replace(f"__DOLLARBLOCK_PLACEHOLDER_{i}__", b)
for i, b in enumerate(inline_codes):
    new_masked = new_masked.replace(f"__INLINECODE_PLACEHOLDER_{i}__", b)
for i, b in enumerate(code_blocks):
    new_masked = new_masked.replace(f"__CODEBLOCK_PLACEHOLDER_{i}__", b)

# Canonicalize backtick file paths where possible
backtick_pat = re.compile(r"`([^`]+)`")
all_backticks = backtick_pat.findall(new_masked)
unique_backticks = sorted(set(all_backticks))
backticks_replaced = 0
replacements = {}
ext_allow = {'.py', '.md', '.ps1', '.txt', '.csv', '.pt', '.h5', '.npy', '.spec', '.ipynb', '.json'}

for bt in unique_backticks:
    bt_strip = bt.strip()
    if "\n" in bt_strip or len(bt_strip) > 260:
        continue
    if "*" in bt_strip or "http://" in bt_strip.lower() or "https://" in bt_strip.lower():
        continue
    if ("/" in bt_strip or "\\" in bt_strip) and os.path.splitext(bt_strip)[1].lower() in ext_allow:
        normalized = bt_strip.replace("/", os.sep).replace("\\", os.sep)
        candidate = os.path.join(workspace, normalized)
        if os.path.exists(candidate):
            rel = os.path.relpath(candidate, workspace).replace(os.sep, "/")
            replacements["`" + bt + "`"] = f"[{bt}]({rel})"
            backticks_replaced += 1
            continue
        base = os.path.basename(normalized)
        matches = []
        for root, dirs, files in os.walk(workspace):
            if base in files:
                matches.append(os.path.join(root, base))
                if len(matches) > 1:
                    break
        if len(matches) == 1:
            rel = os.path.relpath(matches[0], workspace).replace(os.sep, "/")
            replacements["`" + bt + "`"] = f"[{bt}]({rel})"
            backticks_replaced += 1

for k, v in replacements.items():
    new_masked = new_masked.replace(k, v)

# Count remaining single-dollar math pairs
remaining_matches = re.findall(r"(?<!\$)\$(?!\$)([^\n\$]+?)(?<!\$)\$(?!\$)", new_masked)
remaining_pairs = len(remaining_matches)
remaining_snippets = remaining_matches[:10]

# Write checked file
with open(out_checked, "w", encoding="utf-8") as f:
    f.write(new_masked)

# Overwrite original if safe
overwrote = False
if remaining_pairs == 0:
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_masked)
    overwrote = True

print(json.dumps({
    "target": readme_path,
    "out_checked": out_checked,
    "inline_replacements": n_inline_repl,
    "backtick_replacements": backticks_replaced,
    "remaining_single_pairs": remaining_pairs,
    "remaining_snippets": remaining_snippets,
    "has_replacement_char": has_replacement_char,
    "overwrote": overwrote
}, ensure_ascii=False))
