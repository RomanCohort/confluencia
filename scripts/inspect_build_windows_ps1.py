from __future__ import annotations

import pathlib

TARGET = pathlib.Path(r"D:\IGEM集成方案\新建文件夹\build_windows.ps1")


def main() -> None:
    data = TARGET.read_bytes()
    lines = data.splitlines()
    print("path:", TARGET)
    print("byte_len:", len(data))
    print("bom/head:", data[:16].hex(" "))
    nul_count = data.count(b"\x00")
    print("nul_count:", nul_count)
    print("line_count:", len(lines))

    text_all = data.decode("utf-8", "replace")
    block_comment_starts = text_all.count("<#")
    block_comment_ends = text_all.count("#>")
    print("block_comment_starts(<#):", block_comment_starts)
    print("block_comment_ends(#>):", block_comment_ends)

    for ln in [211, 213, 221, 222, 223, 224, 225, 226, 227, 228, 229, 524, 527, 531]:
        if not (1 <= ln <= len(lines)):
            print(f"\nLINE {ln}: <out of range>")
            continue
        b = lines[ln - 1]
        s = b.decode("utf-8", "replace")
        dq_positions = [i for i, ch in enumerate(s) if ch == '"']
        dq_byte_positions = [i for i, bv in enumerate(b) if bv == 0x22]
        print(f"\nLINE {ln}: bytes_len={len(b)}")
        print("text:", s)
        print("dq_count:", len(dq_positions), "dq_positions:", dq_positions[:20])
        print("dq_byte_count:", len(dq_byte_positions), "dq_byte_positions:", dq_byte_positions[:20])
        if dq_byte_positions:
            for p in dq_byte_positions[:4]:
                lo = max(0, p - 10)
                hi = min(len(b), p + 11)
                snippet = b[lo:hi]
                print(f"  around_quote_byte@{p}:", snippet.hex(" "))
        if len(dq_positions) > 20:
            print("... (truncated)")

    print("\n--- tail raw dump (lines 520-532) ---")
    for ln in range(520, min(532, len(lines)) + 1):
        b = lines[ln - 1]
        s = b.decode("utf-8", "replace")
        print(f"{ln:>4}: {s}")
        print("      bytes:", b.hex(" "))

    # Heuristic scan: track quote/here-string/braces state to find the first line
    # where parsing likely goes off the rails.
    in_sq = False
    in_dq = False
    in_here_sq = False
    in_here_dq = False
    in_block_comment = False
    brace_balance = 0
    first_suspect: int | None = None

    def is_here_terminator(line: str, kind: str) -> bool:
        # Here-string terminator must be alone on the line (whitespace allowed).
        s = line.strip()
        return s == ("'@" if kind == "sq" else '"@')

    def starts_here(line: str, kind: str) -> bool:
        # Here-string start token @' or @" cannot have leading non-whitespace.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return False
        return stripped.startswith("@'" if kind == "sq" else '@"')

    for i, raw in enumerate(lines, start=1):
        line = raw.decode("utf-8", "replace")

        # Block comment handling (<# ... #>) outside of strings.
        # If we're in a block comment, ignore everything until #>.
        if in_block_comment:
            if "#>" in line:
                _, _, after = line.partition("#>")
                line = after
                in_block_comment = False
            else:
                continue

        # Strip inline block comments best-effort (not nested-aware).
        while "<#" in line and not (in_sq or in_dq):
            pre, _, rest = line.partition("<#")
            if "#>" in rest:
                _, _, post = rest.partition("#>")
                line = pre + post
            else:
                line = pre
                in_block_comment = True
                break

        if in_here_sq:
            if is_here_terminator(line, "sq"):
                in_here_sq = False
            continue
        if in_here_dq:
            if is_here_terminator(line, "dq"):
                in_here_dq = False
            continue

        if not (in_sq or in_dq):
            if starts_here(line, "sq"):
                in_here_sq = True
                continue
            if starts_here(line, "dq"):
                in_here_dq = True
                continue

        j = 0
        while j < len(line):
            ch = line[j]

            # Line comment starts outside of strings.
            if not (in_sq or in_dq) and ch == "#":
                break

            # Backtick escaping inside double quotes.
            if in_dq and ch == "`":
                j += 2
                continue

            if not in_dq and ch == "'":
                if in_sq:
                    # In single-quoted string, '' escapes a single quote.
                    if j + 1 < len(line) and line[j + 1] == "'":
                        j += 2
                        continue
                    in_sq = False
                else:
                    in_sq = True
                j += 1
                continue

            if not in_sq and ch == '"':
                in_dq = not in_dq
                j += 1
                continue

            if not (in_sq or in_dq):
                if ch == "{":
                    brace_balance += 1
                elif ch == "}":
                    brace_balance -= 1
                    if brace_balance < 0 and first_suspect is None:
                        first_suspect = i

            j += 1

        # If a line ends with in_dq/in_sq True, it's a strong signal.
        if (in_sq or in_dq) and first_suspect is None:
            first_suspect = i

    print("\n--- heuristic scan ---")
    print(
        "end_state:",
        {
            "in_sq": in_sq,
            "in_dq": in_dq,
            "in_here_sq": in_here_sq,
            "in_here_dq": in_here_dq,
            "brace_balance": brace_balance,
        },
    )
    print("first_suspect_line:", first_suspect)

    non_ascii = [i for i, raw in enumerate(lines, start=1) if any(bv >= 0x80 for bv in raw)]
    print("\nnon_ascii_line_count:", len(non_ascii))
    if non_ascii:
        print("first_non_ascii_lines:", non_ascii[:30])


if __name__ == "__main__":
    main()
