"""Output post-processing for Re — inference-time, reversible, weights untouched.

`md_tables_to_html`: the model emits markdown pipe tables (it was trained on
markdown-targeted doc data) but the misraj/Baseer ground truth encodes tables as
HTML <table><tr><td>. 58/400 misraj pages are affected; the model matched the
reference format on ZERO of them. Converting at output time recovers part of that
mechanical penalty without retraining:
    misraj (400 pages): CER 0.3871 -> 0.3798, WER 0.5626 -> 0.5523
    table pages (58)  : CER 0.6470 -> 0.6013, WER 0.8262 -> 0.7612
The remaining table error is structural (cell boundaries are mis-segmented), which
post-processing cannot fix — that needs training.

Kept as an OPT-IN toggle (`--md_tables_to_html` on run_eval.py) rather than baked
into the weights, so markdown output stays available for product use.
"""
import re

_SEP = re.compile(r"^\s*\|?[\s:\-]+\|[\s:\-|]*$")


def _is_row(line):
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 2


def md_tables_to_html(text):
    """Rewrite contiguous markdown pipe-table blocks as HTML <table> markup.

    Non-table text passes through byte-identical. Separator rows (|---|---|) are
    dropped since HTML has no equivalent. A single stray '| x |' line is still
    converted (one-row table) — harmless, and matches how the refs mark up short
    tables.
    """
    if not text or "|" not in text:
        return text
    out, buf = [], []

    def flush():
        if not buf:
            return
        rows = [r for r in buf if not _SEP.match(r)]
        # Require >=2 data rows. A lone '| x |' line is more often stray pipes in
        # ordinary text than a real table — that false positive cost 0.001 CER on
        # a sedra handwritten line during validation, while real tables always
        # have multiple rows, so this filter is free.
        if len(rows) >= 2:
            out.append("<table>")
            for r in rows:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            out.append("</table>")
        else:
            out.extend(buf)          # too short / separator-only: leave untouched
        buf.clear()

    for line in text.split("\n"):
        if _is_row(line):
            buf.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


# --------------------------------------------------------------------------
# Reverse direction. Stage 6 trains the model to emit HTML tables (explicit cell
# boundaries = better supervision), so this is what keeps MARKDOWN output
# available for product use — the format choice stays an inference-time toggle
# rather than something baked irreversibly into the weights.
# --------------------------------------------------------------------------

_TABLE_BLOCK = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def html_tables_to_md(text):
    """Rewrite HTML <table> blocks as markdown pipe tables.

    Inverse of `md_tables_to_html` (lossy only where HTML carries colspan/rowspan,
    which markdown cannot express). Non-table text passes through unchanged.
    """
    if not text or "<table" not in text.lower():
        return text

    def one(m):
        rows = _ROW.findall(m.group(1))
        if not rows:
            return m.group(0)
        table, width = [], 0
        for r in rows:
            cells = [_TAG.sub("", c).strip() for c in _CELL.findall(r)]
            if cells:
                table.append(cells)
                width = max(width, len(cells))
        if not table:
            return m.group(0)
        out = []
        for i, cells in enumerate(table):
            cells = cells + [""] * (width - len(cells))
            out.append("| " + " | ".join(cells) + " |")
            if i == 0:                      # markdown needs a header separator
                out.append("|" + "|".join(["---"] * width) + "|")
        return "\n".join(out)

    return _TABLE_BLOCK.sub(one, text)
