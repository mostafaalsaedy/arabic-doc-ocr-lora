"""TEDS (Tree-Edit-Distance-based Similarity) for HTML tables — pure stdlib.

Baseer reports TEDS 66 on misraj_dococr; Re has never measured it (CER/WER are
blind to structure — a table with correct text but wrong cell boundaries can score
well on CER and terribly on TEDS). This implements the PubTabNet/TEDS definition:

    TEDS(a, b) = 1 - EditDistance(T_a, T_b) / max(|T_a|, |T_b|)

with the standard cost model:
    insert / delete node        = 1
    rename                      = 1 if tag or colspan/rowspan differ
                                = normalized Levenshtein(content) for <td>
                                = 0 otherwise

Implemented with Zhang-Shasha rather than APTED (no `apted` package available and
`uv pip install` has previously broken torch in this venv — see memory). Same
metric value; table trees are depth-3 so the complexity term
O(|T1||T2|·min(d,l)^2) stays small.

Deps: stdlib only (html.parser). Safe to import anywhere.
"""
from html.parser import HTMLParser


# ---------------------------------------------------------------- tree building

class _Node:
    __slots__ = ("tag", "colspan", "rowspan", "content", "children")

    def __init__(self, tag, colspan=1, rowspan=1):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = ""
        self.children = []


class _TableParser(HTMLParser):
    """Collect every <table> in a document as a _Node tree."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in ("table", "tr", "td", "th", "thead", "tbody"):
            return
        if tag in ("thead", "tbody"):          # structural noise — flatten away
            return
        d = dict(attrs)
        def _int(v, dflt=1):
            try:
                return int(v)
            except (TypeError, ValueError):
                return dflt
        node = _Node("td" if tag == "th" else tag,
                     _int(d.get("colspan")), _int(d.get("rowspan")))
        if tag == "table":
            self.tables.append(node)
        elif self._stack:
            self._stack[-1].children.append(node)
        else:
            return                              # stray tr/td with no table
        self._stack.append(node)

    def handle_endtag(self, tag):
        if tag in ("thead", "tbody") or tag not in ("table", "tr", "td", "th"):
            return
        want = "td" if tag == "th" else tag
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i].tag == want:
                del self._stack[i:]
                break

    def handle_data(self, data):
        if self._stack and self._stack[-1].tag == "td":
            self._stack[-1].content += data


def parse_tables(html):
    p = _TableParser()
    try:
        p.feed(html or "")
        p.close()
    except Exception:
        pass
    return p.tables


# ------------------------------------------------------------------- distances

def _lev_norm(a, b):
    """Levenshtein normalized to [0,1] by the longer string."""
    a, b = (a or "").split(), (b or "").split()
    a, b = " ".join(a), " ".join(b)
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _rename_cost(n1, n2):
    if n1.tag != n2.tag or n1.colspan != n2.colspan or n1.rowspan != n2.rowspan:
        return 1.0
    if n1.tag == "td" and (n1.content or n2.content):
        return _lev_norm(n1.content, n2.content)
    return 0.0


def _postorder(root):
    """Iterative post-order; returns (nodes, leftmost-leaf index per node)."""
    nodes, lml, stack = [], [], [(root, False)]
    idx = {}
    while stack:
        node, visited = stack.pop()
        if visited:
            idx[id(node)] = len(nodes)
            # leftmost-LEAF descendant, i.e. lml(first child) — NOT the first
            # child's own index. Those coincide only when the first child is a
            # leaf (depth-2 trees), which is why small hand-built test cases hide
            # the difference; on a real 223-node table it yields a negative index.
            if node.children:
                lml.append(lml[idx[id(node.children[0])]])
            else:
                lml.append(len(nodes))
            nodes.append(node)
        else:
            stack.append((node, True))
            for c in reversed(node.children):
                stack.append((c, False))
    return nodes, lml


def _tree_edit_distance(r1, r2):
    """Zhang-Shasha with the TEDS cost model."""
    A, lmlA = _postorder(r1)
    B, lmlB = _postorder(r2)
    n, m = len(A), len(B)

    def keyroots(lml):
        seen, ks = set(), []
        for i in range(len(lml) - 1, -1, -1):
            if lml[i] not in seen:
                seen.add(lml[i])
                ks.append(i)
        return sorted(ks)

    kA, kB = keyroots(lmlA), keyroots(lmlB)
    TD = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in kA:
        for j in kB:
            li, lj = lmlA[i], lmlB[j]
            w, h = i - li + 2, j - lj + 2
            fd = [[0.0] * h for _ in range(w)]
            for x in range(1, w):
                fd[x][0] = fd[x - 1][0] + 1.0                      # delete
            for y in range(1, h):
                fd[0][y] = fd[0][y - 1] + 1.0                      # insert
            for x in range(1, w):
                for y in range(1, h):
                    ai, bj = li + x - 1, lj + y - 1
                    if lmlA[ai] == li and lmlB[bj] == lj:
                        fd[x][y] = min(fd[x - 1][y] + 1.0,
                                       fd[x][y - 1] + 1.0,
                                       fd[x - 1][y - 1] + _rename_cost(A[ai], B[bj]))
                        TD[ai + 1][bj + 1] = fd[x][y]
                    else:
                        px, py = lmlA[ai] - li, lmlB[bj] - lj
                        fd[x][y] = min(fd[x - 1][y] + 1.0,
                                       fd[x][y - 1] + 1.0,
                                       fd[px][py] + TD[ai + 1][bj + 1])
    return TD[n][m]


# ------------------------------------------------------------------ public API

def teds_score(pred_html, gt_html):
    """TEDS in [0,1] over the tables found in each document.

    Documents are compared table-by-table in order; a missing counterpart scores 0.
    Returns None when NEITHER side contains a table (page is not table-bearing, so
    it should be excluded rather than counted as a perfect or zero score).
    """
    tp, tg = parse_tables(pred_html), parse_tables(gt_html)
    if not tp and not tg:
        return None
    if not tp or not tg:
        return 0.0
    scores = []
    for k in range(max(len(tp), len(tg))):
        if k >= len(tp) or k >= len(tg):
            scores.append(0.0)
            continue
        a, b = tp[k], tg[k]
        na = len(_postorder(a)[0])
        nb = len(_postorder(b)[0])
        d = _tree_edit_distance(a, b)
        scores.append(max(0.0, 1.0 - d / max(na, nb)))
    return sum(scores) / len(scores)
