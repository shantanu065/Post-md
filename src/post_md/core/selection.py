"""Atom-selection mini-DSL.

Grammar::

    expr   := orexp
    orexp  := andexp ('or' andexp)*
    andexp := notexp ('and' notexp)*
    notexp := 'not' notexp | atom
    atom   := '(' expr ')' | keyword
    keyword:= 'protein' | 'backbone' | 'all'
            | 'name'    <list>
            | 'resname' <list>
            | 'resid'   <range-list>
            | 'index'   <range-list>

Geometric selections ("around 5.0 X") are intentionally out of scope for v1.
"""

from __future__ import annotations

import re

import numpy as np

PROTEIN_RESNAMES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "HIE", "HID", "HIP", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL", "ASH", "GLH", "CYM", "CYX", "LYN",
})
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "HA", "H"})


class SelectionError(ValueError):
    pass


_TOKEN_RE = re.compile(r"\s+|\(|\)|[A-Za-z0-9_*+\-]+")


def _tokenize(s: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    for m in _TOKEN_RE.finditer(s):
        if m.start() != pos:
            raise SelectionError(f"Unexpected char at {pos}: {s[pos]!r}")
        tok = m.group()
        pos = m.end()
        if tok.isspace():
            continue
        tokens.append(tok)
    if pos != len(s):
        raise SelectionError(f"Unexpected char at {pos}: {s[pos]!r}")
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.i = 0

    def _peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _consume(self):
        t = self._peek()
        self.i += 1
        return t

    def _expect(self, value: str):
        t = self._peek()
        if t != value:
            raise SelectionError(f"Expected {value!r}, got {t!r}")
        return self._consume()

    def parse(self):
        node = self._expr()
        if self._peek() is not None:
            raise SelectionError(f"Unexpected trailing token {self._peek()!r}")
        return node

    def _expr(self):
        left = self._andexp()
        while (t := self._peek()) is not None and t.lower() == "or":
            self._consume()
            right = self._andexp()
            left = ("or", left, right)
        return left

    def _andexp(self):
        left = self._notexp()
        while (t := self._peek()) is not None and t.lower() == "and":
            self._consume()
            right = self._notexp()
            left = ("and", left, right)
        return left

    def _notexp(self):
        t = self._peek()
        if t is not None and t.lower() == "not":
            self._consume()
            return ("not", self._notexp())
        return self._atom()

    def _atom(self):
        t = self._peek()
        if t is None:
            raise SelectionError("Unexpected end of selection")
        if t == "(":
            self._consume()
            node = self._expr()
            self._expect(")")
            return node
        t = self._consume()
        low = t.lower()
        if low in ("all", "protein", "backbone"):
            return ("keyword", low)
        if low == "name":
            return ("name", self._collect_args())
        if low == "resname":
            return ("resname", self._collect_args())
        if low == "resid":
            return ("resid", self._collect_ranges())
        if low == "index":
            return ("index", self._collect_ranges())
        raise SelectionError(f"Unknown keyword {t!r}")

    def _collect_args(self) -> list[str]:
        items: list[str] = []
        while (t := self._peek()) is not None and t not in ("(", ")") and t.lower() not in ("and", "or", "not"):
            items.append(self._consume())
        if not items:
            raise SelectionError("Expected at least one argument")
        return items

    def _collect_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        while (t := self._peek()) is not None and t not in ("(", ")") and t.lower() not in ("and", "or", "not"):
            it = self._consume()
            if "-" in it and not it.startswith("-"):
                lo_s, hi_s = it.split("-", 1)
                ranges.append((int(lo_s), int(hi_s)))
            else:
                v = int(it)
                ranges.append((v, v))
        if not ranges:
            raise SelectionError("Expected at least one range")
        return ranges


def _evaluate(node, topology) -> np.ndarray:
    n = topology.n_atoms
    op = node[0]
    if op == "or":
        return _evaluate(node[1], topology) | _evaluate(node[2], topology)
    if op == "and":
        return _evaluate(node[1], topology) & _evaluate(node[2], topology)
    if op == "not":
        return ~_evaluate(node[1], topology)
    if op == "keyword":
        kw = node[1]
        if kw == "all":
            return np.ones(n, dtype=bool)
        if kw == "protein":
            return np.isin(topology.residue_names, list(PROTEIN_RESNAMES))
        if kw == "backbone":
            is_p = np.isin(topology.residue_names, list(PROTEIN_RESNAMES))
            is_b = np.isin(topology.atom_names, list(BACKBONE_ATOMS))
            return is_p & is_b
        raise SelectionError(f"Unknown keyword {kw!r}")
    if op == "name":
        return np.isin(topology.atom_names, node[1])
    if op == "resname":
        return np.isin(topology.residue_names, node[1])
    if op == "resid":
        mask = np.zeros(n, dtype=bool)
        for lo, hi in node[1]:
            mask |= (topology.residue_ids >= lo) & (topology.residue_ids <= hi)
        return mask
    if op == "index":
        mask = np.zeros(n, dtype=bool)
        for lo, hi in node[1]:
            mask[lo : hi + 1] = True
        return mask
    raise SelectionError(f"Unknown node {op!r}")


def select(topology, query: str) -> np.ndarray:
    tokens = _tokenize(query)
    if not tokens:
        raise SelectionError("Empty selection")
    ast = _Parser(tokens).parse()
    mask = _evaluate(ast, topology)
    return np.nonzero(mask)[0].astype(np.int64)
