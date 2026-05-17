"""Selection-grammar unit tests."""

from __future__ import annotations

import pytest

from post_md.core.selection import SelectionError, select


def test_select_all(small_topology):
    idx = select(small_topology, "all")
    assert idx.tolist() == [0, 1, 2, 3, 4, 5]


def test_select_protein(small_topology):
    idx = select(small_topology, "protein")
    assert idx.tolist() == [0, 1, 2, 3, 4, 5]


def test_select_backbone(small_topology):
    idx = select(small_topology, "backbone")
    # Every atom in fixture is N/CA/C → all backbone
    assert idx.tolist() == [0, 1, 2, 3, 4, 5]


def test_select_name(small_topology):
    idx = select(small_topology, "name CA")
    assert idx.tolist() == [1, 4]


def test_select_multiple_names(small_topology):
    idx = select(small_topology, "name CA C")
    assert sorted(idx.tolist()) == [1, 2, 4, 5]


def test_select_resname(small_topology):
    idx = select(small_topology, "resname ALA")
    assert idx.tolist() == [0, 1, 2]


def test_select_resid_range(small_topology):
    idx = select(small_topology, "resid 1-1")
    assert idx.tolist() == [0, 1, 2]


def test_select_index_range(small_topology):
    idx = select(small_topology, "index 0-2")
    assert idx.tolist() == [0, 1, 2]


def test_select_boolean_and(small_topology):
    idx = select(small_topology, "name CA and resname GLY")
    assert idx.tolist() == [4]


def test_select_boolean_or(small_topology):
    idx = select(small_topology, "name N or name C")
    assert sorted(idx.tolist()) == [0, 2, 3, 5]


def test_select_not(small_topology):
    idx = select(small_topology, "not name CA")
    assert sorted(idx.tolist()) == [0, 2, 3, 5]


def test_select_parentheses(small_topology):
    idx = select(small_topology, "(name CA or name C) and resname ALA")
    assert sorted(idx.tolist()) == [1, 2]


def test_select_unknown_keyword(small_topology):
    with pytest.raises(SelectionError):
        select(small_topology, "geom 5.0 around protein")


def test_select_empty_raises(small_topology):
    with pytest.raises(SelectionError):
        select(small_topology, "")
