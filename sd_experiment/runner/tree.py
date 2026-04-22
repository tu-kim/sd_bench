"""`speculative_token_tree` construction for vLLM 0.17.1.

vLLM 0.17.1 does **not** expose an `eagle_topk` parameter.  Tree shape is
controlled entirely by the `speculative_token_tree` field of
`SpeculativeConfig`, which takes a string representation of
`list[tuple[int, ...]]` (see VLLM_API_NOTES.md §2.2).

This module maps the harness's (D, K, T) CLI triple to a tree string and
enforces the consistency constraint:

    T = sum(K ** d for d in 1..D)

which is the total-node count of a **perfect K-ary tree of depth D**.
Configurations that violate this — or that don't yield an integer K —
are rejected up front (and, at the grid level, skipped).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class TreeSpec:
    depth: int        # D = num_speculative_tokens
    branching: int    # K (implicit in vLLM; we build the tree string)
    total: int        # T = num_draft_tokens; == sum(K**d for d in 1..D)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _expected_total(depth: int, branching: int) -> int:
    if branching < 1 or depth < 1:
        raise ValueError(f"invalid depth/branching: {depth=}, {branching=}")
    if branching == 1:
        return depth
    # K + K^2 + ... + K^D = K * (K^D - 1) / (K - 1)
    return branching * (branching**depth - 1) // (branching - 1)


def build_tree_string(depth: int, branching: int) -> tuple[str, int]:
    """Return (spec_token_tree_string, total_nodes).

    For K=1, a chain of depth D.  For K>1, a perfect K-ary tree.
    """
    if depth < 1 or branching < 1:
        raise ValueError(f"depth/branching must be >= 1: {depth=}, {branching=}")
    paths: list[tuple[int, ...]] = []
    for d in range(1, depth + 1):
        for p in product(range(branching), repeat=d):
            paths.append(p)
    # vLLM sorts breadth-first after parsing, so our ordering doesn't matter
    # — but we emit BFS-order anyway for human readability.
    return str(paths), len(paths)


# ---------------------------------------------------------------------------
# Validation & derivation
# ---------------------------------------------------------------------------
def validate_tree_params(
    depth: int | None,
    branching: int | None,
    total: int | None,
) -> TreeSpec:
    """Normalise the (D, K, T) triple.  At least two of three must be set.

    Returns a TreeSpec with all three resolved, or raises ValueError if
    the triple is inconsistent.
    """
    if depth is None:
        raise ValueError("--num-speculative-tokens (D) is required")

    if branching is None and total is None:
        # Default: chain (K=1) ⇒ T=D.
        branching = 1
        total = depth
    elif branching is None:
        assert total is not None
        # Solve for K given (D, T).  Linear scan is cheap: K in [1, T].
        found = None
        for k in range(1, total + 1):
            if _expected_total(depth, k) == total:
                found = k
                break
            if _expected_total(depth, k) > total:
                break
        if found is None:
            raise ValueError(
                f"No integer branching K solves sum(K^d, d=1..{depth})={total}; "
                f"skip this config."
            )
        branching = found
    elif total is None:
        total = _expected_total(depth, branching)
    else:
        # Both branching and total given: check consistency.
        expected = _expected_total(depth, branching)
        if expected != total:
            raise ValueError(
                f"Inconsistent tree params: D={depth}, K={branching} "
                f"implies T={expected}, but --num-draft-tokens={total}."
            )

    return TreeSpec(depth=depth, branching=branching, total=total)
