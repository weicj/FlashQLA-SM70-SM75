# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Static checks for autograd ``Function`` signatures.

PyTorch validates that ``Function.backward`` returns exactly as many gradients
as ``Function.forward`` received non-``ctx`` inputs; mismatches raise at
``.backward()`` time.  ``tests/test_gdr.py`` invokes ``chunk_gated_delta_rule_fwd``
and ``chunk_gated_delta_rule_bwd`` directly, bypassing the autograd path, so
drift between the forward signature and the backward return tuple goes
uncaught by the existing suite.

These tests parse the source files with ``ast`` instead of importing the
modules so they run on CPU-only / non-Hopper machines.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_INIT = "flash_qla/ops/gated_delta_rule/chunk/__init__.py"


def _parse(rel_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _get_class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def _get_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name!r} not found on {cls.name}")


def test_chunk_gated_delta_rule_grad_count_matches_forward_inputs():
    """``backward`` must return one gradient per non-``ctx`` input of ``forward``."""
    module = _parse(CHUNK_INIT)
    cls = _get_class(module, "ChunkGatedDeltaRuleFunction")

    fwd = _get_method(cls, "forward")
    fwd_args = fwd.args.args
    assert fwd_args and fwd_args[0].arg == "ctx", (
        "forward must take `ctx` as its first argument"
    )
    n_inputs = len(fwd_args) - 1  # exclude ctx

    bwd = _get_method(cls, "backward")
    returns = [n for n in ast.walk(bwd) if isinstance(n, ast.Return)]
    assert len(returns) == 1, f"expected one Return in backward, got {len(returns)}"
    assert isinstance(returns[0].value, ast.Tuple), (
        "backward must return a tuple literal"
    )
    n_grads = len(returns[0].value.elts)

    assert n_inputs == n_grads, (
        f"backward returns {n_grads} gradients but forward takes {n_inputs} non-ctx "
        f"inputs; PyTorch will raise a count-mismatch error at .backward() time."
    )


if __name__ == "__main__":
    test_chunk_gated_delta_rule_grad_count_matches_forward_inputs()
    print("OK")
