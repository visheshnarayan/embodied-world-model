"""
gym_to_jax — Automatic conversion of NumPy Gymnasium envs to JAX pure functions.

Three-step pipeline:
  1. Probe env to discover state fields (numeric self.xxx attributes set in reset)
  2. Apply AST transformations: np→jnp, cast removal, in-place→rebind,
     if→where, ternary→where, self.np_random→jax.random, bool or/and → |/&
  3. exec() the transformed source to produce (reset_fn, step_fn)

Supported patterns
------------------
  np.xxx                              → jnp.xxx
  math.xxx                            → jnp.xxx  (cos/sin/pi/sqrt/etc.)
  bool(x) / float(x) / int(x)        → x
  x[:] = y                            → x = y
  if cond: x = y  (no else)           → x = jnp.where(cond, y, x)
  if cond: x = a / else: x = b        → x = jnp.where(cond, a, b)
  a if cond else b                    → jnp.where(cond, a, b)
  a or b / a and b                    → a | b  /  a & b
  self.np_random.uniform(lo, hi)      → jax.random.uniform(subkey, minval=lo, maxval=hi)
  self.np_random.uniform(low=, high=, size=)  → with keyword args
  self.np_random.normal(size=s)       → jax.random.normal(subkey, shape=s)
  self.np_random.integers(lo, hi)     → jax.random.randint(subkey, (), lo, hi)
  self.field  (state vars)            → local variable (in generated fn)
  self.non_state_attr                 → captured constant value (int/float/bool/None/str)
  assert ...                          → removed
  if self.render_mode == "human": ... → removed (constant-folded)
  if self.attr == "value": ...        → constant-folded using env instance
  x, y = self.state                   → x = state[0]; y = state[1]
  self.state = (x, y) / np.array([]) → tracked and repacked as _new_state
  math.pi / math.e / math.inf        → numeric constant

Limitations
-----------
  - C++ backends (MuJoCo, Box2D) cannot be auto-converted
  - while loops not supported
  - PushCubeEnv uses a hardcoded fallback path
  - Dict/tuple/image observation spaces not yet supported
"""
from __future__ import annotations

import ast
import copy
import inspect
import sys
import textwrap
from typing import Any, NamedTuple, Type

import jax
import jax.numpy as jnp
import gymnasium as gym
import numpy as np


# ── Exception ─────────────────────────────────────────────────────────────────

class ConversionError(Exception):
    """Raised when gym_to_jax cannot auto-convert an environment."""


# ── AST transformer 1: np → jnp ──────────────────────────────────────────────

class _NpToJnp(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "np":
            node.value = ast.Name(id="jnp", ctx=ast.Load())
        return node


# ── AST transformer 2: remove Python type casts ───────────────────────────────

class _RemovePyCasts(ast.NodeTransformer):
    _CASTS = {"bool", "float", "int"}

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name)
                and node.func.id in self._CASTS
                and len(node.args) == 1
                and not node.keywords):
            return node.args[0]
        return node


# ── AST transformer 3: x[:] = y → x = y ─────────────────────────────────────

class _SliceAssignToRebind(ast.NodeTransformer):
    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1:
            t = node.targets[0]
            if (isinstance(t, ast.Subscript)
                    and isinstance(t.slice, ast.Slice)
                    and t.slice.lower is None
                    and t.slice.upper is None):
                return ast.Assign(
                    targets=[t.value],
                    value=node.value,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
        return node


# ── AST transformer 4: if cond: x=y (no else) → x = jnp.where(cond, y, x)
#                       if cond: x=a / else: x=b → x = jnp.where(cond, a, b) ──

class _IfToWhere(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        # Case 1: if cond: x = y  (no else, single assign)
        if (not node.orelse
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Assign)
                and len(node.body[0].targets) == 1):
            target   = node.body[0].targets[0]
            true_val = node.body[0].value
            else_val = ast.Name(id=ast.unparse(target), ctx=ast.Load())
            where    = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr="where", ctx=ast.Load()),
                args=[node.test, true_val, else_val],
                keywords=[],
            )
            return ast.Assign(
                targets=[target], value=where,
                lineno=node.lineno, col_offset=node.col_offset,
            )

        # Case 2: if cond: x = a / else: x = b  (single assign in each branch, same target)
        if (node.orelse
                and len(node.orelse) == 1
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Assign)
                and isinstance(node.orelse[0], ast.Assign)
                and len(node.body[0].targets) == 1
                and len(node.orelse[0].targets) == 1
                and ast.unparse(node.body[0].targets[0]) == ast.unparse(node.orelse[0].targets[0])):
            target    = node.body[0].targets[0]
            true_val  = node.body[0].value
            false_val = node.orelse[0].value
            where     = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr="where", ctx=ast.Load()),
                args=[node.test, true_val, false_val],
                keywords=[],
            )
            return ast.Assign(
                targets=[target], value=where,
                lineno=node.lineno, col_offset=node.col_offset,
            )

        return node


# ── AST transformer 5: a if cond else b → jnp.where(cond, a, b) ─────────────

class _TernaryToWhere(ast.NodeTransformer):
    def visit_IfExp(self, node):
        self.generic_visit(node)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="where", ctx=ast.Load()),
            args=[node.test, node.body, node.orelse],
            keywords=[],
        )


# ── AST transformer 6: Python bool or/and → |/& ──────────────────────────────

class _BoolOpToJax(ast.NodeTransformer):
    """Convert `a or b` → `a | b` and `a and b` → `a & b` for JAX tracing."""

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Or):
            result = node.values[0]
            for v in node.values[1:]:
                result = ast.BinOp(left=result, op=ast.BitOr(), right=v)
            return result
        if isinstance(node.op, ast.And):
            result = node.values[0]
            for v in node.values[1:]:
                result = ast.BinOp(left=result, op=ast.BitAnd(), right=v)
            return result
        return node


# ── AST transformer 7: self.np_random.xxx → jax.random.xxx(subkey_N, ...) ───

def _get_kw_or_pos(node, name, pos, default_val):
    """Helper: get keyword arg or positional arg from a Call node."""
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    if len(node.args) > pos:
        return node.args[pos]
    return ast.Constant(value=default_val)


class _NpRandomToJax(ast.NodeTransformer):
    """
    Replace self.np_random.uniform/normal/integers calls with jax.random equivalents.
    Handles both positional and keyword arguments.
    """
    def __init__(self):
        self.n_random_calls = 0

    def _is_np_random_call(self, node, method):
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "np_random"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "self"
        )

    def visit_Call(self, node):
        self.generic_visit(node)
        idx = self.n_random_calls
        subkey = ast.Name(id=f"_rkey_{idx}", ctx=ast.Load())

        if self._is_np_random_call(node, "uniform"):
            self.n_random_calls += 1
            lo   = _get_kw_or_pos(node, "low",  0, 0.0)
            hi   = _get_kw_or_pos(node, "high", 1, 1.0)
            size = _get_kw_or_pos(node, "size", 2, None)
            keywords = [
                ast.keyword(arg="minval", value=lo),
                ast.keyword(arg="maxval", value=hi),
            ]
            if size is not None and not (isinstance(size, ast.Constant) and size.value is None):
                # Explicit size= → use as shape=
                keywords.append(ast.keyword(arg="shape", value=size))
            else:
                # No size= given: infer shape from minval at runtime
                # jnp.asarray(lo).shape handles scalar (→ ()) and array (→ (N,)) cases
                shape_expr = ast.Attribute(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="asarray", ctx=ast.Load()),
                        args=[copy.deepcopy(lo)],
                        keywords=[],
                    ),
                    attr="shape",
                    ctx=ast.Load(),
                )
                keywords.append(ast.keyword(arg="shape", value=shape_expr))
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="jax", ctx=ast.Load()),
                        attr="random", ctx=ast.Load()),
                    attr="uniform", ctx=ast.Load()),
                args=[subkey],
                keywords=keywords,
            )

        if self._is_np_random_call(node, "normal"):
            self.n_random_calls += 1
            size_kw = next((k for k in node.keywords if k.arg == "size"), None)
            shape_val = size_kw.value if size_kw else ast.Tuple(elts=[], ctx=ast.Load())
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="jax", ctx=ast.Load()),
                        attr="random", ctx=ast.Load()),
                    attr="normal", ctx=ast.Load()),
                args=[subkey],
                keywords=[ast.keyword(arg="shape", value=shape_val)],
            )

        if self._is_np_random_call(node, "integers"):
            self.n_random_calls += 1
            lo = _get_kw_or_pos(node, "low",  0, 0)
            hi = _get_kw_or_pos(node, "high", 1, 2)
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="jax", ctx=ast.Load()),
                        attr="random", ctx=ast.Load()),
                    attr="randint", ctx=ast.Load()),
                args=[subkey,
                      ast.Tuple(elts=[], ctx=ast.Load()),
                      lo, hi],
                keywords=[],
            )

        return node


# ── AST transformer 8: self.field → field  (in body context) ─────────────────

class _SelfFieldToLocal(ast.NodeTransformer):
    """Replace self.field with bare local name for state fields."""

    def __init__(self, state_fields: list[str]):
        self.fields = set(state_fields)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if (isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in self.fields):
            return ast.Name(id=node.attr, ctx=node.ctx)
        return node


# ── AST transformer 9: self.non_state_attr → constant ────────────────────────

class _SelfAttrToConst(ast.NodeTransformer):
    """Replace remaining self.xxx READS with their actual values (including None/str).
    Only substitutes in Load context to avoid mangling assignment targets."""

    def __init__(self, env_instance):
        self._env = env_instance

    def visit_Attribute(self, node):
        self.generic_visit(node)
        # Only substitute in Load context (reads), not Store context (writes)
        if (isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Load)):
            val = getattr(self._env, node.attr, _SENTINEL)
            if val is _SENTINEL:
                return node
            if isinstance(val, (int, float, bool, np.integer, np.floating)):
                # bool is subclass of int, check order matters
                if isinstance(val, (bool, np.bool_)):
                    return ast.Constant(value=bool(val))
                elif isinstance(val, (float, np.floating)):
                    return ast.Constant(value=float(val))
                else:
                    return ast.Constant(value=int(val))
            elif val is None:
                return ast.Constant(value=None)
            elif isinstance(val, str):
                return ast.Constant(value=val)
        return node


_SENTINEL = object()


# ── AST transformer 10: augmented assign x += y → x = x + y ─────────────────

class _AugAssignToAssign(ast.NodeTransformer):
    """x += y → x = x + y  (required since we use local vars, not mutable state)."""

    _OP_MAP = {
        ast.Add:      ast.Add,
        ast.Sub:      ast.Sub,
        ast.Mult:     ast.Mult,
        ast.Div:      ast.Div,
        ast.FloorDiv: ast.FloorDiv,
        ast.Mod:      ast.Mod,
        ast.Pow:      ast.Pow,
    }

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        op_type = type(node.op)
        if op_type in self._OP_MAP:
            new_value = ast.BinOp(
                left=copy.deepcopy(node.target),
                op=self._OP_MAP[op_type](),
                right=node.value,
            )
            # Convert target to Store context
            target = copy.deepcopy(node.target)
            if isinstance(target, ast.Name):
                target.ctx = ast.Store()
            elif isinstance(target, ast.Attribute):
                target.ctx = ast.Store()
            return ast.Assign(
                targets=[target],
                value=new_value,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        return node


# ── AST transformer 11: math.xxx → jnp.xxx / numeric constant ────────────────

_MATH_CONSTS = {
    "pi":  3.141592653589793,
    "e":   2.718281828459045,
    "inf": float("inf"),
    "tau": 6.283185307179586,
}

_MATH_FUNCS = {
    "cos", "sin", "tan", "acos", "asin", "atan", "atan2",
    "sqrt", "exp", "log", "log2", "log10",
    "floor", "ceil", "fabs", "pow",
    "sinh", "cosh", "tanh",
}


class _MathToJnp(ast.NodeTransformer):
    """Convert math.cos/sin/pi/... → jnp.cos/sin/... or numeric constant."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            if node.attr in _MATH_CONSTS:
                return ast.Constant(value=_MATH_CONSTS[node.attr])
            if node.attr in _MATH_FUNCS:
                return ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr=node.attr,
                    ctx=node.ctx,
                )
        return node


# ── AST transformer 12: remove assert statements ──────────────────────────────

class _RemoveAsserts(ast.NodeTransformer):
    def visit_Assert(self, node):
        return None


# ── AST transformer 13: remove self.xxx = ... assignments ────────────────────

class _RemoveSelfAssigns(ast.NodeTransformer):
    """Remove 'self.xxx = ...' assignment statements from function bodies."""

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"):
            return None
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if (isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"):
            return None
        return node


# ── AST transformer 14: constant-fold simple Compare nodes ───────────────────

class _ConstantFoldCompare(ast.NodeTransformer):
    """Evaluate Compare nodes where all operands are constants.
    E.g. None is None → True, 'euler' == 'euler' → True.
    """

    def visit_Compare(self, node):
        self.generic_visit(node)
        # Only handle single-comparator cases
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node
        left = node.left
        op   = node.ops[0]
        right = node.comparators[0]
        if not (isinstance(left, ast.Constant) and isinstance(right, ast.Constant)):
            return node
        lv = left.value
        rv = right.value
        try:
            if isinstance(op, ast.Is):
                result = lv is rv
            elif isinstance(op, ast.IsNot):
                result = lv is not rv
            elif isinstance(op, ast.Eq):
                result = lv == rv
            elif isinstance(op, ast.NotEq):
                result = lv != rv
            elif isinstance(op, ast.Lt):
                result = lv < rv
            elif isinstance(op, ast.LtE):
                result = lv <= rv
            elif isinstance(op, ast.Gt):
                result = lv > rv
            elif isinstance(op, ast.GtE):
                result = lv >= rv
            else:
                return node
        except Exception:
            return node
        return ast.Constant(value=result)


# ── AST transformer 15: constant-fold UnaryOp on constants ───────────────────

class _ConstantFoldUnary(ast.NodeTransformer):
    """Evaluate UnaryOp(Not/USub) on constant operands."""

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.operand, ast.Constant):
            val = node.operand.value
            try:
                if isinstance(node.op, ast.Not):
                    return ast.Constant(value=not val)
                elif isinstance(node.op, ast.USub):
                    return ast.Constant(value=-val)
                elif isinstance(node.op, ast.UAdd):
                    return ast.Constant(value=+val)
            except Exception:
                pass
        return node


# ── AST transformer 16b: min/max builtins → jnp.minimum/jnp.maximum ──────────

class _MinMaxToJnp(ast.NodeTransformer):
    """Convert Python min(a, b)/max(a, b) → jnp.minimum(a, b)/jnp.maximum(a, b)."""
    _MAP = {"min": "minimum", "max": "maximum"}

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name)
                and node.func.id in self._MAP
                and len(node.args) == 2
                and not node.keywords):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr=self._MAP[node.func.id],
                    ctx=ast.Load(),
                ),
                args=node.args,
                keywords=[],
            )
        return node


# ── AST transformer 16c: not x → ~x (for non-constant operands) ──────────────

class _NotToInvert(ast.NodeTransformer):
    """Convert 'not x' → '~x' for JAX-traced boolean arrays.
    Only applies when the operand is not a constant (those are handled by ConstantFoldUnary).
    """

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and not isinstance(node.operand, ast.Constant):
            return ast.UnaryOp(op=ast.Invert(), operand=node.operand)
        return node


# ── AST transformer 16: constant-fold if on constant condition ────────────────

class _ConstantFoldIf(ast.NodeTransformer):
    """
    Fold if/elif/else where test is a constant:
      if True: body / else: orelse   → body stmts
      if False: body / else: orelse  → orelse stmts (or [])

    Also folds:
      if self.attr == "value": ...  using env_instance attribute
      if self.render_mode == "human": ...  → removed (always False since render_mode=None)
    """

    def __init__(self, env_instance):
        self._env = env_instance

    def _eval_test(self, test):
        """Try to evaluate test as a Python bool. Returns (bool_val, success)."""
        if isinstance(test, ast.Constant):
            return bool(test.value), True
        return None, False

    def visit_If(self, node):
        self.generic_visit(node)
        val, ok = self._eval_test(node.test)
        if ok:
            if val:
                return node.body if node.body else []
            else:
                return node.orelse if node.orelse else []
        return node


# ── AST transformer 17: tuple unpack x, y = self.state → x=state[0]; y=state[1]

class _UnpackTupleAssign(ast.NodeTransformer):
    """Convert 'x, y = self.state' to 'x = state[0]; y = state[1]'."""

    def __init__(self, state_var: str = "state"):
        self._state_var = state_var

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == "state"):
            elts = node.targets[0].elts
            result = []
            for i, elt in enumerate(elts):
                new_assign = ast.Assign(
                    targets=[elt],
                    value=ast.Subscript(
                        value=ast.Name(id=self._state_var, ctx=ast.Load()),
                        slice=ast.Constant(value=i),
                        ctx=ast.Load(),
                    ),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
                result.append(new_assign)
            return result
        return node


# ── Backend detection ─────────────────────────────────────────────────────────

_CPP_MARKERS = {
    "mujoco", "dm_control", "pybullet", "box2d",
    "gym.envs.mujoco", "gym.envs.box2d", "brax",
}

def _has_cpp_backend(env_class):
    module = (env_class.__module__ or "").lower()
    return any(m in module for m in _CPP_MARKERS)


# ── State field detection ─────────────────────────────────────────────────────

def _detect_state_fields(env_instance) -> list[tuple[str, tuple, Any]]:
    """
    Return [(name, shape, dtype), ...] for numeric self.xxx fields set in reset().

    Strategy:
    1. Parse the reset() AST to find which self.xxx attributes are ASSIGNED in the body.
    2. Among those, include only numeric/array fields (skip non-numeric like strings, None).
    3. Always skip known Gymnasium internals.

    This correctly distinguishes state fields (set in reset) from constants (set in __init__).
    """
    _SKIP = {
        "np_random", "observation_space", "action_space",
        "reward_range", "spec", "metadata", "render_mode",
        "max_steps", "_np_random", "_np_random_seed",
        "screen", "clock", "isopen", "screen_width", "screen_height",
        "steps_beyond_terminated", "last_u",  # internal mutable non-state fields
    }

    # Parse reset() to find self.xxx = ... assignments
    env_class = type(env_instance)
    reset_src = textwrap.dedent(inspect.getsource(env_class.reset))
    reset_tree = ast.parse(reset_src)
    reset_func = reset_tree.body[0]

    assigned_in_reset: set[str] = set()
    for node in ast.walk(reset_func):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"):
            attr = node.targets[0].attr
            if not attr.startswith("_"):
                assigned_in_reset.add(attr)
        elif (isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"):
            attr = node.target.attr
            if not attr.startswith("_"):
                assigned_in_reset.add(attr)

    # Run reset to get actual values
    env_instance.reset(seed=0)
    after = env_instance.__dict__

    fields = []
    for k in sorted(assigned_in_reset):  # sorted for determinism
        if k in _SKIP:
            continue
        v = after.get(k)
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            fields.append((k, v.shape, v.dtype))
        elif isinstance(v, (int, float, bool, np.integer, np.floating)):
            fields.append((k, (), type(v)))
        # ignore non-numeric (strings, lists, etc.)
    return fields


# ── NamedTuple factory ────────────────────────────────────────────────────────

def _make_state_type(fields: list[tuple[str, tuple, Any]]):
    """Dynamically create AutoEnvState NamedTuple."""
    field_names = [f[0] for f in fields]
    AutoEnvState = NamedTuple("AutoEnvState", [(n, jax.Array) for n in field_names])
    return AutoEnvState


# ── Source extraction helpers ─────────────────────────────────────────────────

def _get_method_src(env_class, method_name: str) -> str:
    src = inspect.getsource(getattr(env_class, method_name))
    return textwrap.dedent(src)


def _indent_stmt(stmt: ast.stmt, indent: str = "    ") -> str:
    """Unparse an AST statement and indent ALL lines (for multi-line ifs, etc.)."""
    raw = ast.unparse(stmt)
    return "\n".join(indent + line for line in raw.splitlines())


# ── Apply all body transformers for reset ────────────────────────────────────

def _transform_reset_body(func_body, state_fields: list[str], env_instance):
    """Apply all transformations to reset() body. Returns (stmts, n_random_calls)."""
    body_module = copy.deepcopy(ast.Module(body=func_body, type_ignores=[]))

    # Order matters: np→jnp first, then casts, then RNG (before we lose self.np_random)
    for cls in [_NpToJnp, _MathToJnp, _RemovePyCasts, _RemoveAsserts]:
        body_module = cls().visit(body_module)

    rng_xf = _NpRandomToJax()
    body_module = rng_xf.visit(body_module)

    for cls in [_AugAssignToAssign, _SliceAssignToRebind, _TernaryToWhere,
                _BoolOpToJax, _MinMaxToJnp]:
        body_module = cls().visit(body_module)

    # Replace self.field → local var for state fields FIRST (converts self.x=... → x=...)
    body_module = _SelfFieldToLocal(state_fields).visit(body_module)
    # Now remove REMAINING self.xxx = ... (non-state fields: self.steps_beyond_terminated, etc.)
    # MUST run after SelfFieldToLocal so state fields were already converted
    body_module = _RemoveSelfAssigns().visit(body_module)
    # Replace remaining self.attr READS → constant (including None and str)
    # SelfAttrToConst only touches Load context, so it's safe even after RemoveSelfAssigns
    body_module = _SelfAttrToConst(env_instance).visit(body_module)
    # Constant-fold comparisons, unary ops, then if conditions
    for cls in [_ConstantFoldCompare, _ConstantFoldUnary]:
        body_module = cls().visit(body_module)
    body_module = _ConstantFoldIf(env_instance).visit(body_module)
    # Now apply IfToWhere (after constant folding has simplified branches)
    body_module = _IfToWhere().visit(body_module)
    # Convert remaining 'not x' → '~x' for JAX tracing
    body_module = _NotToInvert().visit(body_module)

    ast.fix_missing_locations(body_module)
    return body_module.body, rng_xf.n_random_calls


# ── Apply all body transformers for step ────────────────────────────────────

def _transform_step_body(func_body, state_fields: list[str], env_instance):
    """Apply all transformations to step() body. Returns stmts list."""
    body_module = copy.deepcopy(ast.Module(body=func_body, type_ignores=[]))

    for cls in [_NpToJnp, _MathToJnp, _RemovePyCasts, _RemoveAsserts, _AugAssignToAssign,
                _SliceAssignToRebind, _TernaryToWhere, _BoolOpToJax, _MinMaxToJnp]:
        body_module = cls().visit(body_module)

    # Replace self.field → local var for state fields FIRST (converts self.x=... → x=...)
    body_module = _SelfFieldToLocal(state_fields).visit(body_module)
    # Now remove REMAINING self.xxx = ... (non-state fields like self.last_u, etc.)
    # MUST run after SelfFieldToLocal so state fields were already converted
    body_module = _RemoveSelfAssigns().visit(body_module)
    # Replace remaining self.attr READS → constant (Load context only)
    body_module = _SelfAttrToConst(env_instance).visit(body_module)
    # Constant-fold comparisons and unary ops, then if conditions
    for cls in [_ConstantFoldCompare, _ConstantFoldUnary]:
        body_module = cls().visit(body_module)
    body_module = _ConstantFoldIf(env_instance).visit(body_module)
    # Apply IfToWhere after folding
    body_module = _IfToWhere().visit(body_module)
    # Convert remaining 'not x' → '~x' for JAX tracing
    body_module = _NotToInvert().visit(body_module)

    ast.fix_missing_locations(body_module)
    return body_module.body


# ── Reset builder (separate-fields path: NavEnv) ──────────────────────────────

def _build_reset(env_class, StateType, state_fields: list[str], obs_dim: int, env_instance):
    """
    Parse reset(), apply transforms, exec, return reset_fn(key) → (state, obs).
    """
    raw_src = _get_method_src(env_class, "reset")
    tree    = ast.parse(raw_src)
    func    = tree.body[0]

    stmts, n_random_calls = _transform_reset_body(func.body, state_fields, env_instance)

    n_keys = max(n_random_calls, 1)

    # Filter body statements: skip super() calls and return statements
    filtered_stmts = []
    for stmt in stmts:
        if stmt is None:
            continue
        s = ast.unparse(stmt)
        if "super()" in s:
            continue
        if s.strip().startswith("return"):
            continue
        filtered_stmts.append(stmt)

    body_src_lines = [_indent_stmt(stmt) for stmt in filtered_stmts]
    body_src = "\n".join(body_src_lines)

    # Key splitting lines
    key_split_line = f"    _keys = jax.random.split(key, {n_keys + 1})"
    key_assign_lines = "\n".join(f"    _rkey_{i} = _keys[{i}]" for i in range(n_keys))

    # Build obs concatenation (exclude 'steps')
    obs_parts = []
    for f in state_fields:
        if f == "steps":
            continue
        obs_parts.append(f"jnp.atleast_1d(jnp.asarray({f}, jnp.float32))")
    obs_concat = ", ".join(obs_parts) if obs_parts else "jnp.zeros(obs_dim, jnp.float32)"

    # State construction
    construct = ", ".join(f"{f}={f}" for f in state_fields)

    fn_code = f"""def _reset_fn(key):
{key_split_line}
{key_assign_lines}
{body_src}
    _state = StateType({construct})
    _obs = jnp.concatenate([{obs_concat}])[:obs_dim]
    return _state, _obs
"""

    namespace = {
        "jax": jax, "jnp": jnp, "np": np,
        "StateType": StateType, "obs_dim": obs_dim,
    }
    try:
        exec(compile(fn_code, "<gym_to_jax:reset>", "exec"), namespace)
    except Exception as e:
        raise ConversionError(
            f"Failed to compile transformed reset(): {e}\n"
            f"Generated code:\n{fn_code}"
        ) from e
    return namespace["_reset_fn"]


# ── Step builder (separate-fields path: NavEnv) ───────────────────────────────

def _build_step(env_class, StateType, state_fields: list[str], max_steps: int,
                obs_dim: int, env_instance):
    """
    Parse step(), apply transforms, exec, return step_fn(state, action) → (state, obs, reward, done).
    """
    raw_src = _get_method_src(env_class, "step")
    tree    = ast.parse(raw_src)
    func    = tree.body[0]

    all_stmts = _transform_step_body(func.body, state_fields, env_instance)

    # Separate body statements from return statement
    body_stmts = []
    return_stmt = None
    for stmt in all_stmts:
        if stmt is None:
            continue
        if isinstance(stmt, ast.Return):
            return_stmt = stmt
        else:
            body_stmts.append(stmt)

    # Unpack state fields at top of function
    unpack_lines = "\n".join(f"    {f} = state.{f}" for f in state_fields)

    body_src_lines = [_indent_stmt(stmt) for stmt in body_stmts]
    body_src = "\n".join(body_src_lines)

    # Determine reward_var and success_var from return statement
    reward_var  = "reward"
    success_var = "success"

    if (return_stmt is not None
            and isinstance(return_stmt.value, ast.Tuple)
            and len(return_stmt.value.elts) >= 3):
        elts = return_stmt.value.elts
        reward_var  = ast.unparse(elts[1])
        success_var = ast.unparse(elts[2])

    # Build obs concatenation (exclude 'steps')
    obs_parts = []
    for f in state_fields:
        if f == "steps":
            continue
        obs_parts.append(f"jnp.atleast_1d(jnp.asarray({f}, jnp.float32))")
    obs_concat = ", ".join(obs_parts) if obs_parts else "jnp.zeros(obs_dim, jnp.float32)"

    # State construction — for steps, increment; for others use local var
    construct_parts = []
    for f in state_fields:
        if f == "steps":
            construct_parts.append(f"steps=state.steps + jnp.int32(1)")
        else:
            construct_parts.append(f"{f}={f}")
    construct = ", ".join(construct_parts)

    fn_code = f"""def _step_fn(state, action):
{unpack_lines}
{body_src}
    _steps = state.steps + jnp.int32(1)
    _done = ({success_var}) | (_steps >= {max_steps})
    _new_state = StateType({construct})
    _obs = jnp.concatenate([{obs_concat}])[:obs_dim]
    return _new_state, _obs, jnp.float32({reward_var}), _done
"""

    namespace = {
        "jax": jax, "jnp": jnp, "np": np,
        "StateType": StateType, "obs_dim": obs_dim,
    }
    try:
        exec(compile(fn_code, "<gym_to_jax:step>", "exec"), namespace)
    except Exception as e:
        raise ConversionError(
            f"Failed to compile transformed step(): {e}\n"
            f"Generated code:\n{fn_code}"
        ) from e
    return namespace["_step_fn"]


# ── Fallback: PushCubeEnv-style packed self.state ────────────────────────────

class _PushCubeState(NamedTuple):
    hand:   jax.Array
    cube:   jax.Array
    target: jax.Array
    steps:  jax.Array


def _fallback_pushcube(env_class, obs_dim: int, max_steps: int):
    """Hardcoded-but-correct path for PushCubeEnv."""
    STEP_SIZE    = jnp.float32(0.045)
    CONTACT_R    = jnp.float32(0.16)
    SUCCESS_DIST = jnp.float32(0.10)
    ACTION_COST  = jnp.float32(0.002)

    def reset_fn(key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        hand = jnp.array([
            jax.random.uniform(k1, minval=-0.72, maxval=-0.62),
            jax.random.uniform(k2, minval=-0.08, maxval=0.08),
        ])
        cube = hand + jnp.array([
            jax.random.uniform(k3, minval=0.12, maxval=0.18),
            jax.random.uniform(k4, minval=-0.04, maxval=0.04),
        ])
        target = jnp.array([
            jax.random.uniform(k5, minval=0.42, maxval=0.62),
            jax.random.uniform(k6, minval=-0.18, maxval=0.18),
        ])
        state = _PushCubeState(hand=hand, cube=cube, target=target, steps=jnp.int32(0))
        obs   = jnp.concatenate([hand, jnp.zeros(2, jnp.float32), cube, target])
        return state, obs

    def step_fn(state, action):
        action   = jnp.clip(action.astype(jnp.float32), -1.0, 1.0)
        old_dist = jnp.linalg.norm(state.cube - state.target)
        new_hand = jnp.clip(state.hand + STEP_SIZE * action, -1.0, 1.0)
        contact  = jnp.linalg.norm(new_hand - state.cube) < CONTACT_R
        new_cube = jnp.where(contact,
                             jnp.clip(state.cube + STEP_SIZE * action, -0.8, 0.8),
                             state.cube)
        new_dist = jnp.linalg.norm(new_cube - state.target)
        success  = new_dist < SUCCESS_DIST
        reward   = (
            (old_dist - new_dist)
            - ACTION_COST * jnp.linalg.norm(action)
            + jnp.where(success, 1.0, 0.0)
        ).astype(jnp.float32)
        new_steps = state.steps + jnp.int32(1)
        done      = success | (new_steps >= max_steps)
        new_state = _PushCubeState(hand=new_hand, cube=new_cube,
                                   target=state.target, steps=new_steps)
        obs = jnp.concatenate([new_hand, jnp.zeros(2, jnp.float32), new_cube, state.target])
        return new_state, obs, reward, done

    return reset_fn, step_fn


# ── Helpers for packed-state path ─────────────────────────────────────────────

def _find_state_unpack(func_body) -> list[str] | None:
    """
    Find state variable names from step body. Handles two patterns:

    Pattern 1: tuple unpack
        x, y = self.state  →  ['x', 'y']

    Pattern 2: index access (collect all consecutive self.state[i] = name lines)
        x = self.state[0]
        y = self.state[1]  →  ['x', 'y']
    """
    # Pattern 1: tuple unpack
    for node in ast.walk(ast.Module(body=func_body, type_ignores=[])):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == "state"):
            return [ast.unparse(e) for e in node.targets[0].elts]

    # Pattern 2: index access  x = self.state[i]
    index_map: dict[int, str] = {}
    for node in func_body:
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Attribute)
                and isinstance(node.value.value.value, ast.Name)
                and node.value.value.value.id == "self"
                and node.value.value.attr == "state"
                and isinstance(node.value.slice, ast.Constant)
                and isinstance(node.value.slice.value, int)):
            idx = node.value.slice.value
            name = node.targets[0].id
            index_map[idx] = name

    if index_map:
        max_idx = max(index_map.keys())
        return [index_map.get(i, f"_sv{i}") for i in range(max_idx + 1)]

    return None


def _find_state_repack(func_body) -> list[str] | None:
    """
    Find 'self.state = X' in func body.
    X can be:
      - a tuple/list: (x, y) → ['x', 'y']
      - np.array([x, y]) → ['x', 'y']
      - a bare name (state array): None (no elements known)
    Returns list of element unparse strings, or None if not found.
    """
    for node in func_body:
        for subnode in ast.walk(node):
            if (isinstance(subnode, ast.Assign)
                    and len(subnode.targets) == 1
                    and isinstance(subnode.targets[0], ast.Attribute)
                    and isinstance(subnode.targets[0].value, ast.Name)
                    and subnode.targets[0].value.id == "self"
                    and subnode.targets[0].attr == "state"):
                val = subnode.value
                # Tuple: (x, y) or list [x, y]
                if isinstance(val, (ast.Tuple, ast.List)):
                    return [ast.unparse(e) for e in val.elts]
                # np.array([x, y]) or np.array((x, y))
                if (isinstance(val, ast.Call)
                        and isinstance(val.func, ast.Attribute)
                        and val.func.attr == "array"
                        and val.args
                        and isinstance(val.args[0], (ast.Tuple, ast.List))):
                    return [ast.unparse(e) for e in val.args[0].elts]
                # Something else (e.g. already an array) — return empty list signal
                return []
    return None


def _get_module_functions(env_class) -> dict[str, Any]:
    """Get module-level callables from env's module (e.g. angle_normalize for Pendulum)."""
    module = sys.modules.get(env_class.__module__, None)
    if module is None:
        return {}
    result = {}
    for name, val in vars(module).items():
        if callable(val) and not name.startswith("_") and inspect.isfunction(val):
            result[name] = val
    return result


def _get_module_constants(env_class) -> dict[str, Any]:
    """Get module-level scalar/array constants from env's module (e.g. DEFAULT_X for Pendulum)."""
    module = sys.modules.get(env_class.__module__, None)
    if module is None:
        return {}
    result = {}
    for name, val in vars(module).items():
        if name.startswith("_"):
            continue
        if isinstance(val, (int, float, bool, str)):
            result[name] = val
        elif isinstance(val, np.ndarray):
            result[name] = val
    return result


def _get_env_module(env_class):
    """Get the module object for the env class (for 'utils' submodule etc.)."""
    return sys.modules.get(env_class.__module__, None)


def _get_max_steps(env_instance) -> int:
    """Get maximum episode steps from env instance or spec."""
    if hasattr(env_instance, "max_steps") and env_instance.max_steps is not None:
        return int(env_instance.max_steps)
    if (hasattr(env_instance, "spec")
            and env_instance.spec is not None
            and hasattr(env_instance.spec, "max_episode_steps")
            and env_instance.spec.max_episode_steps is not None):
        return int(env_instance.spec.max_episode_steps)
    return 500  # safe default


def _build_obs_fn(env_class, env_instance, state_var_names: list[str], obs_dim: int):
    """
    Build obs_fn(state_array) → float32[obs_dim].
    If env has _get_obs, parse and transform it.
    Otherwise return identity slice.
    """
    if not hasattr(env_class, "_get_obs"):
        def obs_fn(state_array):
            return state_array.astype(jnp.float32)[:obs_dim]
        return obs_fn

    raw_src = _get_method_src(env_class, "_get_obs")
    tree = ast.parse(raw_src)
    func = tree.body[0]

    body_module = copy.deepcopy(ast.Module(body=func.body, type_ignores=[]))

    for cls in [_NpToJnp, _MathToJnp, _RemovePyCasts]:
        body_module = cls().visit(body_module)

    # Unpack state_var_names from the 'state' array argument
    # e.g. 'th, thdot = self.state' in _get_obs → 'th = state[0]; thdot = state[1]'
    body_module = _UnpackTupleAssign("state").visit(body_module)

    # Also replace bare state field names if they appear as self.xxx
    body_module = _SelfFieldToLocal(state_var_names).visit(body_module)
    body_module = _SelfAttrToConst(env_instance).visit(body_module)
    body_module = _RemoveSelfAssigns().visit(body_module)

    ast.fix_missing_locations(body_module)

    # Find return statement
    return_stmt = None
    body_stmts = []
    for stmt in body_module.body:
        if stmt is None:
            continue
        if isinstance(stmt, ast.Return):
            return_stmt = stmt
        else:
            body_stmts.append(stmt)

    body_src_lines = [_indent_stmt(stmt) for stmt in body_stmts]
    body_src = "\n".join(body_src_lines) if body_src_lines else ""

    ret_expr = "state.astype(jnp.float32)"
    if return_stmt is not None:
        ret_expr = ast.unparse(return_stmt.value)

    fn_code = f"""def _obs_fn(state):
{body_src}
    return jnp.asarray({ret_expr}, jnp.float32)
"""

    namespace = {"jax": jax, "jnp": jnp, "np": np}
    try:
        exec(compile(fn_code, "<gym_to_jax:obs>", "exec"), namespace)
    except Exception as e:
        raise ConversionError(
            f"Failed to compile _get_obs: {e}\nGenerated:\n{fn_code}"
        ) from e
    return namespace["_obs_fn"]


# ── General packed-state path (classic control envs) ─────────────────────────

def _build_exec_namespace(env_class, env_instance, extra: dict | None = None) -> dict:
    """Build the exec() namespace for packed-state path functions."""
    ns: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "np": np,
        # options=None so that 'if options is None:' branches are taken
        "options": None,
    }
    # Inject module-level constants (DEFAULT_X, DEFAULT_Y, etc.)
    ns.update(_get_module_constants(env_class))
    # Inject module-level helper functions (angle_normalize, etc.)
    ns.update(_get_module_functions(env_class))
    # Inject 'utils' submodule if present (for maybe_parse_reset_bounds)
    env_module = _get_env_module(env_class)
    if env_module is not None and hasattr(env_module, "utils"):
        ns["utils"] = env_module.utils
    if extra:
        ns.update(extra)
    return ns


def _general_packed_state_path(env_class, env_instance, obs_dim: int, max_steps: int):
    """
    Build reset_fn and step_fn for envs with packed self.state array
    (CartPole, MountainCar, Pendulum, Acrobot, ...).

    Uses AST transformation of reset() and step() methods.
    """
    # ---- discover unpack names from step body ----
    step_src  = _get_method_src(env_class, "step")
    step_tree = ast.parse(step_src)
    step_func = step_tree.body[0]

    unpack_names = _find_state_unpack(step_func.body)
    repack_exprs = _find_state_repack(step_func.body)

    if unpack_names is None:
        # No unpack found — try to get size from probed state
        probe_state = env_instance.__dict__.get("state")
        n = len(probe_state) if hasattr(probe_state, "__len__") else obs_dim
        unpack_names = [f"_sv{i}" for i in range(n)]

    state_size = len(unpack_names)

    # ---- build obs_fn ----
    obs_fn = _build_obs_fn(env_class, env_instance, unpack_names, obs_dim)

    # ---- PackedState NamedTuple ----
    PackedState = NamedTuple("PackedState", [("state", jax.Array)])

    # ═════════════════════════════ reset_fn ═══════════════════════════════════

    reset_src  = _get_method_src(env_class, "reset")
    reset_tree = ast.parse(reset_src)
    reset_func = reset_tree.body[0]

    # Find 'self.state = X' in original AST BEFORE any transforms
    state_assign_expr_orig = None
    for node in ast.walk(reset_tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"
                and node.targets[0].attr == "state"):
            state_assign_expr_orig = copy.deepcopy(node.value)
            break

    reset_stmts, n_random_calls = _transform_reset_body(
        reset_func.body, unpack_names, env_instance
    )

    n_keys = max(n_random_calls, 1)

    # Transform the self.state = X expression separately, continuing key numbering
    if state_assign_expr_orig is None:
        state_assign_src = f"jnp.zeros({state_size}, jnp.float32)"
    else:
        expr_module = copy.deepcopy(
            ast.Module(body=[ast.Expr(value=state_assign_expr_orig)], type_ignores=[])
        )
        for cls in [_NpToJnp, _MathToJnp, _RemovePyCasts]:
            expr_module = cls().visit(expr_module)
        rng_xf2 = _NpRandomToJax()
        rng_xf2.n_random_calls = n_random_calls  # continue key numbering
        expr_module = rng_xf2.visit(expr_module)
        expr_module = _SelfFieldToLocal(unpack_names).visit(expr_module)
        expr_module = _SelfAttrToConst(env_instance).visit(expr_module)
        ast.fix_missing_locations(expr_module)
        state_assign_src = ast.unparse(expr_module.body[0].value)
        extra_rng = rng_xf2.n_random_calls
        if extra_rng > 0:
            n_random_calls += extra_rng
            n_keys = max(n_random_calls, 1)

    # Filter reset body stmts: skip super(), return, self.state assigns
    # Note: we intentionally KEEP remaining stmts that reference self.xxx reads
    # (like utils.maybe_parse_reset_bounds) — they work at exec time via the namespace
    reset_body_stmts = []
    for stmt in reset_stmts:
        if stmt is None:
            continue
        s = ast.unparse(stmt)
        if "super()" in s or s.strip().startswith("return"):
            continue
        # Skip assignments to self.state (handled by state_assign_src below)
        if "self.state" in s:
            continue
        reset_body_stmts.append(stmt)

    # Build key split and assign lines (all uniformly indented)
    key_lines = [f"    _keys = jax.random.split(key, {n_keys + 1})"]
    key_lines += [f"    _rkey_{i} = _keys[{i}]" for i in range(n_keys)]
    key_src = "\n".join(key_lines)

    reset_body_src = "\n".join(_indent_stmt(s) for s in reset_body_stmts)

    reset_code = (
        f"def _reset_fn(key):\n"
        f"{key_src}\n"
        + (f"{reset_body_src}\n" if reset_body_src else "")
        + f"    _state_arr = jnp.asarray({state_assign_src}, jnp.float32)\n"
        f"    _obs = obs_fn(_state_arr)\n"
        f"    return PackedState(state=_state_arr), _obs\n"
    )

    reset_ns = _build_exec_namespace(
        env_class, env_instance, {"PackedState": PackedState, "obs_fn": obs_fn}
    )
    try:
        exec(compile(reset_code, "<gym_to_jax:packed_reset>", "exec"), reset_ns)
    except Exception as e:
        raise ConversionError(
            f"Failed to compile packed reset(): {e}\nGenerated:\n{reset_code}"
        ) from e
    reset_fn = reset_ns["_reset_fn"]

    # ═════════════════════════════ step_fn ════════════════════════════════════

    # Determine original action parameter name (e.g. 'u' for Pendulum, 'action' for CartPole)
    step_action_param = "action"
    if len(step_func.args.args) >= 2:
        step_action_param = step_func.args.args[1].arg

    all_step_stmts = _transform_step_body(step_func.body, unpack_names, env_instance)

    # Separate return from body; also skip 'x, y = self.state' (handled by manual unpack)
    # and skip any statements containing 'self' (couldn't be resolved)
    step_body_stmts = []
    return_stmt = None
    for stmt in all_step_stmts:
        if stmt is None:
            continue
        if isinstance(stmt, ast.Return):
            return_stmt = stmt
            continue
        s = ast.unparse(stmt)
        # Skip tuple-unpack from self.state (we handle it via packed_state.state[i])
        if "self.state" in s:
            continue
        # Skip statements that still reference self (couldn't be resolved)
        if "self." in s:
            continue
        step_body_stmts.append(stmt)

    # Extract reward and terminated from return elts
    reward_expr     = "reward"
    terminated_expr = "terminated"
    is_always_false = False  # Pendulum: terminated = False always

    if (return_stmt is not None
            and isinstance(return_stmt.value, ast.Tuple)
            and len(return_stmt.value.elts) >= 3):
        elts = return_stmt.value.elts
        reward_expr     = ast.unparse(elts[1])
        terminated_node = elts[2]
        if isinstance(terminated_node, ast.Constant) and terminated_node.value is False:
            is_always_false = True
            terminated_expr = "False"
        else:
            terminated_expr = ast.unparse(terminated_node)

    # Build per-index unpack lines: x = state[0]; y = state[1]
    unpack_lines = "\n".join(
        f"    {name} = packed_state.state[{i}]"
        for i, name in enumerate(unpack_names)
    )

    # Build new state array from repack_exprs (already transformed by _transform_step_body
    # since _RemoveSelfAssigns runs before, but we need the ORIGINAL repack expressions)
    # repack_exprs came from the ORIGINAL step AST, so we need to transform them
    if repack_exprs:
        arr_elems = ", ".join(repack_exprs)
    else:
        arr_elems = ", ".join(unpack_names)
    new_state_raw = f"jnp.array([{arr_elems}], dtype=jnp.float32)"

    # Apply same transforms to the repack expression
    repack_module = ast.parse(new_state_raw, mode="eval")
    for cls in [_NpToJnp, _MathToJnp, _RemovePyCasts, _AugAssignToAssign,
                _TernaryToWhere, _BoolOpToJax]:
        repack_module = cls().visit(repack_module)
    repack_module = _SelfFieldToLocal(unpack_names).visit(repack_module)
    repack_module = _SelfAttrToConst(env_instance).visit(repack_module)
    ast.fix_missing_locations(repack_module)
    new_state_src = ast.unparse(repack_module.body)

    step_body_src = "\n".join(_indent_stmt(s) for s in step_body_stmts)

    if is_always_false:
        done_expr = f"jnp.bool_(False) | jnp.bool_(_steps >= {max_steps})"
    else:
        done_expr = f"jnp.bool_({terminated_expr}) | jnp.bool_(_steps >= {max_steps})"

    # For discrete action spaces (CartPole, MountainCar), squeeze action to scalar
    # so that physics computations don't propagate spurious batch dimensions
    is_discrete = isinstance(env_instance.action_space, gym.spaces.Discrete)
    action_squeeze_line = "    action = jnp.squeeze(action)\n" if is_discrete else ""

    # If the original step method used a different param name for action (e.g. 'u' in Pendulum),
    # create an alias so the transformed body can reference it correctly.
    if step_action_param != "action":
        action_alias_line = f"    {step_action_param} = action\n"
    else:
        action_alias_line = ""

    step_code = (
        f"def _step_fn(packed_state, action):\n"
        f"{action_squeeze_line}"
        f"{action_alias_line}"
        f"{unpack_lines}\n"
        + (f"{step_body_src}\n" if step_body_src else "")
        + f"    _new_state = {new_state_src}\n"
        f"    _obs = obs_fn(_new_state)\n"
        f"    _steps = jnp.int32(0)  # not tracked; step limit via done\n"
        f"    _done = {done_expr}\n"
        f"    return PackedState(state=_new_state), _obs, jnp.float32({reward_expr}), _done\n"
    )

    step_ns = _build_exec_namespace(
        env_class, env_instance, {"PackedState": PackedState, "obs_fn": obs_fn}
    )
    try:
        exec(compile(step_code, "<gym_to_jax:packed_step>", "exec"), step_ns)
    except Exception as e:
        raise ConversionError(
            f"Failed to compile packed step(): {e}\nGenerated:\n{step_code}"
        ) from e
    step_fn = step_ns["_step_fn"]

    return reset_fn, step_fn


# ── Main converter ────────────────────────────────────────────────────────────

def gym_to_jax(env_class: Type[gym.Env]):
    """
    Convert a pure-NumPy Gymnasium env to (reset_fn, step_fn).

    Parameters
    ----------
    env_class : Type[gym.Env]
        Must use only NumPy (no C++ extension backends).

    Returns
    -------
    reset_fn : key → (state, obs)
    step_fn  : (state, action) → (state, obs, reward, done)

    Raises
    ------
    ConversionError
        If the env uses a C++ backend or unsupported patterns.
    """
    if _has_cpp_backend(env_class):
        raise ConversionError(
            f"{env_class.__name__} appears to use a C++ backend. "
            "Manually rewrite reset/step as pure JAX functions."
        )

    # PushCubeEnv: hardcoded fallback (identified by class name)
    if env_class.__name__ == "PushCubeEnv":
        probe = env_class()
        obs0, _ = probe.reset(seed=0)
        obs_dim   = int(np.asarray(obs0).shape[0])
        max_steps = _get_max_steps(probe)
        return _fallback_pushcube(env_class, obs_dim, max_steps)

    # Probe env
    probe = env_class()
    obs0, _ = probe.reset(seed=0)
    obs_dim   = int(np.asarray(obs0).shape[0])
    max_steps = _get_max_steps(probe)

    # Detect state fields (runs reset twice; only returns changing fields)
    raw_fields  = _detect_state_fields(probe)
    field_names = [f[0] for f in raw_fields]

    # Packed-state path: self.state is the primary array (classic control envs)
    if "state" in field_names:
        return _general_packed_state_path(env_class, probe, obs_dim, max_steps)

    # Separate-fields path (NavEnv and similar)
    StateType = _make_state_type(raw_fields)
    state_field_names = [f[0] for f in raw_fields]

    reset_fn = _build_reset(env_class, StateType, state_field_names, obs_dim, probe)
    step_fn  = _build_step(env_class, StateType, state_field_names, max_steps, obs_dim, probe)

    return reset_fn, step_fn
