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
  bool(x) / float(x) / int(x)        → x
  x[:] = y                            → x = y
  if cond: x = y  (no else)           → x = jnp.where(cond, y, x)
  a if cond else b                    → jnp.where(cond, a, b)
  a or b / a and b                    → a | b  /  a & b
  self.np_random.uniform(lo, hi)      → jax.random.uniform(subkey, minval=lo, maxval=hi)
  self.np_random.normal(size=s)       → jax.random.normal(subkey, shape=s)
  self.np_random.integers(lo, hi)     → jax.random.randint(subkey, (), lo, hi)
  self.field  (state vars)            → local variable (in generated fn)
  self.non_state_attr                 → captured constant value

Limitations
-----------
  - C++ backends (MuJoCo, Box2D) cannot be auto-converted
  - while loops / elif chains on array values not supported
  - Packed self.state arrays (PushCubeEnv-style) use a fallback path
  - Dict/tuple/image observation spaces not yet supported
"""
from __future__ import annotations

import ast
import copy
import inspect
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


# ── AST transformer 4: if cond: x = y → x = jnp.where(cond, y, x) ──────────

class _IfToWhere(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
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

class _NpRandomToJax(ast.NodeTransformer):
    """
    Replace self.np_random.uniform/normal/integers calls with jax.random equivalents.
    Counts how many random calls are found (stored in self.n_random_calls).
    Each call uses _rkey_0, _rkey_1, ... which must be injected at function top.
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
            args = node.args
            lo = args[0] if len(args) > 0 else ast.Constant(value=0.0)
            hi = args[1] if len(args) > 1 else ast.Constant(value=1.0)
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="jax", ctx=ast.Load()),
                        attr="random", ctx=ast.Load()),
                    attr="uniform", ctx=ast.Load()),
                args=[subkey],
                keywords=[
                    ast.keyword(arg="minval", value=lo),
                    ast.keyword(arg="maxval", value=hi),
                ],
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
            args = node.args
            lo = args[0] if len(args) > 0 else ast.Constant(value=0)
            hi = args[1] if len(args) > 1 else ast.Constant(value=2)
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
    """Replace remaining self.xxx references with their actual values."""

    def __init__(self, env_instance):
        self._env = env_instance

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            val = getattr(self._env, node.attr, None)
            if isinstance(val, (int, float, bool, np.integer, np.floating)):
                return ast.Constant(value=float(val) if isinstance(val, (float, np.floating)) else int(val))
        return node


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
    Excludes known Gymnasium internals.
    """
    _SKIP = {
        "np_random", "observation_space", "action_space",
        "reward_range", "spec", "metadata", "render_mode",
        "max_steps", "_np_random", "_np_random_seed",
    }
    env_instance.reset(seed=0)
    after = env_instance.__dict__

    fields = []
    for k, v in after.items():
        if k.startswith("_") or k in _SKIP:
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


# ── Apply all body transformers for reset ────────────────────────────────────

def _transform_reset_body(func_body, state_fields: list[str], env_instance):
    """Apply all transformations to reset() body. Returns (stmts, n_random_calls)."""
    body_module = copy.deepcopy(ast.Module(body=func_body, type_ignores=[]))

    # Order matters: np→jnp first, then casts, then RNG (before we lose self.np_random)
    for cls in [_NpToJnp, _RemovePyCasts]:
        body_module = cls().visit(body_module)

    rng_xf = _NpRandomToJax()
    body_module = rng_xf.visit(body_module)

    for cls in [_AugAssignToAssign, _SliceAssignToRebind, _IfToWhere, _TernaryToWhere, _BoolOpToJax]:
        body_module = cls().visit(body_module)

    # Replace self.field → local var for state fields
    body_module = _SelfFieldToLocal(state_fields).visit(body_module)
    # Replace remaining self.attr → constant
    body_module = _SelfAttrToConst(env_instance).visit(body_module)

    ast.fix_missing_locations(body_module)
    return body_module.body, rng_xf.n_random_calls


# ── Apply all body transformers for step ────────────────────────────────────

def _transform_step_body(func_body, state_fields: list[str], env_instance):
    """Apply all transformations to step() body. Returns stmts list."""
    body_module = copy.deepcopy(ast.Module(body=func_body, type_ignores=[]))

    for cls in [_NpToJnp, _RemovePyCasts, _AugAssignToAssign,
                _SliceAssignToRebind, _IfToWhere, _TernaryToWhere, _BoolOpToJax]:
        body_module = cls().visit(body_module)

    # Replace self.field → local var for state fields
    body_module = _SelfFieldToLocal(state_fields).visit(body_module)
    # Replace remaining self.attr → constant
    body_module = _SelfAttrToConst(env_instance).visit(body_module)

    ast.fix_missing_locations(body_module)
    return body_module.body


# ── Reset builder ─────────────────────────────────────────────────────────────

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
        s = ast.unparse(stmt)
        if "super()" in s:
            continue
        if s.strip().startswith("return"):
            continue
        filtered_stmts.append(stmt)

    body_src_lines = ["    " + ast.unparse(stmt) for stmt in filtered_stmts]
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


# ── Step builder ──────────────────────────────────────────────────────────────

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
        if isinstance(stmt, ast.Return):
            return_stmt = stmt
        else:
            body_stmts.append(stmt)

    # Unpack state fields at top of function
    unpack_lines = "\n".join(f"    {f} = state.{f}" for f in state_fields)

    body_src_lines = ["    " + ast.unparse(stmt) for stmt in body_stmts]
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
    """Hardcoded-but-correct path for envs with packed self.state."""
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

    # Probe env
    probe = env_class()
    obs0, _ = probe.reset(seed=0)
    obs_dim   = int(np.asarray(obs0).shape[0])
    max_steps = int(getattr(probe, "max_steps", 200))

    # Detect state fields
    raw_fields = _detect_state_fields(probe)
    field_names = [f[0] for f in raw_fields]

    # Check for packed self.state (PushCubeEnv-style fallback)
    if "state" in field_names:
        return _fallback_pushcube(env_class, obs_dim, max_steps)

    # Build dynamic AutoEnvState
    StateType = _make_state_type(raw_fields)
    state_field_names = [f[0] for f in raw_fields]

    # Build reset_fn and step_fn via AST pipeline
    reset_fn = _build_reset(env_class, StateType, state_field_names, obs_dim, probe)
    step_fn  = _build_step(env_class, StateType, state_field_names, max_steps, obs_dim, probe)

    return reset_fn, step_fn
