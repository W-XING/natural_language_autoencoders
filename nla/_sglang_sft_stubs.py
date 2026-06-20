"""Stub sglang modules so SFT can run without a full sglang+router install.

SFT (--debug-train-only) never touches the sglang engine — it's pure FSDP
training on pre-generated parquets. But miles' top-level imports pull in
sglang, sglang_router, and transitive deps with tight version pins (e.g.
sglang 0.5.x → transformers 4.57+ for GptOssConfig). If your environment has
an older transformers and you can't upgrade, this lets SFT run anyway.

These stubs satisfy the import chain with no-ops. The actual sglang engine
code paths (rollout generation, router) are unreachable in SFT and will
crash loudly if somehow hit. That's the intended failure mode — you'll know
immediately if you accidentally try to use this for RL without a real env.

Usage: in your launcher, before importing train.py:
    import nla._sglang_sft_stubs  # noqa: F401 — sets up sys.modules
"""

import sys
import types


def _stub_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# ─── sglang.srt.constants — string constants, used as offload tags ───
# (real values don't matter for SFT; these just need to be hashable)
_srt = _stub_module("sglang.srt")
_stub_module("sglang").srt = _srt
_constants = _stub_module("sglang.srt.constants")
_constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
_constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
_constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
_srt.constants = _constants

# ─── sglang.srt.patch_torch — miles' fsdp update_weight_utils imports
# `monkey_patch_torch_reductions` at module top level. It rewrites torch's
# CUDA-tensor IPC reduction for zero-copy weight transfer to the sglang engine;
# SFT never transfers weights to an engine, so a no-op is correct. (Registered
# as its own sys.modules entry so `from sglang.srt.patch_torch import ...`
# resolves without sglang.srt being a real package.) ───
_patch_torch = _stub_module("sglang.srt.patch_torch")
_patch_torch.monkey_patch_torch_reductions = lambda *a, **k: None
_srt.patch_torch = _patch_torch


# ─── sglang.srt.utils — MultiprocessingSerializer + the alternate patch_torch
# location. Both are part of the engine weight-transfer path, never hit in SFT;
# the import must resolve but the symbols only need to exist. ───
class _Unused:
    """Stub for engine-only classes — importing is fine, using raises loud."""

    def __init__(self, *a, **k):
        raise RuntimeError(
            "SFT stub: engine weight-transfer class instantiated — you're in an "
            "RL/rollout path with only SFT stubs loaded. Build the real env "
            "(build_conda.sh)."
        )


_utils = _stub_module("sglang.srt.utils")
_utils.MultiprocessingSerializer = _Unused
_srt.utils = _utils
_utils_patch = _stub_module("sglang.srt.utils.patch_torch")
_utils_patch.monkey_patch_torch_reductions = lambda *a, **k: None
_utils.patch_torch = _utils_patch

# ─── sglang.srt.weight_sync.tensor_bucket.FlattenedTensorBucket ───
_weight_sync = _stub_module("sglang.srt.weight_sync")
_tensor_bucket = _stub_module("sglang.srt.weight_sync.tensor_bucket")
_tensor_bucket.FlattenedTensorBucket = _Unused
_weight_sync.tensor_bucket = _tensor_bucket
_srt.weight_sync = _weight_sync

# ─── sglang.srt.batch_invariant_ops.enable_batch_invariant_mode (lazy, RL) ───
_biops = _stub_module("sglang.srt.batch_invariant_ops")
_biops.enable_batch_invariant_mode = lambda *a, **k: None
_srt.batch_invariant_ops = _biops

# ─── sglang_router.launch_router.RouterArgs — miles' add_router_arguments calls
# RouterArgs.add_cli_args unconditionally, and RolloutManager._start_router reads
# args.sglang_router_ip / _port / _request_timeout_secs. The real package
# registers these under miles' prefixing (so the bare --sglang-router-ip flag
# isn't recognized); stub add_cli_args to register exactly those flags and
# return the parser. SFT passes --sglang-router-ip <addr> so _start_router
# returns early (no router started) — from_cli_args is then never reached, and
# under --debug-train-only RolloutManager creates no engines. ───
_router = _stub_module("sglang_router")
_launch = _stub_module("sglang_router.launch_router")
_router.launch_router = _launch


class _RouterArgs:
    @staticmethod
    def add_cli_args(parser, *args, **kwargs):  # noqa: ARG004
        parser.add_argument("--sglang-router-ip", type=str, default=None)
        parser.add_argument("--sglang-router-port", type=int, default=None)
        parser.add_argument("--sglang-router-request-timeout-secs", type=int, default=1800)
        return parser


_launch.RouterArgs = _RouterArgs

# ─── miles.backends.sglang_utils.* — these import sglang engine internals ───
# We replace the whole submodules with stubs. SGLangEngine is wrapped in
# ray.remote() inside _create_rollout_engines — SFT never calls that.
_sglang_utils = _stub_module("miles.backends.sglang_utils")
_engine_mod = _stub_module("miles.backends.sglang_utils.sglang_engine")
_args_mod = _stub_module("miles.backends.sglang_utils.arguments")


class _SGLangEngine:
    def __init__(self, *_, **__):
        raise RuntimeError(
            "SGLangEngine stub — you're trying to run RL rollout but only have "
            "the SFT stubs loaded. Build the miles conda env (build_conda.sh)."
        )


_engine_mod.SGLangEngine = _SGLangEngine
_sglang_utils.sglang_engine = _engine_mod


def _add_sglang_arguments(parser):  # noqa: ARG001
    # miles' arg chain does `parser = add_sglang_arguments(parser)`, so this
    # MUST return the parser (the original stub returned None → broke the chain
    # → add_network_arguments got parser=None). No sglang CLI args in SFT.
    return parser


def _sglang_validate_args(args):  # noqa: ARG001
    return args


_args_mod.add_sglang_arguments = _add_sglang_arguments
_args_mod.validate_args = _sglang_validate_args
_sglang_utils.arguments = _args_mod
