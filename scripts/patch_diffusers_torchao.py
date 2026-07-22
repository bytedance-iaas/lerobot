#!/usr/bin/env python3
"""Make diffusers' torchao safe-globals import resilient to torchao's dtype-module refactor.

diffusers <0.36 hardcodes torchao internal module paths (torchao.dtypes.floatx.float8_layout,
torchao.dtypes.uintx.{uint4,uintx}_layout) that torchao >=0.14 removed, so at import time the
whole try-block aborts with "Unable to import `torchao` Tensor objects" and registers NONE of the
available Tensor globals. lerobot pins diffusers <0.36 but we need torchao 0.17 for fp8 training,
so patch diffusers to import each Tensor optionally. Functionally harmless before (DiffusionPolicy
worked), this just removes the scary warning and registers the globals that DO exist.

Idempotent; a no-op if already patched or the target block isn't present (diffusers changed).
"""
import importlib.util
import sys

spec = importlib.util.find_spec("diffusers.quantizers.torchao.torchao_quantizer")
if spec is None or not spec.origin:
    print("  diffusers torchao quantizer not found — nothing to patch"); sys.exit(0)
path = spec.origin
src = open(path).read()
if "robot_sft: torchao dtype-module refactor" in src:
    print("  already patched"); sys.exit(0)

start = src.find("    try:\n        from torchao.dtypes import NF4Tensor")
anchor = "torch.serialization.add_safe_globals(safe_globals=safe_globals)"
apos = src.find(anchor, start) if start != -1 else -1
if start == -1 or apos == -1:
    print("  target block not found (diffusers changed?) — skipping"); sys.exit(0)
end = src.find("\n", apos) + 1   # consume the whole finally-body line too

new = (
    "    # robot_sft: torchao dtype-module refactor (>=0.14 moved floatx/uintx layouts). Import\n"
    "    # each optionally so the Tensor globals that DO exist still register — no crash, no warning.\n"
    "    import importlib as _il\n"
    "    for _mod, _name in (\n"
    '        ("torchao.dtypes", "NF4Tensor"),\n'
    '        ("torchao.dtypes.floatx.float8_layout", "Float8AQTTensorImpl"),\n'
    '        ("torchao.dtypes.uintx.uint4_layout", "UInt4Tensor"),\n'
    '        ("torchao.dtypes.uintx.uintx_layout", "UintxAQTTensorImpl"),\n'
    '        ("torchao.dtypes.uintx.uintx_layout", "UintxTensor"),\n'
    "    ):\n"
    "        try:\n"
    "            safe_globals.append(getattr(_il.import_module(_mod), _name))\n"
    "        except Exception:\n"
    "            pass\n"
    "    torch.serialization.add_safe_globals(safe_globals=safe_globals)\n"
)
open(path, "w").write(src[:start] + new + src[end:])
print(f"  patched {path}")
