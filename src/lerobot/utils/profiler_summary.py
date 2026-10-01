#!/usr/bin/env python
"""Turn an Ascend profiler output directory into an MFU / overlap summary.

Used by `lerobot_train.py` to append a summary when LEROBOT_PROF=1, and by
`benchmarks/prof_flops.py` as a CLI. The FLOP formulas live here so the two cannot drift.

Two numbers come out of one profile:

* **MFU** -- model FLOPs are computed from the per-kernel `Input Shapes`/`Output Shapes`
  that `kernel_details.csv` records. This works with the fused NPU kernels on, which
  `torch.utils.flop_counter` cannot do, and it is not inflated by the cube's alignment
  padding the way the `aic_cube_fops` PMU counter is. Validated against
  `benchmarks/mfu.py` on Pi0.5/950PR: 8.195411345 vs 8.195414230 TFLOP/sample (3e-7).

* **compute/communication overlap** -- from `step_trace_time.csv`, as
  `Overlapped / (Overlapped + Communication(Not Overlapped))`.

The step time used for MFU is deliberately taken from UNPROFILED steps: the profiler
perturbs the step, so MFU measured inside the profiled window would understate the real
one. The caller passes the step time it measured outside the window.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

TFLOP = 1e12
US_PER_MS = 1000.0
# 950PR bf16, measured manually with GEMM operations. Override per machine.
DEFAULT_PEAK_TFLOPS = 368.6

CUBE_TYPES = frozenset(
    {
        "MatMulV3",
        "BatchMatMul",
        "BatchMatMulV2",
        "FlashAttentionScore",
        "FlashAttentionScoreGrad",
        "Conv2DV2",
        "Conv3DBackpropFilterV2",
    }
)


def dims(field: str | None) -> list[list[int]]:
    """Parse a shape field: '"4096,1152;1152,1152;1152"' -> [[4096,1152],[1152,1152],[1152]].

    FlashAttention leaves most of its optional inputs blank; those become [].
    """
    out: list[list[int]] = []
    for part in (field or "").strip().strip('"').split(";"):
        part = part.strip()
        if not part:
            out.append([])
            continue
        try:
            out.append([int(x) for x in part.split(",")])
        except ValueError:
            out.append([])
    return out


def model_flops(row: dict) -> float:
    """Mathematical FLOPs of one kernel from its shapes -- NOT what the cube issued.

    This is the MFU numerator: it ignores the cube's 16-alignment padding, and for
    FlashAttention it counts the matmuls the eager path would do, so it is comparable with
    benchmarks/mfu.py. Returns 0.0 for kernels with no matmul.
    """
    t = row.get("Type", "")
    ins = dims(row.get("Input Shapes"))
    outs = dims(row.get("Output Shapes"))

    if t in ("MatMulV3", "BatchMatMul", "BatchMatMulV2"):
        # in[0] = "...,M,K", B is stored transposed, out[0] = "...,M,N".
        if ins and outs and len(ins[0]) >= 2 and len(outs[0]) >= 2:
            m, k = ins[0][-2], ins[0][-1]
            n = outs[0][-1]
            batch = 1
            for d in outs[0][:-2]:
                batch *= d
            return 2.0 * batch * m * k * n
        return 0.0

    if t in ("FlashAttentionScore", "FlashAttentionScoreGrad"):
        # in[0] = query, B,N,S,D. Forward is two matmuls (QK^T and AV) = 4*B*N*S^2*D.
        if ins and len(ins[0]) == 4:
            b, n, s, d = ins[0]
            fwd = 4.0 * b * n * s * s * d
            # Backward computes dV, dP, dQ, dK -- four matmuls, i.e. 2x forward. The fused
            # kernel ALSO recomputes S=QK^T (a fifth matmul, 2.5x) because it does not keep
            # the score matrix. That recompute is hardware work, not model work -- the same
            # distinction as gradient checkpointing -- so it is excluded here and shows up
            # as the ~1.43 hardware/model ratio on FlashAttentionScoreGrad.
            return fwd * 2.0 if t.endswith("Grad") else fwd
        return 0.0

    if t == "Conv2DV2":
        # out = N,Cout,Ho,Wo ; in[1] = Cout,Cin,kh,kw
        if outs and len(outs[0]) == 4 and len(ins) > 1 and len(ins[1]) == 4:
            n, cout, ho, wo = outs[0]
            cin, kh, kw = ins[1][1], ins[1][2], ins[1][3]
            return 2.0 * n * cout * ho * wo * cin * kh * kw
        return 0.0

    if t == "Conv3DBackpropFilterV2":
        # out = Cout,Cin,kd,kh,kw ; in[0] = N,Cin,D,H,W
        if outs and len(outs[0]) == 5 and ins and len(ins[0]) == 5:
            cout, cin = outs[0][0], outs[0][1]
            n, _, dd, hh, ww = ins[0]
            return 2.0 * cout * cin * n * dd * hh * ww
        return 0.0

    return 0.0


def _num(row: dict, key: str) -> float:
    try:
        return float((row.get(key) or "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _column(row: dict[str, str], *needles: str) -> float:
    """Fetch a step_trace_time column by name, tolerating header drift.

    Exact match must be tried first: "Overlapped" is a substring of
    "Communication(Not Overlapped)", which precedes it in the real header, so a
    substring-only lookup silently returns the wrong column.
    """
    norm = {k.lower().replace(" ", ""): k for k in row}
    key = norm.get("".join(needles))
    if key is None:
        key = next((norm[n] for n in norm if all(x in n for x in needles)), None)
    if key is None:
        return 0.0
    try:
        return float(row[key])
    except (ValueError, TypeError):
        return 0.0


@dataclass(frozen=True)
class ProfSummary:
    n_steps: int
    model_tflop_per_step: float
    hw_tflop_per_step: float | None
    unmodelled: tuple[str, ...]
    computing_ms: float
    exposed_ms: float
    overlapped_ms: float
    free_ms: float

    @property
    def total_comm_ms(self) -> float:
        return self.overlapped_ms + self.exposed_ms

    @property
    def overlap_pct(self) -> float | None:
        return 100.0 * self.overlapped_ms / self.total_comm_ms if self.total_comm_ms else None


def find_output_dir(root: Path) -> Path | None:
    if root.name == "ASCEND_PROFILER_OUTPUT":
        return root
    hits = sorted(root.rglob("ASCEND_PROFILER_OUTPUT"))
    return hits[0] if hits else None


def read(out_dir: Path) -> ProfSummary | None:
    """Read kernel_details.csv and step_trace_time.csv. None if neither is usable."""
    kd = out_dir / "kernel_details.csv"
    stt = out_dir / "step_trace_time.csv"

    n_steps = 0
    model = hw = 0.0
    unmodelled: set[str] = set()
    have_hw = False
    if kd.exists():
        with kd.open() as fh:
            rows = list(csv.DictReader(fh))
        steps = sorted({(r.get("Step Id") or "").strip() for r in rows} - {""})
        sel = [r for r in rows if (r.get("Step Id") or "").strip() in steps] if steps else rows
        n_steps = len(steps) or 1
        have_hw = bool(rows) and "aic_cube_fops" in rows[0]
        for r in sel:
            f = model_flops(r)
            model += f
            if have_hw:
                hw += _num(r, "aic_cube_fops")
            if f == 0.0 and r.get("Type") in CUBE_TYPES:
                unmodelled.add(r["Type"])

    computing = exposed = overlapped = free = 0.0
    n_trace = 0
    if stt.exists():
        with stt.open() as fh:
            trace = [r for r in csv.DictReader(fh) if any((v or "").strip() for v in r.values())]
        for r in trace:
            computing += _column(r, "computing")
            exposed += _column(r, "communication(notoverlapped)")
            overlapped += _column(r, "overlapped")
            free += _column(r, "free")
        n_trace = len(trace)

    if not n_steps and not n_trace:
        return None
    ns = n_steps or n_trace or 1
    nt = n_trace or 1
    return ProfSummary(
        n_steps=ns,
        model_tflop_per_step=model / TFLOP / ns,
        hw_tflop_per_step=(hw / TFLOP / ns) if have_hw else None,
        unmodelled=tuple(sorted(unmodelled)),
        computing_ms=computing / US_PER_MS / nt,
        exposed_ms=exposed / US_PER_MS / nt,
        overlapped_ms=overlapped / US_PER_MS / nt,
        free_ms=free / US_PER_MS / nt,
    )


def format_lines(
    s: ProfSummary,
    *,
    step_s: float | None,
    step_s_label: str,
    batch_size: int,
    num_processes: int,
    peak_tflops: float,
) -> list[str]:
    """Human-readable summary. Every number carries the caveat that qualifies it."""
    out = [
        f"profiler summary ({s.n_steps} profiled step(s), rank-local)",
        f"  model FLOPs      : {s.model_tflop_per_step:.3f} TFLOP/step/rank"
        f"  ({s.model_tflop_per_step / batch_size:.3f} per sample, from kernel shapes)",
    ]
    if s.hw_tflop_per_step:
        out.append(
            f"  hardware FLOPs   : {s.hw_tflop_per_step:.3f} TFLOP/step/rank"
            f"  ({s.hw_tflop_per_step / s.model_tflop_per_step:.3f}x model -> HFU numerator)"
        )
    if s.unmodelled:
        out.append(
            f"  WARNING: no shape formula for {', '.join(s.unmodelled)}; model FLOPs is an UNDERCOUNT"
        )

    if step_s and step_s > 0:
        achieved = s.model_tflop_per_step / step_s
        mfu = 100.0 * achieved / peak_tflops
        out += [
            f"  step time        : {step_s:.3f} s  ({step_s_label})",
            f"  throughput       : {batch_size * num_processes / step_s:.1f} samples/s global"
            f"  ({batch_size / step_s:.2f} per rank)",
            f"  achieved         : {achieved:.1f} TFLOP/s per card",
            f"  MFU              : {mfu:.1f} %  of {peak_tflops:.1f} TFLOP/s per card",
        ]
    else:
        out.append("  MFU              : unavailable (no unprofiled step time was measured)")

    if s.total_comm_ms:
        out += [
            f"  collectives      : {s.total_comm_ms:.1f} ms/step, "
            f"{s.overlap_pct:.0f} % hidden behind compute",
            f"  exposed comm     : {s.exposed_ms:.1f} ms/step on the critical path",
        ]
    else:
        out.append("  collectives      : none (single process, or no HCCL in the profiled window)")
    out.append(
        f"  device phases    : computing {s.computing_ms:.1f} ms, exposed comm {s.exposed_ms:.1f} ms, "
        f"free {s.free_ms:.1f} ms"
    )
    out.append(
        "  note: FLOPs and the phase split come from the PROFILED steps, which the profiler "
        "slows down; the step time above is from unprofiled steps, so MFU reflects the real run."
    )
    return out


def summarise(
    prof_dir: str | Path,
    *,
    step_s: float | None,
    step_s_label: str,
    batch_size: int,
    num_processes: int,
    peak_tflops: float = DEFAULT_PEAK_TFLOPS,
) -> list[str]:
    """Convenience wrapper: locate, read and format. Returns [] if there is nothing to read."""
    out_dir = find_output_dir(Path(prof_dir))
    if out_dir is None:
        return []
    s = read(out_dir)
    if s is None:
        return []
    return format_lines(
        s,
        step_s=step_s,
        step_s_label=step_s_label,
        batch_size=batch_size,
        num_processes=num_processes,
        peak_tflops=peak_tflops,
    )
