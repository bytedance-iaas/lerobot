#!/usr/bin/env python3
"""Cross-node (multi-machine) helper for robot_sft: check / status / clean / launch.

Every cross-node failure we have actually hit came from hand-rolled ssh one-liners:
a redirect that landed on the wrong host, a worker missing the model cache or HF token,
a pre-created output_dir, a master disk that filled during checkpoint save (which surfaces
as an NCCL *timeout on the worker*), and orphaned `setsid` ranks still holding GPU memory
on the next retry. This encodes each of those once, so the agent stops improvising.

Subcommands
  check   all preflight gates for a cross-node run (read-only)
  status  one-line phase per node, from both logs (read-only)
  clean   kill leftover ranks on every node, verify GPU freed, optionally drop output_dir
  launch  start the workers (ssh) + print the master command, recording PIDs for `clean`

Addresses are whatever the user gave (headless DNS preferred over pod IPs). Remote commands
run through `bash -lc` so ~/.bashrc credentials (HF_TOKEN/HF_ENDPOINT) are actually loaded —
a plain non-interactive ssh does NOT source them.

    python multinode.py check --worker <addr> --master <addr> \
        --dataset-root /opt/data/ds --output-dir /opt/data/run1 \
        --require-model lerobot/pi05_base --require-model google/paligemma-3b-pt-224 \
        --steps 10544 --save-freq 1100 --ckpt-gb 9
    python multinode.py status --worker <addr> --output-dir /opt/data/run1
    python multinode.py clean  --worker <addr> --output-dir /opt/data/run1 [--remove-output-dir]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

# LogLevel=ERROR matters: without it ssh prints "Warning: Permanently added … to the list of
# known hosts" (pod host keys change every restart, so it fires constantly) and that line lands
# in the output we parse — it silently corrupted the version compare and the GPU count.
SSH_OPTS = ["-o", "ConnectTimeout=8", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "LogLevel=ERROR"]
TRAIN_PATTERNS = ["accelerate", "lerobot-train", "lerobot_train"]
STATE_FILE = ".multinode_pids.json"   # written into output_dir's parent by `launch`


# --------------------------------------------------------------------------- shell helpers
def _result(p) -> tuple[int, str]:
    """Parse STDOUT only — stderr (ssh notices, tool chatter) must never reach a comparison.
    On failure fall back to stderr so the reason is still visible."""
    out = (p.stdout or "").strip()
    return p.returncode, out if (p.returncode == 0 or out) else (p.stderr or "").strip()


def run_local(cmd: str, timeout: int = 60) -> tuple[int, str]:
    return _result(subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                                  timeout=timeout))


def run_remote(addr: str, cmd: str, timeout: int = 60) -> tuple[int, str]:
    """Run cmd on `addr` under a LOGIN shell so ~/.bashrc creds load.

    The command is fed over STDIN (`bash -l -s`), never as an ssh argument. Passing it as an
    argument puts the text through TWO shell expansions on the remote side — sshd's own login
    shell and then `bash -lc` — so nested quotes get eaten: a flag like
    `--dataset.episodes='[0, 1]'` arrives mangled and lerobot dies in YAML parsing. Over stdin
    there is no argv quoting layer at all, so the text arrives byte-for-byte.
    """
    full = ["ssh", *SSH_OPTS, f"root@{addr}", "bash -l -s"]
    try:
        p = subprocess.run(full, input=cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 255, f"ssh timeout after {timeout}s"
    return _result(p)


def write_remote_file(addr: str, path: str, content: str, timeout: int = 30) -> tuple[int, str]:
    """Create `path` on `addr` with exactly `content` (sent over stdin).

    Only the trivial `cat > path` goes through argv (no nested quotes to mangle); the payload —
    which may contain single quotes, brackets, JSON — rides stdin untouched. This is why the
    launcher ships a SCRIPT instead of a giant one-line command.
    """
    full = ["ssh", *SSH_OPTS, f"root@{addr}", f"cat > {shlex.quote(path)}"]
    try:
        p = subprocess.run(full, input=content, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 255, f"ssh timeout after {timeout}s"
    return _result(p)


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "", ok_detail: str = "") -> bool:
        """`detail` explains a FAILURE — it must never print on a pass (a "[PASS] … not cached"
        line is worse than no line). Use ok_detail for the success annotation."""
        if not ok:
            self.failed += 1
        note = ok_detail if ok else detail
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {note}" if note else ""))
        return ok

    def done(self, what: str) -> int:
        if self.failed:
            print(f"\n{what}: {self.failed} check(s) FAILED — fix these before launching.")
            return 1
        print(f"\n{what}: all checks passed.")
        return 0


def hf_cache_dirname(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


# --------------------------------------------------------------------------- RDMA
# An ACTIVE port is NOT proof an HCA is usable. On a VKE console pod `ibv_devinfo` lists
# SIX mlx5 devices all PORT_ACTIVE, but five of them are the host's physical RoCE NICs whose
# netdevs live in the HOST netns — the pod only gets their /dev/infiniband char devices. Left
# to itself NCCL enumerates all six, picks mlx5_0, reads a GID belonging to the NODE's IP and
# dies with `ibv_modify_qp failed with 19 No such device`. Measured, not theorised.
#
# The discriminator is the GID's backing netdev: gid_attrs/ndevs/<i> names the interface the
# GID is bound to, and only the HCA whose netdev EXISTS IN THIS NETNS can complete a QP. On a
# console pod that is the vRDMA device riding the pod's own ENI (ndev=eth0, GID = the POD's
# IP). Prefer the RoCE v2 IPv4-mapped GID — that is the one that routes.
_RDMA_PROBE = r"""
for d in /sys/class/infiniband/*; do
  [ -e "$d/ports/1/state" ] || continue
  case "$(cat $d/ports/1/state 2>/dev/null)" in *ACTIVE*) ;; *) continue;; esac
  n=$(basename $d); best=""
  for g in $d/ports/1/gids/*; do
    v=$(cat $g 2>/dev/null); i=$(basename $g)
    case "$v" in ""|0000:0000:0000:0000:0000:0000:0000:0000) continue;; esac
    case "$(cat $d/ports/1/gid_attrs/types/$i 2>/dev/null)" in *v2*) ;; *) continue;; esac
    nd=$(cat $d/ports/1/gid_attrs/ndevs/$i 2>/dev/null)
    [ -n "$nd" ] && [ -e "/sys/class/net/$nd" ] || continue   # netdev must be in THIS netns
    case "$v" in *ffff:*) echo "$n $i $nd $v"; best=done; break;; esac  # RoCE v2 IPv4 wins
    [ -z "$best" ] && best="$n $i $nd $v"
  done
  case "$best" in done|"") ;; *) echo "$best";; esac
done
"""


def probe_rdma(addr: str | None) -> list[tuple[str, str, str, str]]:
    """[(hca, gid_index, netdev, gid), ...] for HCAs that can actually establish a QP here."""
    rc, out = (run_local(_RDMA_PROBE) if addr in (None, "local") else run_remote(addr, _RDMA_PROBE))
    found = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].startswith(("mlx", "rocep", "irdma", "bnxt", "hns")):
            found.append(tuple(parts))  # type: ignore[arg-type]
    return found


# --------------------------------------------------------------------------- check
def cmd_rdma(a) -> int:
    """Report the HCA each node can actually use. `--export` prints just this node's, so a
    launch script can do: `eval $(multinode.py rdma --export)` and stay node-correct."""
    if a.export:
        u = probe_rdma(None)
        print(f"NCCL_IB_HCA={u[0][0]}" if u else "", end="\n" if u else "")
        return 0 if u else 1
    ok = True
    for label, addr in [("master", None)] + [(w, w) for w in a.worker]:
        u = probe_rdma(addr)
        if u:
            h, gi, nd, gid = u[0]
            print(f"{label:<28} NCCL_IB_HCA={h}   (gid index {gi} on {nd}, {gid})")
        else:
            ok = False
            print(f"{label:<28} NO USABLE HCA — launch with NCCL_IB_DISABLE=1 (TCP) or fix RDMA")
    if not ok:
        print("\nNote: an ACTIVE port is not enough — the GID must be bound to a netdev inside "
              "the pod's own netns, else NCCL fails with `ibv_modify_qp ... 19 No such device`.")
    return 0 if ok else 1


def cmd_check(a) -> int:
    r = Report()
    hf_home = a.hf_home or os.environ.get("HF_HOME") or "/opt/data/.cache/huggingface"
    hub = f"{hf_home}/hub"
    rdma_hca: dict[str, str] = {}   # node -> its OWN NCCL_IB_HCA (never shared between nodes)

    for w in a.worker:
        print(f"\n=== worker {w} ===")
        # 1. reachable, and it is a DIFFERENT host than the master
        rc, out = run_remote(w, "hostname")
        if not r.check(f"{w}: ssh reachable", rc == 0, out if rc else out):
            continue  # every later check needs ssh

        # 2. reverse path — the rendezvous is bidirectional; a worker that cannot resolve
        #    the master hangs forever instead of erroring.
        rc, out = run_remote(w, f"getent hosts {shlex.quote(a.master)} || true")
        r.check(f"{w}: can resolve master ({a.master})", bool(out.strip()),
                out or "no A record — accelerate rendezvous will hang, not error")

        # 3. env parity: a torch/lerobot mismatch corrupts training subtly rather than loudly
        vcmd = ("cd /lerobot && python -c \"import torch,importlib.metadata as m;"
                "print(torch.__version__, m.version('lerobot'))\"")
        rc_l, ver_l = run_local(vcmd)
        rc_w, ver_w = run_remote(w, vcmd)
        r.check(f"{w}: torch/lerobot match master", rc_l == 0 and rc_w == 0 and ver_l == ver_w,
                f"master={ver_l!r} worker={ver_w!r}")

        # 4. dataset at the SAME path (each rank reads locally; DDP ships batches, not data)
        if a.dataset_root:
            rc, out = run_remote(w, f"ls {shlex.quote(a.dataset_root)}/meta/info.json")
            r.check(f"{w}: dataset at {a.dataset_root}", rc == 0,
                    "missing — the worker dies seconds after launch")

        # 5. model cache + credentials. EVERY rank builds the policy itself, so the backbone
        #    must be cached (or downloadable) on every node; gated repos also need HF_TOKEN.
        for repo in a.require_model:
            rc, out = run_remote(w, f"ls -d {shlex.quote(hub)}/{hf_cache_dirname(repo)}")
            r.check(f"{w}: model cache {repo}", rc == 0,
                    "not cached — a gated repo will 403 mid-launch")
        if a.require_model:
            rc, out = run_remote(w, 'test -n "$HF_TOKEN" && echo yes || echo no')
            r.check(f"{w}: HF_TOKEN visible to a login shell", out.strip().endswith("yes"),
                    "absent — gated repos 403. Note: non-login ssh does NOT read ~/.bashrc")

        # 6. GPU count feeds --num_processes
        rc, out = run_remote(w, "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
        n = out.strip().splitlines()[-1] if out.strip() else "0"
        r.check(f"{w}: GPU count = {n}", n.isdigit() and int(n) > 0, out)

        # 7. GPU actually free — orphaned ranks from a previous attempt hold memory and
        #    make the retry OOM. `clean` fixes this.
        rc, out = run_remote(w, "nvidia-smi --query-gpu=memory.used --format=csv,noheader")
        used = [int(x.split()[0]) for x in out.splitlines() if x.strip() and x.split()[0].isdigit()]
        r.check(f"{w}: GPUs idle", all(u < 1024 for u in used) if used else False,
                f"in use: {out.strip()} — run `multinode.py clean` first" if used else out)

        # 8. RDMA. Gradient all-reduce is the cross-node bottleneck; if these NICs exist NCCL
        #    should use them, but a misconfig makes it fall back to TCP SILENTLY (no error, just
        #    many times slower). Report capability here; only a NCCL_DEBUG=INFO run showing
        #    "NET/IB" proves it is actually used (see SKILL.md M0b).
        rc, out = run_remote(w, "ls /dev/infiniband/uverbs* 2>/dev/null | wc -l")
        n_uverbs = int(out.strip().splitlines()[-1]) if out.strip().split() and out.strip().splitlines()[-1].strip().isdigit() else 0
        if n_uverbs:
            # NOT "is a port ACTIVE" — that is true of HCAs this netns cannot use, and gating on
            # it is what hides the failure until NCCL dies mid-launch. Gate on a USABLE HCA.
            usable = probe_rdma(w)
            r.check(f"{w}: RDMA HCA usable from this netns", bool(usable),
                    f"{n_uverbs} uverbs device(s) present but NONE has a RoCE v2 GID bound to a "
                    f"netdev in this netns — NCCL would pick one anyway and die with "
                    f"`ibv_modify_qp failed with 19 No such device`. Launch WITHOUT RDMA "
                    f"(NCCL_IB_DISABLE=1) or fix the pod's RDMA networking first.",
                    ok_detail=" ".join(f"{h}(gid{gi} on {nd})" for h, gi, nd, _ in usable))
            if usable:
                hca = usable[0][0]
                rdma_hca[w] = hca
                print(f"       -> export NCCL_IB_HCA={hca}   # THIS node only; do not copy to others")
                print(f"          (leave NCCL_IB_GID_INDEX UNSET — the index differs per node and "
                      f"NCCL picks the right GID once the HCA is pinned)")
        else:
            print(f"[INFO] {w}: no /dev/infiniband/uverbs* — no RDMA, NCCL will use TCP")

        # 9. output_dir must NOT exist: lerobot-train aborts on an existing dir without --resume
        if a.output_dir:
            rc, out = run_remote(w, f"test -e {shlex.quote(a.output_dir)} && echo exists || echo clean")
            r.check(f"{w}: output_dir clean", out.strip().endswith("clean"),
                    "exists — delete it on EVERY node or every retry fails identically")

    print("\n=== master ===")
    # 9. checkpoint disk budget — ONLY the master writes checkpoints, and running out mid-save
    #    surfaces as an NCCL timeout on the WORKER, sending you to debug the wrong node.
    if a.output_dir and a.steps and a.save_freq:
        need = (a.steps // max(1, a.save_freq) + 1) * a.ckpt_gb
        rc, out = run_local(f"df -BG --output=avail {shlex.quote(os.path.dirname(a.output_dir))} "
                            "| tail -1 | tr -dc '0-9'")
        avail = int(out) if out.strip().isdigit() else -1
        r.check(f"master: checkpoint disk budget (~{need}G needed, {avail}G free)",
                avail >= need,
                "master fills up mid-save -> looks like a worker NCCL timeout, not a disk error")

    # 10. rendezvous port free on the master
    rc, out = run_local(f"ss -ltn 2>/dev/null | grep -w {a.port} || true")
    r.check(f"master: port {a.port} free", not out.strip(),
            out.strip() or "in use — give this run a distinct --main_process_port")

    if a.output_dir:
        r.check("master: output_dir clean", not os.path.exists(a.output_dir),
                "exists — delete it before launching")

    # 11. master's own usable HCA (same rule as the workers — see probe_rdma).
    if os.path.exists("/dev/infiniband"):
        usable = probe_rdma(None)
        r.check("master: RDMA HCA usable from this netns", bool(usable),
                "no RoCE v2 GID bound to a netdev in this netns — NCCL would pick an unusable "
                "HCA and die with `ibv_modify_qp failed with 19 No such device`",
                ok_detail=" ".join(f"{h}(gid{gi} on {nd})" for h, gi, nd, _ in usable))
        if usable:
            rdma_hca["master"] = usable[0][0]

    if rdma_hca:
        print("\n=== NCCL_IB_HCA (per node — NOT interchangeable) ===")
        for node, hca in rdma_hca.items():
            print(f"  {node:<28} export NCCL_IB_HCA={hca}")
        print("  Put this in EACH node's launch env, leave NCCL_IB_GID_INDEX unset, and PROVE it "
              "with NCCL_DEBUG=INFO: the transport line must read NET/IB and name this HCA.")

    return r.done("check")


# --------------------------------------------------------------------------- status
PHASES = [
    ("first checkpoint written", ("Checkpoint saved", "checkpoints/")),
    ("training steps running", ("it/s]", "loss:")),
    ("training started", ("Start offline training", "Effective batch size")),
    ("building policy/dataset (the long quiet phase)", ("Creating policy", "Creating dataset",
                                                        "Loading", "resolving")),
    ("rendezvous / process group init", ("Distributed environment", "rendezvous", "nproc")),
]


def classify(text: str) -> str:
    for label, needles in PHASES:
        if any(n in text for n in needles):
            return label
    return "no recognizable phase yet"


def tail_of(path: str, addr: str | None, n: int = 40) -> str:
    cmd = f"tail -{n} {shlex.quote(path)} 2>/dev/null || true"
    _, out = (run_remote(addr, cmd) if addr else run_local(cmd))
    return out


def cmd_status(a) -> int:
    for label, addr, log in [("master", None, a.master_log)] + \
                            [(f"worker {w}", w, a.worker_log) for w in a.worker]:
        text = tail_of(log, addr)
        if not text.strip():
            print(f"[{label}] log EMPTY at {log} — if this is a worker, the launch redirect "
                  f"probably ran on the master (quote it: ssh host 'cmd > log 2>&1')")
            continue
        last = [ln for ln in text.splitlines() if ln.strip()][-1][:160]
        print(f"[{label}] {classify(text)}\n          last: {last}")
    print("\n(poll this every ~20-30s and relay it; multi-node startup is minutes of silence)")
    return 0


# --------------------------------------------------------------------------- clean
def cmd_clean(a) -> int:
    r = Report()
    pat = "|".join(TRAIN_PATTERNS)
    kill = (f"pkill -9 -f '{pat}' 2>/dev/null; sleep 3; "
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader")
    targets: list[tuple[str, str | None]] = [("master", None)] + [(w, w) for w in a.worker]

    for label, addr in targets:
        _, out = (run_remote(addr, kill) if addr else run_local(kill))
        used = [int(x.split()[0]) for x in out.splitlines() if x.strip() and x.split()[0].isdigit()]
        # `setsid`-detached ranks outlive the launcher; without this the next attempt OOMs.
        r.check(f"{label}: GPUs released", all(u < 1024 for u in used) if used else True,
                f"still used: {out.strip()} — find stragglers with `pgrep -af python`")

        if a.remove_output_dir and a.output_dir:
            rm = f"rm -rf {shlex.quote(a.output_dir)} && echo removed"
            _, o2 = (run_remote(addr, rm) if addr else run_local(rm))
            r.check(f"{label}: output_dir removed", "removed" in o2, o2)

    return r.done("clean")


# --------------------------------------------------------------------------- launch
def cmd_launch(a) -> int:
    """Start the workers over ssh (logs written ON each worker) and print the master command.

    The master is NOT started here: it must run under the watchdog so the run is supervised.
    """
    if "--machine_rank" not in a.command:
        print("ERROR: --command must be the full accelerate command containing --machine_rank",
              file=sys.stderr)
        return 2
    pids: dict[str, str] = {}
    script_path = a.worker_log.rsplit("/", 1)[0] + "/worker_rank.sh"
    for i, w in enumerate(a.worker, start=1):
        cmd = a.command.replace("--machine_rank=0", f"--machine_rank={i}")
        if cmd == a.command:
            print(f"ERROR: could not set rank for {w}: no literal --machine_rank=0 in --command",
                  file=sys.stderr)
            return 2
        # Ship a SCRIPT, don't inline the command: flags like --dataset.episodes='[0, 1]' carry
        # quotes/brackets that get eaten when the text passes through ssh's argv + a remote shell,
        # and lerobot then dies parsing YAML. The script body travels over stdin verbatim.
        # `-l` so ~/.bashrc creds (HF_TOKEN) load; redirect INSIDE the script so the log is created
        # on the WORKER (a master-side redirect leaves the worker log empty).
        body = f"#!/usr/bin/env bash\nset -x\ncd /lerobot\nexec {cmd}\n"
        rc, out = write_remote_file(w, script_path, body)
        if rc != 0:
            print(f"[worker {w}] ERROR writing {script_path}: {out}", file=sys.stderr)
            return 2
        launch = (f"chmod +x {shlex.quote(script_path)} && "
                  f"setsid nohup bash -l {shlex.quote(script_path)} "
                  f"> {shlex.quote(a.worker_log)} 2>&1 < /dev/null & echo $!")
        rc, out = run_remote(w, launch, timeout=30)
        pid = out.strip().splitlines()[-1] if out.strip() else "?"
        pids[w] = pid
        print(f"[worker {w}] rank={i} started pid={pid} script={script_path} log={a.worker_log}")

    if a.output_dir:
        state = os.path.join(os.path.dirname(a.output_dir) or ".", STATE_FILE)
        try:
            with open(state, "w") as f:
                json.dump({"workers": pids, "worker_log": a.worker_log}, f, indent=2)
            print(f"\nrecorded worker PIDs -> {state}")
        except OSError as e:
            print(f"(could not record PIDs: {e})")

    print("\nNow start the MASTER under the watchdog (rank 0), e.g.:\n"
          f"  {a.command}\n"
          "Then poll:  python multinode.py status --worker <addr> ...")
    return 0


# --------------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--worker", action="append", default=[],
                       help="worker address exactly as the user gave it (repeatable)")
        p.add_argument("--output-dir", default=None)

    c = sub.add_parser("check", help="all cross-node preflight gates (read-only)")
    common(c)
    c.add_argument("--master", required=True, help="master address AS THE WORKERS REACH IT")
    c.add_argument("--dataset-root", default=None)
    c.add_argument("--require-model", action="append", default=[],
                   help="HF repo id whose cache must exist on every node (repeatable)")
    c.add_argument("--hf-home", default=None)
    c.add_argument("--steps", type=int, default=None)
    c.add_argument("--save-freq", type=int, default=None)
    c.add_argument("--ckpt-gb", type=float, default=9.0)
    c.add_argument("--port", type=int, default=29500)

    s = sub.add_parser("status", help="one-line phase per node (read-only)")
    common(s)
    s.add_argument("--master-log", required=True)
    s.add_argument("--worker-log", required=True)

    cl = sub.add_parser("clean", help="kill leftover ranks everywhere, verify GPUs freed")
    common(cl)
    cl.add_argument("--remove-output-dir", action="store_true",
                    help="also delete output_dir on every node (required before a retry)")

    rd = sub.add_parser("rdma", help="print each node's usable NCCL_IB_HCA (read-only)")
    rd.add_argument("--worker", action="append", default=[],
                    help="worker address (repeatable); the master is always probed")
    rd.add_argument("--export", action="store_true",
                    help="emit only `NCCL_IB_HCA=<hca>` for the MASTER, for `eval $(...)`")

    la = sub.add_parser("launch", help="start workers over ssh; master runs under the watchdog")
    common(la)
    la.add_argument("--command", required=True,
                    help="the full rank-0 accelerate command (must contain --machine_rank=0)")
    la.add_argument("--worker-log", default="/opt/data/robot_sft/worker.log")

    a = ap.parse_args()
    return {"check": cmd_check, "status": cmd_status, "rdma": cmd_rdma,
            "clean": cmd_clean, "launch": cmd_launch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
