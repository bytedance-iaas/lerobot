#!/usr/bin/env python3
"""Cross-node (multi-machine) helper for robot_sft: check / status / clean / launch.

Every cross-node failure we have actually hit came from hand-rolled ssh one-liners:
a redirect that landed on the wrong host, a worker missing the model cache or HF token,
a pre-created output_dir, a master disk that filled during checkpoint save (which surfaces
as an NCCL *timeout on the worker*), and orphaned `setsid` ranks still holding GPU memory
on the next retry. This encodes each of those once, so the agent stops improvising.

Subcommands
  env     put a credential (HF_TOKEN…) on every node, visible to every shell
  check   all preflight gates for a cross-node run (read-only)
  status  one-line phase per node, from both logs (read-only)
  clean   kill leftover ranks on every node, verify GPU freed, optionally drop output_dir
  launch  start the workers (ssh) + print the master command, recording PIDs for `clean`

Addresses are whatever the user gave (headless DNS preferred over pod IPs). Remote commands
run through `bash -lc` so the credentials installed by `env set` (HF_TOKEN/HF_ENDPOINT) are
actually loaded — a login shell reads them via /etc/profile.d; a plain non-interactive ssh
reads nothing at all.

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
import pathlib
import re
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
    """Run cmd on `addr` under a LOGIN shell, so `env set` creds load via /etc/profile.d.

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
            # sysfs gid index — printed as evidence only. It is NOT NCCL_IB_GID_INDEX: sysfs is
            # netns-filtered and numbered per pod, while NCCL reads the verbs API, which is not.
            print(f"{label:<28} NCCL_IB_HCA={h}   (sysfs gid[{gi}] on {nd} = {gid}; "
                  f"do NOT pass this as NCCL_IB_GID_INDEX)")
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
                    "absent — gated repos 403. Fix on ALL nodes at once (value stays off argv):\n"
                    "        python multinode.py env set HF_TOKEN --worker <addr> <<'EOF'\n"
                    "        <token>\n        EOF")

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
        if a.output_dir and getattr(a, "resume", False):
            # Inverted on purpose: on a resume the worker MUST have the checkpoint, because it
            # reads it too — see cmd_resume. Only rank 0 ever wrote one.
            _, out = run_remote(w, f"test -f {shlex.quote(a.output_dir)}/checkpoints/last/pretrained_model/"
                                   f"train_config.json && echo ok || echo missing")
            r.check(f"{w}: resume checkpoint present", out.strip().endswith("ok"),
                    f"absent — run `multinode.py resume --worker {w} --output-dir {a.output_dir}` "
                    f"first; every rank reads the checkpoint, only rank 0 writes it")
        elif a.output_dir:
            rc, out = run_remote(w, f"test -e {shlex.quote(a.output_dir)} && echo exists || echo clean")
            r.check(f"{w}: output_dir clean", out.strip().endswith("clean"),
                    "exists — delete it on EVERY node or every retry fails identically")

    print("\n=== master ===")
    # 9. checkpoint disk budget — ONLY the master writes checkpoints, and running out mid-save
    #    surfaces as an NCCL timeout on the WORKER, sending you to debug the wrong node.
    if a.output_dir and a.steps and a.save_freq:
        need = (a.steps // max(1, a.save_freq) + 1) * a.ckpt_gb
        # df needs a path that EXISTS. output_dir is created by the run, and its parent usually
        # does not exist yet either (the whole tree is new), so probing dirname() alone made df
        # error out, `avail` fall back to -1, and the gate fail with a nonsense "-1G free" —
        # a preflight that cries wolf gets ignored, which defeats the point. Walk up to the
        # nearest existing ancestor: that is the filesystem the checkpoints will land on.
        probe = os.path.dirname(os.path.abspath(a.output_dir)) or "/"
        rc, out = run_local(
            f'p={shlex.quote(probe)}; while [ ! -e "$p" ] && [ "$p" != / ]; do p=$(dirname "$p"); done; '
            "df -BG --output=avail \"$p\" | tail -1 | tr -dc '0-9'")
        avail = int(out) if out.strip().isdigit() else -1
        if avail < 0:
            # Unknown != insufficient. Say which one it is instead of reporting a fake number.
            r.check(f"master: checkpoint disk budget (~{need}G needed)", False,
                    f"could not read free space for {probe} (df said: {out.strip() or 'nothing'})")
        else:
            r.check(f"master: checkpoint disk budget (~{need}G needed, {avail}G free)",
                    avail >= need,
                    "master fills up mid-save -> looks like a worker NCCL timeout, not a disk error")

    # 10. rendezvous port free on the master
    rc, out = run_local(f"ss -ltn 2>/dev/null | grep -w {a.port} || true")
    r.check(f"master: port {a.port} free", not out.strip(),
            out.strip() or "in use — give this run a distinct --main_process_port")

    if a.output_dir and getattr(a, "resume", False):
        r.check("master: resume checkpoint present",
                os.path.isfile(os.path.join(a.output_dir, "checkpoints", "last",
                                            "pretrained_model", "train_config.json")),
                "no checkpoints/last — if the run died before the first save_freq step there is "
                "nothing to resume from; delete output_dir and start over")
    elif a.output_dir:
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
    # The watchdog is the one instruction in this whole workflow that was only ever DESCRIBED —
    # SKILL.md says "master under the watchdog" 31 times and `launch` printed a reminder, but
    # nothing ever failed when it was skipped. Everything the agent reliably does here is
    # enforced by an exit code, so this became one too: an unsupervised master means no stall
    # detection and no auto-resume, and you find out hours later.
    rc, out = run_local("pgrep -fa 'watchdog.py' | head -1 || true")
    supervised = bool(out.strip())
    if supervised:
        print(f"[master] watchdog: supervising ({out.strip().splitlines()[-1][:100]})")
    else:
        print("[master] watchdog: ABSENT — this run has no stall detection and no auto-resume.\n"
              "         Start it (session.py add-run + watchdog.py + monitor_server.py) or say\n"
              "         explicitly that this run is deliberately unsupervised.")
    print("\n(poll this every ~20-30s and relay it; multi-node startup is minutes of silence)")
    return 0 if supervised or a.allow_unsupervised else 2


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
        # `-l` so `env set` creds (HF_TOKEN) load via /etc/profile.d; redirect INSIDE the script
        # so the log is created
        # on the WORKER (a master-side redirect leaves the worker log empty).
        # NO `exec`. It replaces the shell with a single program, but the command handed to
        # `launch` is routinely COMPOUND — plan_training.py emits `cd /lerobot && python -u -m
        # …`, so the worker ran `exec cd /lerobot && …` and bash answered `cd: not found`
        # (`cd` is a builtin, there is nothing to exec). The worker died instantly, silently,
        # and the master sat in the rendezvous waiting for a rank that no longer existed — which
        # reads as "the smoke test is slow", on the node that is fine. `set -x` puts the failure
        # in the worker log; without exec, a compound command just works.
        body = (f"#!/usr/bin/env bash\nset -x\ncd /lerobot\n{cmd}\n"
                f"echo \"[worker] exited rc=$?\"\n")
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

    # NOT "e.g." — the master must be started under the watchdog, because the watchdog OWNS the
    # training subprocess (starting it afterwards would launch a SECOND master and collide on the
    # rendezvous port). Workers are up and waiting in the rendezvous from this moment, so this is
    # the step that must not be improvised.
    print("\n=== next: start the MASTER (rank 0) UNDER THE WATCHDOG ===")
    print("  The watchdog launches the process itself — do not start training first and attach\n"
          "  the watchdog after; that starts a second master and collides on the rendezvous port.")
    print(f"    1. session.py add-run   (record the run + its multi_node.master_launch_command)\n"
          f"    2. watchdog.py --run <run_id>   -> it runs:\n"
          f"         {a.command}\n"
          f"    3. monitor_server.py    (dashboard)")
    print("  Then poll:  python multinode.py status --worker <addr> ...\n"
          "  `status` FAILS (exit 2) while no watchdog is running — that is deliberate.")
    return 0


# --------------------------------------------------------------------------- cli
# ------------------------------------------------------------------------------- env sharing
# Credentials (HF_TOKEN above all) have to be visible to EVERY shell on EVERY node, and getting
# there by hand is where multi-node setup keeps dying. Two independent failures, both real:
#
#   1. Getting the value in. Inlining it — `ssh w "export HF_TOKEN=$T"`, or appending to a
#      remote rc file with the value in the command — puts it through two shell expansions and
#      into argv, history and every approval prompt. One awkward character and it becomes
#      `not a valid identifier` or a Python parse error. Here the value NEVER touches a command
#      line: it arrives on stdin and ships to the workers on stdin (write_remote_file).
#   2. Making later commands see it. `~/.bashrc` is read by INTERACTIVE shells and
#      `/etc/profile.d` by LOGIN shells — but the agent's own tool calls are plain
#      `bash -c`, which reads NEITHER. That is why "I exported it" is followed by a command
#      that cannot see it. The only hook a non-interactive bash honours is $BASH_ENV.
#
# So: one file, two hooks — profile.d for login shells (what run_remote uses), $BASH_ENV for
# non-interactive ones. ENV_FILE sits on the PVC, so it also survives a pod restart.
ENV_FILE = "/opt/data/.console-env.sh"
PROFILE_HOOK = "/etc/profile.d/10-console-env.sh"
# `. file` only — no logic. This is sourced by every single bash on the node, so anything that
# can fail here breaks every command on the box.
HOOK_BODY = f'[ -r {ENV_FILE} ] && . {ENV_FILE}\n'


def _parse_env_file(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^export ([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
        if m:
            try:
                out[m.group(1)] = " ".join(shlex.split(m.group(2)))
            except ValueError:
                pass
    return out


def _render_env_file(env: dict) -> str:
    head = ("# Managed by `multinode.py env`. Sourced by EVERY shell on this node (profile.d for\n"
            "# login shells, $BASH_ENV for non-interactive ones). Exports only — no logic.\n")
    return head + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(env.items()))


def _mask(v: str) -> str:
    return f"{v[:4]}…{v[-4:]} ({len(v)} chars)" if len(v) > 12 else f"({len(v)} chars)"


# Login-shell order is /etc/profile -> profile.d -> ~/.profile -> ~/.bashrc, so ANY leftover
# `export HF_TOKEN=…` in an rc file runs AFTER our hook and silently WINS. That is not
# hypothetical: a worker was found serving a 3-character HF_TOKEN from two hand-edited rc files
# while the master had none — every `test -n "$HF_TOKEN"` check passed and the run 403s hours in.
# So each node is swept before the file is installed. Commented out, not deleted: the line stays
# visible for anyone wondering where their token went, and it is trivially reversible.
RC_FILES = ["/root/.bashrc", "/root/.profile", "/opt/data/.bashrc", "/opt/data/.profile"]


def _shadow_sweep(addr, name: str) -> str:
    """Comment out competing `export NAME=` lines in rc files on one node. Returns a report."""
    cmd = (
        f'for f in {" ".join(RC_FILES)}; do [ -f "$f" ] || continue; '
        f'n=$(grep -cE "^[[:space:]]*export {name}=" "$f" 2>/dev/null || true); '
        f'[ "${{n:-0}}" -gt 0 ] || continue; '
        f'sed -i -E "s|^([[:space:]]*export {name}=)|# [multinode env] shadowed, disabled: \\1|" "$f"; '
        f'echo "$f($n)"; done')
    rc, out = (run_local(cmd) if addr is None else run_remote(addr, cmd))
    return " ".join(out.split()) if rc == 0 else f"sweep failed: {out}"


def _install(addr, env_text):
    """Write the env file + the login-shell hook on one node. addr=None means locally."""
    if addr is None:
        pathlib.Path(ENV_FILE).write_text(env_text)
        os.chmod(ENV_FILE, 0o600)
        pathlib.Path(PROFILE_HOOK).write_text(HOOK_BODY)
        return 0, ""
    rc, out = write_remote_file(addr, ENV_FILE, env_text)
    if rc:
        return rc, out
    rc, out = write_remote_file(addr, PROFILE_HOOK, HOOK_BODY)
    if rc:
        return rc, out
    return run_remote(addr, f"chmod 600 {ENV_FILE}")


def cmd_env(a) -> int:
    nodes = [None] + list(a.worker)          # None == this node

    if a.action == "show":
        for n in nodes:
            label = "master" if n is None else n
            # -l: a login shell, i.e. the same way run_remote and the launcher see it.
            rc, out = (run_local if n is None else (lambda c, w=n: run_remote(w, c)))(
                f'for v in {" ".join(a.name or ["HF_TOKEN"])}; do '
                f'eval "x=\\${{$v:-}}"; printf "%s=%s(chars) " "$v" "${{#x}}"; done; echo')
            print(f"  {label}: {out if rc == 0 else 'UNREACHABLE ' + out}")
        return 0

    if a.action == "unset":
        local = pathlib.Path(ENV_FILE)
        env = _parse_env_file(local.read_text()) if local.exists() else {}
        if a.name[0] not in env:
            print(f"  {a.name[0]} not in {ENV_FILE} — nothing to remove locally")
        env.pop(a.name[0], None)
        text = _render_env_file(env)
        for n in nodes:
            rc, out = _install(n, text)
            print(f"  {'master' if n is None else n}: {'ok' if rc == 0 else 'FAILED ' + out}")
        print(f"\nremoved {a.name[0]}. A value set OUTSIDE this file (~/.bashrc, /root/.bashrc,\n"
              "the container env) is NOT touched — check with `env show` and clear it by hand.")
        return 0

    # ---- set: value arrives on STDIN, never in argv
    if sys.stdin.isatty():
        print("value must arrive on stdin, e.g.:\n"
              f"  python multinode.py env set {a.name[0]} --worker <addr> <<'EOF'\n"
              "  <value>\n  EOF", file=sys.stderr)
        return 2
    value = sys.stdin.read().strip("\n").strip()
    if not value:
        print("empty value on stdin — refusing", file=sys.stderr)
        return 2

    local = pathlib.Path(ENV_FILE)
    env = _parse_env_file(local.read_text()) if local.exists() else {}
    env[a.name[0]] = value
    text = _render_env_file(env)

    failed = 0
    for n in nodes:
        label = "master" if n is None else n
        shadowed = _shadow_sweep(n, a.name[0])
        rc, out = _install(n, text)
        print(f"  {label}: {'ok' if rc == 0 else 'FAILED ' + out}"
              + (f"  [disabled shadowing exports in {shadowed}]" if shadowed else ""))
        failed += rc != 0

    print(f"\n{a.name[0]} = {_mask(value)}  ->  {ENV_FILE} on {len(nodes)} node(s)")
    # Verify the way it will actually be consumed, and compare the LENGTH. A non-empty test is
    # useless here: it passes for a stale value that overrode ours, which is the exact failure
    # this command exists to end.
    want = len(value)
    for n in nodes:
        label = "master" if n is None else n
        rc, out = (run_local if n is None else (lambda c, w=n: run_remote(w, c)))(
            f'echo ${{#{a.name[0]}}}')
        got = out.strip().splitlines()[-1] if out.strip() else "?"
        ok = got == str(want)
        failed += not ok
        print(f"  login shell on {label}: {got} chars"
              + ("" if ok else f"  <-- MISMATCH, expected {want}: something else still sets "
                               f"{a.name[0]} on this node (grep the rc files)"))
    if not os.environ.get("BASH_ENV"):
        print("\nNOTE: BASH_ENV is unset in this container, so plain `bash -c` still cannot see it.\n"
              f"      Until the next image rollout, prefix one-off commands with `bash -lc`.\n"
              f"      (multinode.py / launch already use login shells, so training is unaffected.)")
    return 1 if failed else 0


# ---------------------------------------------------------------------------- multi-node resume
# Resuming a MULTI-NODE run needs one thing single-node resume does not: the checkpoint has to
# exist on every node. lerobot writes it from rank 0 only —
#     if cfg.save_checkpoint and is_saving_step:
#         if is_main_process: save_checkpoint(...)
# — but reads it from EVERY rank on the way back in:
#     if cfg.resume and step > 0:
#         saved_num_processes = load_training_num_processes(cfg.checkpoint_path)
# and every rank builds the policy from `--config_path`. So a worker whose output_dir was never
# written to dies immediately, while the master looks fine. (This is what the image bakes rsync
# and pod-to-pod ssh for; nothing had actually used it.)
#
# Only `last` and its target are copied, not the whole checkpoints/ tree: for pi05 each one is
# ~9 GB and the workers need exactly the one being resumed from.
def _ckpt_paths(output_dir: str) -> tuple[str, str]:
    return os.path.join(output_dir, "checkpoints"), os.path.join(output_dir, "checkpoints", "last")


def cmd_resume(a) -> int:
    r = Report()
    ck_dir, last = _ckpt_paths(a.output_dir)

    # 1. the master must actually have something to resume FROM.
    rc, out = run_local(f"readlink {shlex.quote(last)} || true")
    target = out.strip().splitlines()[-1] if out.strip() else ""
    if not r.check("master: checkpoints/last exists", bool(target),
                   f"no {last} — nothing to resume from. If the run died before the first "
                   f"save_freq step there IS no checkpoint: delete output_dir and start over."):
        return 1
    step_dir = os.path.normpath(os.path.join(ck_dir, target))   # `last` is a RELATIVE symlink
    cfg_json = os.path.join(step_dir, "pretrained_model", "train_config.json")
    rc, out = run_local(f"test -f {shlex.quote(cfg_json)} && echo ok || echo missing")
    if not r.check(f"master: {os.path.basename(step_dir)} complete", out.strip().endswith("ok"),
                   f"{cfg_json} missing — the checkpoint was interrupted mid-write; "
                   f"use the previous one under {ck_dir}/"):
        return 1
    rc, out = run_local(f"du -sh {shlex.quote(step_dir)} | cut -f1")
    size = out.strip().splitlines()[-1] if out.strip() else "?"
    print(f"[INFO] resuming from {step_dir} ({size})")

    # 2. put it on every worker, byte-identical, with the `last` symlink recreated there.
    for w in a.worker:
        rc, _ = run_remote(w, f"mkdir -p {shlex.quote(ck_dir)}")
        if not r.check(f"{w}: checkpoints dir", rc == 0):
            continue
        # -a keeps perms/times, --delete makes a retry idempotent rather than merging two
        # half-copies. Errors are surfaced: a partial checkpoint fails LATER, on every rank.
        cmd = (f"rsync -a --delete -e {shlex.quote(' '.join(['ssh', *SSH_OPTS]))} "
               f"{shlex.quote(step_dir + '/')} root@{w}:{shlex.quote(step_dir)}/")
        print(f"[INFO] {w}: copying {size} …")
        rc, out = run_local(cmd, timeout=a.timeout)
        if not r.check(f"{w}: checkpoint copied", rc == 0, out):
            continue
        rc, out = run_remote(w, f"cd {shlex.quote(ck_dir)} && ln -sfn {shlex.quote(target)} last && readlink last")
        r.check(f"{w}: last -> {target}", out.strip().endswith(target), out)
        rc, out = run_remote(w, f"test -f {shlex.quote(cfg_json)} && echo ok || echo missing")
        r.check(f"{w}: train_config.json present", out.strip().endswith("ok"),
                "the copy did not land — every rank reads this on resume")

    print("\n=== resume command (same run_id, same output_dir — NOT a new run) ===")
    print(f"  add to the rank-0 command:  --resume=true "
          f"--config_path={cfg_json}")
    print("  workers: identical, only --machine_rank differs")
    return r.done("resume")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--worker", action="append", default=[],
                       help="worker address exactly as the user gave it (repeatable)")
        p.add_argument("--output-dir", default=None)

    c = sub.add_parser("check", help="all cross-node preflight gates (read-only)")
    c.add_argument("--resume", action="store_true",
                   help="this is a RESUME: output_dir must exist (with a checkpoint) on every "
                        "node, instead of being absent")
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
    s.add_argument("--allow-unsupervised", action="store_true",
                   help="exit 0 even with no watchdog (only when the user chose that on purpose)")
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

    ev = sub.add_parser("env", help="share credentials (HF_TOKEN…) across every node + shell")
    ev.add_argument("action", choices=["set", "show", "unset"])
    ev.add_argument("name", nargs="*", help="variable name(s); `set` takes exactly one")
    ev.add_argument("--worker", action="append", default=[],
                    help="worker address (repeatable); the master is always included")

    rs = sub.add_parser("resume", help="copy the master's checkpoint to every worker, then print "
                                      "the resume command (multi-node resume needs it on ALL nodes)")
    rs.add_argument("--worker", action="append", default=[], required=True,
                    help="worker address (repeatable)")
    rs.add_argument("--output-dir", required=True, help="the ORIGINAL run's output_dir")
    rs.add_argument("--timeout", type=int, default=1800,
                    help="seconds allowed for one checkpoint copy (default 1800; ~9GB for pi05)")

    la = sub.add_parser("launch", help="start workers over ssh; master runs under the watchdog")
    common(la)
    la.add_argument("--command", required=True,
                    help="the full rank-0 accelerate command (must contain --machine_rank=0)")
    la.add_argument("--worker-log", default="/opt/data/robot_sft/worker.log")

    a = ap.parse_args()
    if a.cmd == "env" and a.action in ("set", "unset") and len(a.name) != 1:
        ap.error(f"`env {a.action}` takes exactly one variable name")
    return {"check": cmd_check, "status": cmd_status, "rdma": cmd_rdma, "env": cmd_env,
            "resume": cmd_resume, "clean": cmd_clean, "launch": cmd_launch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
