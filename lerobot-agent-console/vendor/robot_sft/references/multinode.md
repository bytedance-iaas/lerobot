# 跨机训练 (multi-node / cross-machine distributed)

Read this in full BEFORE touching a cross-node run. Every rule here is a failure that already
happened once; none of it is derivable from the single-node path.

Only when the user **explicitly asks** for cross-node / multi-machine / 跨机 training (2+ GPU
boxes). lerobot-train runs under HF **accelerate**; multi-node is DDP/FSDP across machines. robot_sft
drives the **master** node normally (plan → preflight → watchdog + monitors + eval); the **worker**
nodes are started **manually by the user** — by design, even when they are ssh-reachable console
pods (the watchdog can only supervise the local master process). See
https://huggingface.co/docs/lerobot/en/torch_accelerators (and `multi_gpu_training.mdx`).

**Order of operations:** ask for the addresses → **Phase M0 (communication check, HARD GATE)** →
plan (`plan_training.py` with cross-node totals) → persist the `multi_node` block → emit per-node
commands → **multi-node smoke test (HARD GATE)** → real run (master under watchdog, workers by hand).

**Preconditions — enforce these and tell the user BEFORE launching:**
1. **Dataset must already be on EVERY node at the SAME path.** Each rank's dataloader reads frames
   locally (DDP only shards the batch, it does not ship data). **We require the user to prepare the
   dataset on the other node(s) themselves** — ask them to confirm it's present at the same
   `--dataset.root` / repo path on every node before you emit any command.
2. **Ask the user for — never guess any of it:** (a) **how to address each worker node** (an IP /
   hostname, or, if it is a pod, its **pod name + its headless service name**), (b) the **master's
   address as reachable from the workers**, (c) **each node's GPU count** (assume equal GPUs/node —
   the simple case; unequal needs a per-node accelerate config file, flag that). Same lerobot
   **image/env** on all nodes; the **`--main_process_port` (default 29500) must be open**
   master↔workers.
   ⚠️ **There is no default worker.** The worker is whatever machine/pod the user names — it is NOT
   assumed to be any particular pod (e.g. not "the test pod"). Ask, echo back what you understood,
   and use exactly that address everywhere (commands, scp, `multi_node` block).

**Node addressing + ssh.** Whatever the user names as a node, address it exactly as they gave it.
When a node **is a pod running this same console image**, cluster networking is already wired —
verified: DNS resolves, TCP connects, and `ssh`/`scp` are **passwordless** (the image bakes a shared
keypair + `authorized_keys` + sshd with `StrictHostKeyChecking=no`; pods from the SAME image trust
each other automatically). A pod running a *different* image has no shared key — treat it like a
plain node.
- **Address pods by headless DNS, NOT pod IP** — pod IPs change on every restart/rollout. The form is
  `<pod-name>.<headless-service-name>` (+`.default.svc.cluster.local` when persisting it), where BOTH
  parts come from the user. ⚠️ A **bare pod name does NOT resolve** — the service segment is
  mandatory. Use this DNS name for `--main_process_ip` so the command survives pod restarts.
  (In this cluster the console pods happen to be `lerobot-console-0.lerobot-console` and
  `lerobot-console-test-0.lerobot-console-test` — an illustration of the FORM only, never an
  assumption about which node the user is using.)
- **Verify the address before building any command:** `getent hosts <dns-name>` (or
  `ssh root@<dns-name> hostname`). If it doesn't resolve, ask the user again — do not guess a name.
- Use ssh to **verify the worker's dataset yourself** instead of asking the user to go do it:
  `ssh root@<worker-dns> 'ls <dataset_root>/meta/info.json'`.
- **Still start the workers MANUALLY** (give the user the per-node command; see below). Do NOT
  ssh-dispatch long-running training onto another node — the watchdog can't supervise a remote
  process, and orphaned runs/logs there are invisible to this session.
- On **non-console** nodes (plain machines), assume no shared keys: the user runs the commands and
  the scp themselves, addressed by IP.

### Phase M0 — 多机通信校验 (HARD GATE: run this BEFORE planning or launching anything)
Cross-node runs fail in ways that look like a hang, not an error (rendezvous never completes, NCCL
silently retries, a worker reads a dataset that isn't there). Verify connectivity **first**, cheaply,
and only then plan the run. Run every check from the **master**; when the nodes are pods on this same
image you can do all of it yourself over ssh — otherwise hand the worker-side ones to the user.
Report each check's result; **do not proceed while any of them fails.**

**Use `scripts/multinode.py` — do NOT hand-roll ssh one-liners.** Every cross-node failure we have
actually hit came from improvised ssh (a redirect that landed on the wrong host, a missing model
cache, a pre-created output_dir, orphaned ranks holding GPUs). The script encodes each gate once:
```bash
python scripts/multinode.py check \
  --master <master-addr> --worker <worker-addr> [--worker <addr2> …] \
  --dataset-root <dataset_root> --output-dir <output_dir> \
  --require-model <policy repo id> [--require-model <backbone repo id> …] \
  --steps <steps> --save-freq <save_freq> --ckpt-gb <size of one checkpoint>
```
It verifies, per node: ssh reachable · **worker can resolve the master** (rendezvous is
bidirectional — one-way reachability hangs forever instead of erroring) · torch+lerobot versions
match · dataset at the same path · **model cache + `HF_TOKEN`** · GPU count · **GPUs actually idle**
· output_dir absent; and on the master: **checkpoint disk budget**, rendezvous port free.
`--require-model` takes the policy AND its backbone (e.g. `lerobot/pi05_base` and
`google/paligemma-3b-pt-224`). Exit code is non-zero while anything fails — **do not proceed**;
report each failure. The other subcommands: `status` (see the polling rule below), `clean` (before
any retry), `launch` (start workers with correct redirects + recorded PIDs).

Why these specific gates — each one is a failure we hit and misdiagnosed:
- **Every rank builds the policy itself**, so the pretrained backbone must be cached on EVERY node
  and gated repos need `HF_TOKEN` there. A worker without them 403s seconds after launch. ⚠️ A
  non-login ssh reads no rc file at all, so run remote commands through `bash -lc` (which picks
  up `env set` credentials via /etc/profile.d) or the
  token is invisible even when it is configured.
- **The master's disk filling during a checkpoint save surfaces as an NCCL *timeout on the worker***
  — it sends you to debug the wrong node. Only the master writes checkpoints, so budget
  `(steps/save_freq) × ckpt_size` there BEFORE launching.
- **Generally, in a multi-node run any rank's local problem appears as "some other rank timed out."**
  When you see a collective timeout, check every node's own log for the real error first.
- **Orphaned ranks survive a failed run** (`setsid`-detached workers outlive the master) and keep
  holding GPU memory, so the next attempt OOMs or rendezvous misbehaves. Always
  `multinode.py clean` between attempts and confirm GPU memory actually returned to ~0.
- Cleanup naturally involves force-kills and recursive deletes, which the console's **security scan
  may block**. Don't fight it — surface the exact command and let the user approve it.

**Only after all of M0 passes** do you plan the run — and the multi-node *smoke test* (below) is
still a separate, later gate: M0 proves the nodes can talk, the smoke test proves accelerate/NCCL
actually rendezvous and train.

### M0b — RDMA (RoCE / InfiniBand): prefer it, and PROVE you got it
Gradient all-reduce is the cross-node bottleneck, and it runs over NCCL. If the nodes have RDMA
NICs, NCCL should use them — **but when anything is misconfigured NCCL silently falls back to TCP:
no error, no warning, just a run that is many times slower.** Never assume; verify.

**1. Detect the USABLE HCA — run the detector, do not eyeball `ibv_devinfo`.**
```bash
python scripts/multinode.py rdma --worker <worker-addr>     # one line per node
```
⚠️ **`PORT_ACTIVE` is NOT evidence the HCA works here — and neither is `ibv_devinfo` showing a
GID.** RDMA is split across the container boundary: the `/dev/infiniband/uverbs*` **char devices**
are files, so they are NOT namespaced and the pod sees ALL of the host's HCAs; the **netdevs** are
namespaced, so they stay on the host. RoCE GIDs are derived from a netdev's IP, so an HCA whose
netdev is in the host netns has no GID *that this pod can use*.

The trap: **`ibv_devinfo` (the verbs API, which is what NCCL reads) is not netns-filtered** — it
happily reports `mlx5_0` GID `::ffff:<the NODE's IP>`. NCCL believes it, selects that HCA, and the
QP then fails in the kernel with `ibv_modify_qp failed with 19 No such device`. Only **sysfs** is
filtered by netns, so it is the honest view: the same `mlx5_0` reads back all-zero GIDs inside the
pod. Gating on "a port is ACTIVE" (or on `ibv_devinfo` showing a GID) is exactly what lets a doomed
run start and then die mid-launch.

The real test, which `multinode.py rdma` applies: an HCA is usable **iff it has a RoCE v2 GID whose
backing netdev exists in THIS netns** (`gid_attrs/ndevs/<i>` naming an interface present in
`/sys/class/net/`). On a console pod that is the **vRDMA device riding the pod's own ENI**
(`ndev=eth0`, GID = the *pod's* IP) — in practice `mlx5_5`, but never hardcode it.

> No usable HCA on some node? Then RDMA is not available to that pod: launch it with
> `NCCL_IB_DISABLE=1` (honest TCP) rather than let NCCL pick a dead HCA and crash the run.

**2. Pin `NCCL_IB_HCA` per node — this is the whole fix.** Left to itself NCCL enumerates all six
devices, picks `mlx5_0`, reads a GID belonging to the **node's** IP, and dies with
`ibv_modify_qp failed with 19 No such device`. Pinning the detected HCA makes the identical job run
end-to-end over `NET/IB`. ⚠️ **This is the one legitimate exception to "the same command on every
node"** — the device is a per-machine fact, so `NCCL_IB_HCA` is set per node while every
`lerobot-train` flag stays identical.
> Device names like `mlx5_5` are **per-machine facts, not constants** — never copy one from these
> docs or another cluster. On bare-metal multi-GPU nodes also weigh GPU affinity
> (`nvidia-smi topo -m`, the NIC marked `PIX`); with several GPUs list several HCAs, since pinning
> a single NIC caps you at that NIC's bandwidth.

**3. Env, set per node** (every `<…>` below is measured on that node, not copied):
```bash
export NCCL_IB_DISABLE=0                  # 1 forces TCP — never set it to 1 "to test"
eval $(python scripts/multinode.py rdma --export)   # -> NCCL_IB_HCA=<this node's HCA>
export NCCL_SOCKET_IFNAME=<data iface>    # bootstrap/out-of-band only, NOT the data path;
                                          # get it from `ip -o -4 addr` (often eth0, don't assume)
```
❌ **Do NOT set `NCCL_IB_GID_INDEX`** — and be aware there are **two different GID numberings**:
- **sysfs** (`/sys/class/infiniband/<dev>/ports/1/gids/<i>`) is **filtered by netns**: it shows only
  GIDs whose netdev is in this pod, indexed per pod (measured 15 on one console pod, 19 on the
  other, both `::ffff:<that pod's own IP>`). This is the view the detector uses, and the only view
  that answers "can this HCA work here".
- **the verbs API** (`ibv_devinfo -v`, and what NCCL reads) is **NOT netns-filtered**: it returns
  the host's table — both console pods report the same `GID[5] = ::ffff:192.168.1.114` (the ENI's
  primary IP, not either pod's).

So an index read off sysfs is meaningless to NCCL, and one read off `ibv_devinfo` may name an
address that does not exist in this netns. Pin the HCA and let NCCL choose — that is verified to
work (`NET/IB`, all-reduce OK); a hand-picked index is not.

> **No k8s resource request is needed.** Claiming `vke.volcengine.com/rdma` is NOT what enables
> this — verified by removing the claim from one pod and re-running: it still gets `uverbs0-5` and
> a working vRDMA HCA, and still completes a 2-pod `NET/IB` all-reduce. The claim only adds
> `umad*`/`issm*` management nodes for a physical NIC, which NCCL never touches.

**4. PROVE it — the whole point.** Run a 2-node NCCL all-reduce (or the multi-node smoke test)
with `NCCL_DEBUG=INFO` and read the transport line:
- **`NET/IB`** → RDMA is in use ✅
- **`NET/Socket`** → it fell back to TCP ❌ — fix it before burning GPU-hours; do not proceed and
  call it "working".
Also scan for `NCCL WARN`, GID/HCA selection errors, and "no usable device" notes. Report which
transport was selected; "the job started" is NOT evidence of RDMA.

This check is cheap (tens of seconds) and belongs **before** the real run — a run that quietly
trains over TCP looks healthy on every dashboard while wasting most of the interconnect.

**Plan with cross-node totals (steps math) but master-local eval:** run `plan_training.py` with
**`--gpus <TOTAL GPUs across ALL nodes>`** (so `global_batch = batch × total_processes` and the
steps/save_freq math is right) **and `--cuda <master-local training GPU indices>`** (so the
spare-GPU/eval_mode logic stays master-local). When `--gpus` exceeds the box's GPU count the plan
emits a `multi_node_hint` reminding you of exactly this. Effective batch = `batch × num_processes`
(per-process batch unchanged) — consider rescaling LR.

**Persist the topology (MANDATORY once the user gives you the addresses).** Write a `multi_node`
block into `<session>/training_plan.json` — the conversation is not durable, and the watchdog keys
off this block. Store the addresses **exactly as the user gave them** (DNS name or IP):
```json
"multi_node": {
  "master_addr": "…", "worker_addrs": ["…"], "gpus_per_node": 8,
  "num_machines": 2, "num_processes": 16,
  "master_launch_command": "cd /lerobot && accelerate launch --multi_gpu … --machine_rank=0 … $(which lerobot-train) <flags>",
  "master_resume_command": "cd /lerobot && accelerate launch --multi_gpu … --machine_rank=0 … $(which lerobot-train) --resume=true --config_path=<out>/checkpoints/last/pretrained_model/train_config.json"
}
```
⚠️ If any address is a **pod IP**, it changes on pod restart — after a restart, re-confirm
`master_addr`/`worker_addrs` and regenerate every node's commands before resuming. (Headless DNS
names don't have this problem, which is why they're preferred.)

**Launch — run the SAME command on every node, differing ONLY by `--machine_rank`.** Take the exact
`lerobot-train` flags `plan_training.py` emitted (steps/batch/dataset/policy/**output_dir**/…) and wrap
them with the accelerate multi-node prefix:
```bash
cd /lerobot && accelerate launch \
  --multi_gpu \
  --num_machines=<N> \
  --num_processes=<TOTAL GPUs across ALL nodes> \
  --machine_rank=<R> \              # 0 = master, 1,2,… = each worker
  --main_process_ip=<MASTER_IP> \
  --main_process_port=29500 \
  $(which lerobot-train) \
  <the SAME lerobot-train flags for every node>
```
- **Emit a ready-to-paste command for EACH node** — master (`--machine_rank=0`) and one per worker
  (`--machine_rank=1`, `2`, …), filling in `<MASTER_IP>`, `<N>`, `<TOTAL GPUs>`. The flags after
  `$(which lerobot-train)` are **identical on every node** (same dataset path, same `output_dir`).
- Worker-side dataset self-check BEFORE launch — `<dataset_root>/meta/info.json` must exist at the
  **same path** as on the master. Console pods: check it yourself over ssh
  (`ssh root@<worker-dns> 'ls <dataset_root>/meta/info.json'`); other nodes: give the user the `ls`.

**Multi-node smoke test (HARD GATE — the multi-node analogue of preflight).** `preflight.py` only
smokes the single-node command; it cannot catch the top cross-node failures (rendezvous not
connecting, NCCL hangs, dataset missing on a worker, env drift between nodes). Before the real
launch, run the **full multi-node command on ALL nodes** with `--steps=2 --save_freq=1` and a
throwaway `--output_dir`, and confirm every rank reaches step 2 and the master writes a checkpoint.
⚠️ **Do not pre-create the smoke `output_dir`, and before ANY retry reset EVERY node** with
`python scripts/multinode.py clean --worker <addr> --output-dir <dir> --remove-output-dir` —
it kills leftover ranks, verifies the GPUs actually came back, and removes the dir everywhere.
lerobot-train aborts on an existing output_dir without `--resume`, and an orphaned rank still
holding GPU memory makes the retry OOM — so skipping this makes every retry fail identically.
**RDMA in the smoke run — detect per node, then assert.** Every rank's script must resolve its OWN
HCA (the device name is per-machine, and `NCCL_IB_GID_INDEX` must stay unset — see M0b):
```bash
if hca=$(python scripts/multinode.py rdma --export); then export "$hca"; else export NCCL_IB_DISABLE=1; fi
export NCCL_DEBUG=INFO
```
`--export` prints `NCCL_IB_HCA=<hca>` for the node it runs on and exits non-zero when that node has
no usable HCA, so the `else` degrades to honest TCP instead of letting NCCL grab a dead device and
kill the run with `ibv_modify_qp failed with 19 No such device`.
⚠️ Capture it in a variable as above — **`eval $(… --export) || export NCCL_IB_DISABLE=1` is
broken**: `eval`'s exit status is that of the string it evaluated, and on failure the string is
empty, so `eval ""` returns 0 and the fallback never fires.
Then **assert the transport** in the smoke log: it must say **`NET/IB`** and name that HCA
(`NET/IB : Using [0]mlx5_5:1/RoCE`), not `NET/Socket` (silent TCP fallback) — see M0b. The smoke
test is the cheapest place to catch either failure; do not carry an unproven interconnect into a
run that costs GPU-hours.
Other aids: if rendezvous connects but NCCL hangs on a multi-NIC node, set
`NCCL_SOCKET_IFNAME=<iface>`. Note NCCL also uses **ephemeral ports beyond 29500** — open
node↔node traffic, not just one port; and if two trainings share a master box, give each a
distinct `--main_process_port`.

**⚠️ Multi-node startup is MINUTES OF SILENCE — you MUST stream progress from BOTH nodes.** This is
the single-node "slow commands must stream progress" rule, made stricter: a cross-node launch spends
minutes in rendezvous + model load (a VLA like pi05 is slow to load) emitting nothing, and the
**worker's output lives on the worker**, so the user sees an idle screen and concludes it died.
- **NEVER block on the launch** — no blocking `wait`, no bare `sleep N` with output only at the end.
  Launch in the background writing to a log, then poll.
- **The worker's log must be written ON THE WORKER**: `ssh root@$W 'cmd > /path/worker.log 2>&1'`
  (redirect INSIDE the quotes). Unquoted, `ssh root@$W cmd > worker.log` redirects on the MASTER and
  the worker-side file stays empty — a real trap that costs a whole run to notice.
- **⚠️ NEVER inline the training command in an ssh argument — ship a SCRIPT.** Text passed as an
  ssh argument goes through TWO shell expansions on the remote side (sshd's login shell, then your
  `bash -lc`), so nested quotes are eaten: `--dataset.episodes='[0, 1, 2]'` and
  `--rename_map='{"front": …}'` arrive mangled and lerobot dies in YAML parsing — with an error
  that points at the config, not at the quoting. Write the command into a script file on the worker
  (send the body over **stdin**, e.g. `ssh root@$W 'cat > /path/rank.sh'`), then run that script.
  `multinode.py launch` does exactly this (and `run_remote` feeds every command over stdin to
  `bash -l -s`), which is another reason not to hand-roll the ssh.
- **Poll every ~20–30 s and relay a one-line status from EACH node**, naming the phase so "alive but
  slow" is distinguishable from "hung":
  ```bash
  python scripts/multinode.py status --worker <addr> \
      --master-log <master.log> --worker-log <worker.log>
  ```
  It prints the phase per node (rendezvous → policy/dataset construction, the long quiet one →
  `Start offline training` → first tqdm `N/M [` → first checkpoint) plus that node's last line, and
  flags an empty worker log as the classic mis-redirect. Relay it, and say how long each node has
  been in its phase.
- **If a node emits nothing for minutes, say so explicitly** ("master: still loading the policy, 3m
  in, no output yet — normal for pi05; worker: rendezvous connected") instead of going quiet. Escalate
  to the M0 checks / `NCCL_DEBUG=INFO` only after the wait clearly exceeds a model-load time.

**Run: master under the watchdog, workers by hand.** Master runs under the **watchdog** as usual
(`session.py add-run`, `watchdog.py`, `monitor_server.py`); it uses `multi_node.master_launch_command`.
Workers are started manually by the user and rendezvous with the master's IP:port. Before the first
step the master legitimately waits in the rendezvous for the workers — the watchdog knows
(`multi_node` present) and won't count that wait as a stall; still, start the workers promptly.

**Checkpoints & resume (master-only + manual scp; watchdog will NOT auto-restart):**
- **Only the master (rank 0) writes checkpoints** to `output_dir/checkpoints/` — accelerate saves and
  logs only on the main process. Workers write nothing.
- **The watchdog never auto-relaunches a multi-node run.** On crash/stall it sets the run
  `blocked` with the manual procedure (it can't restart workers, and the single-node
  resume_command would silently restart with the WRONG world size). A multi-node resume is driven
  by hand, but not improvised — one command puts the checkpoint everywhere it is needed:
  ```bash
  python .../multinode.py resume --worker <addr> --output-dir <the ORIGINAL run's output_dir>
  python .../multinode.py check  --resume --master <addr> --worker <addr> --output-dir <same>
  ```
  `resume` resolves `checkpoints/last` (a relative symlink), refuses a half-written checkpoint,
  rsyncs **only that step dir** to every worker (~9 GB for pi05; the whole `checkpoints/` tree
  would be several times that), recreates `last` there, re-verifies, and prints the exact
  `--resume=true --config_path=…` to append. `check --resume` then INVERTS the output_dir gate:
  on a resume the directory must exist, with a checkpoint, on every node.
  Why the copy is needed at all: every rank loads the checkpoint from its **own local**
  `output_dir`, but only rank 0 ever wrote one — so an un-synced worker dies instantly while the
  master looks healthy. Relaunch every node with the same accelerate prefix (only
  `--machine_rank` differs), then rerun the watchdog on the master (it uses
  `multi_node.master_resume_command`). The plan's bare `resume_command` is single-node —
  **never use it for a multi-node run**.
  ⚠️ If the run died **before** the first `save_freq` step there is no checkpoint at all: resume
  is impossible. Delete `output_dir` on every node (`clean --remove-output-dir`) and start over.
  A leftover `output_dir` with no checkpoint is exactly what raises
  `FileExistsError: Output directory … already exists and resume is False`.

**Eval runs ONLY on the master node** — checkpoints exist only there. Start `eval_watcher.py` /
`offline_eval.py` on the master (spare GPU on master → concurrent; else `post_training` on master after
the run). **Never on a worker** (no checkpoints, and it would contend with that worker's training).

