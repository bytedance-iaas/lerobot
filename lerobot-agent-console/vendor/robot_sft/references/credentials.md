# Credentials on the console pods (HF_TOKEN, TOS keys, HF_ENDPOINT)

Why the one-command rule exists, and what each alternative actually breaks. SKILL.md carries the
command; this file carries the reasoning — read it if you are tempted to do it another way, or if
a credential is set but something still cannot see it.

- **Creds the user gives in chat go through `multinode.py env set` — never `export`, never an rc
file you edit by hand.** One command puts the value on every node and in front of every shell:

```bash
# value on STDIN with a QUOTED heredoc — it is taken literally, so $ ' " | ; in a token are safe
python /opt/data/skills/robot_sft/scripts/multinode.py env set HF_TOKEN --worker <worker-addr> <<'EOF'
hf_xxxxxxxxxxxxxxxxxxxx
EOF
python .../multinode.py env show HF_TOKEN HF_ENDPOINT --worker <worker-addr>   # verify, prints only LENGTHS
```
Same for `TOS_ACCESS_KEY`/`TOS_SECRET_KEY`/`HF_ENDPOINT`. Drop `--worker` for a single-node run.

Why not the obvious alternatives — each is a failure we actually hit:
- `export FOO=…` alone dies with the agent's shell; the **background** watchdog/`lerobot-train`
  never sees it.
- Editing `~/.bashrc` is worse than useless: `~` is `/opt/data` for the console but `/root` over
  ssh, so the two disagree — and `.bashrc` is read only by INTERACTIVE shells anyway, while the
  agent's own tool calls are plain `bash -c`, which reads **no** rc file. That is the "I set it,
  but the next command cannot see it" loop.
- Putting the value in the command (`ssh w "export HF_TOKEN=$T"`) sends it through two shell
  expansions into argv, history and every approval prompt — one awkward character and it is
  `not a valid identifier` or a Python syntax error.

`env set` writes ONE file, `/opt/data/.console-env.sh` (0600, on the PVC → survives restarts),
pulled in by `/etc/console-shell-init.sh`, which is hooked from all THREE places a bash can
read, because they cover disjoint shell types:
| shell | hook |
|---|---|
| login — `bash -l`, ssh login, the launcher | `/etc/profile.d/10-console-env.sh` |
| non-interactive — `bash -c`, python subprocesses | `$BASH_ENV` |
| **ssh remote command — `ssh host 'cmd'`** | **`~/.bashrc`** |

⚠️ That third row is its own trap, and it bites PATH before it bites credentials: **sshd builds
its environment from scratch**, so neither the image's `ENV` nor anything exported by PID 1
crosses into an ssh session — `$BASH_ENV` included. Measured on the worker,
`ssh host 'which accelerate'` returned nothing with
`PATH=/root/.local/bin:/usr/local/sbin:…` (no venv), while the identical command under
`bash -l -s` found it. So `/etc/console-shell-init.sh` also puts `/lerobot/.venv/bin` on PATH
(idempotently — it is sourced by every bash, including nested ones). It ships the
file to each `--worker` over stdin, then re-checks each node through a login shell **by
comparing the length** — not by testing non-empty, because the failure below passes that test.

Before writing, it sweeps each node for competing `export NAME=` lines in `/root/.bashrc`,
`/opt/data/.bashrc` and the matching `.profile`s, and comments them out (reporting each file).
This is mandatory, not tidiness: a login shell runs `/etc/profile` → `profile.d` → `~/.profile`
→ `~/.bashrc`, so a leftover rc export executes **after** the hook and silently **wins**. Measured
on the live pods: `env set` reported 26 chars on both nodes while the worker was still serving a
3-char token from two hand-edited rc files.
⚠️ On a pod started before this was added, `$BASH_ENV` is unset — `env set` says so, and until the
next rollout one-off commands need `bash -lc`. `multinode.py` and `launch` already use login
shells, so training itself is unaffected either way.
⚠️ `env show` prints **lengths, not values** — use it. A *wrong* token is far worse than a missing
one: `test -n "$HF_TOKEN"` passes and you 403 six hours in. (Seen for real: worker `HF_TOKEN=3
chars`, master `0` — leftover from hand-edited rc files.) Clear a bad one with `env unset`, and
note it does not touch values set outside the file — grep `/root/.bashrc` and `/opt/data/.bashrc`.
