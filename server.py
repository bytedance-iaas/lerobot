#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LeRobot Agent Console — a single-page web app to operate LeRobot.

This is a standalone app that *uses* LeRobot (it is not part of the LeRobot
source tree). It assumes a LeRobot checkout / install is reachable in the same
environment (e.g. the same container), and drives it via a shell + the
``hermes`` agent.

Serves on :8080 (configurable via PORT) and exposes:

  GET  /                 the single-page white UI (see image.png reference)
  GET  /api/status       {chat_ready, model, base_url, skill_installed}
  POST /api/volcano-key  configure the Volcengine (Ark) api key for hermes chat
  WS   /ws/term          PTY bridge to an interactive shell (the "ssh console")
  WS   /ws/chat          chat bridge to the `hermes` agent (one turn per message)

Design notes
------------
* The terminal is a real PTY running ``$CONSOLE_SHELL`` (default ``bash``) in
  ``$CONSOLE_WORKDIR`` — the same thing you'd get by ``kubectl exec`` into the
  pod, just over the browser.
* Chat runs the ``hermes`` agent non-interactively, one turn per user message,
  resuming a named session so context carries across turns. The Volcengine api
  key only affects chat (it is written to hermes' own config); the terminal and
  everything else are untouched.

This module has no LeRobot imports on purpose: it is a thin ops shell that
launches LeRobot commands as subprocesses, so it stays importable/runnable even
in a minimal image.
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hmac
import json
import logging
import os
import re
import shutil
import signal
import ssl
import struct
import termios
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agent_console")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

PORT = int(os.environ.get("PORT", "8080"))
SHELL = os.environ.get("CONSOLE_SHELL") or shutil.which("bash") or "/bin/sh"
# Working dir for the shell console + hermes agent (the lerobot checkout).
# Precedence: CONSOLE_WORKDIR > LEROBOT_HOME > cwd.
# IMPORTANT: LEROBOT_HOME is *lerobot's own* (deprecated) cache var — recent
# lerobot RAISES if it's set. So we read it for back-compat but then REMOVE it
# from the environment, so it never leaks to children (shell, hermes, lerobot CLIs).
WORKDIR = os.environ.get("CONSOLE_WORKDIR") or os.environ.get("LEROBOT_HOME") or os.getcwd()
os.environ.pop("LEROBOT_HOME", None)
HERMES_BIN = os.environ.get("HERMES_BIN") or shutil.which("hermes") or "hermes"
# Session LISTING calls hermes' own store API (hermes_state.SessionDB) — but it has to run
# in HERMES' interpreter, not ours: the two venvs disagree on 11 shared packages, so an
# in-process import would shadow our deps with theirs. HERMES_BIN is a symlink into the
# hermes venv's bin/, so its resolved sibling `python` is that interpreter.
HERMES_PY = os.environ.get("HERMES_PY") or str(Path(HERMES_BIN).resolve().with_name("python"))
HERMES_SESSION_API = os.environ.get("HERMES_SESSION_API") or "/opt/hermes/session_api.py"
# Single-user HTTP Basic auth, credentials from the environment. When both are
# set, EVERY route (page, static, WS, proxy) requires them; otherwise the console
# is open (logged as a warning). This is a single account by design.
AUTH_USER = os.environ.get("CONSOLE_USER") or os.environ.get("AUTH_USER") or ""
AUTH_PASS = os.environ.get("CONSOLE_PASSWORD") or os.environ.get("AUTH_PASSWORD") or ""
AUTH_ENABLED = bool(AUTH_USER and AUTH_PASS)
# Native TLS: if both paths are set and exist, the console serves HTTPS (wss)
# directly — no TLS-terminating sidecar needed. Use a self-signed cert behind an
# L4 LB, or any cert. Unset => plain HTTP (handy for local dev).
TLS_CERT = os.environ.get("CONSOLE_TLS_CERT") or ""
TLS_KEY = os.environ.get("CONSOLE_TLS_KEY") or ""


def _tls_enabled() -> bool:
    """True when both cert + key are set and present → the console serves HTTPS."""
    return bool(TLS_CERT and TLS_KEY and os.path.exists(TLS_CERT) and os.path.exists(TLS_KEY))


# Skill to preload so the agent knows how to drive LeRobot SFT (requirement f).
CHAT_SKILL = os.environ.get("HERMES_CHAT_SKILL", "robot_sft")

# One-time steer (first turn of a session). Answers render as Markdown in the chat bubble
# (via marked + DOMPurify), which is what models emit most reliably.
CHAT_DIRECTIVE = (
    "[System] 请用简洁的 Markdown 作答：用 **加粗**、`行内代码`、有序/无序列表、表格等呈现"
    "结构化信息；代码或命令放进 ``` 代码块。不要输出完整 HTML 页面或 <script>/<style>。\n"
    "运行环境：lerobot 源码在 `/lerobot`，其 uv 虚拟环境（`/lerobot/.venv`）已在 PATH 上——"
    "直接用 `python` / `lerobot-*` 命令即可，无需激活。若确需 `uv run`，务必加 `--no-sync`"
    "（否则 uv 会尝试联网重新同步依赖，很慢）。\n"
    "**重要——单 agent 执行**：所有工作都由你（主 agent / main agent）亲自、顺序完成。"
    "**不要**派生、委派或编排任何 subagent（不要用 task / delegation / 子 agent / 多 agent）。"
    "即使 robot_sft 技能里写的是“多 agent 流水线”“每个阶段由子 agent 执行”，在本环境中一律改为："
    "你自己按阶段顺序执行、逐段把 session 状态落盘到 `/opt/data`，靠重读 session 状态续跑——"
    "不靠子 agent。这样避免上下文丢失、额外开销和 ACP 中断问题。\n\n"
)

# SINGLE SOURCE OF TRUTH for the chat model + endpoint is hermes' config.yaml
# (model.default / model.base_url), seeded once in the console Dockerfile via
# `hermes config set ...`. server.py READS them from there — see read_chat_config().
# These ENV vars are last-resort fallbacks ONLY, with NO hardcoded model/URL string in the
# code: if hermes has no value and the env is unset, the field stays empty and the UI shows a
# not-configured state (the key modal) instead of guessing a model.
DEFAULT_BASE_URL = os.environ.get("ARK_BASE_URL", "")
DEFAULT_MODEL = os.environ.get("ARK_MODEL", "")
# The chat-header dropdown's *alternative* models come from the ARK_MODELS env (comma-separated),
# set once in the console Dockerfile. The currently-configured hermes model is always added to the
# list (see _model_choices), so the dropdown never hides what's actually in use.
_ENV_MODELS = [m.strip() for m in os.environ.get("ARK_MODELS", "").split(",") if m.strip()]

# Placeholder values that mean "no real key yet" — treat chat as not-ready.
_PLACEHOLDERS = {"", "your-api-key", "changeme", "<set-me>", "null", "none"}


# --------------------------------------------------------------------------- #
# hermes config helpers (chat only)                                           #
# --------------------------------------------------------------------------- #
def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _hermes_config_path() -> Path:
    return _hermes_home() / "config.yaml"


def _parse_model_block(text: str) -> dict:
    """Extract model.{default,base_url,api_key} from config.yaml WITHOUT pyyaml.

    pyyaml may not be installed in every runtime; we must NOT silently report the
    chat as not-ready just because a yaml import failed (that traps the key modal
    in a loop). hermes writes a plain block-style `model:` mapping, so a tiny
    indentation-aware line parser is enough and dependency-free.
    """
    out: dict[str, str] = {}
    in_model = False
    for raw in text.splitlines():
        if re.match(r"^model:\s*$", raw):
            in_model = True
            continue
        if in_model:
            if raw and not raw[0].isspace():  # next top-level key → end of block
                break
            m = re.match(r"^\s+(default|base_url|api_key):\s*(.*?)\s*$", raw)
            if m:
                out[m.group(1)] = m.group(2).strip().strip("\"'")
    return out


def read_chat_config() -> dict:
    """Return {chat_ready, model, base_url} by reading hermes' config.yaml."""
    cfg_path = _hermes_config_path()
    model = DEFAULT_MODEL
    base_url = DEFAULT_BASE_URL
    api_key = ""
    try:
        text = cfg_path.read_text()
    except FileNotFoundError:
        text = ""
    except Exception as e:  # noqa: BLE001
        log.warning("could not read hermes config: %s", e)
        text = ""
    if text:
        fields = {}
        try:
            import yaml  # preferred when available

            data = yaml.safe_load(text) or {}
            m = data.get("model")
            if isinstance(m, dict):
                fields = {k: m.get(k) for k in ("default", "base_url", "api_key")}
        except ImportError:
            fields = _parse_model_block(text)  # dependency-free fallback
        except Exception as e:  # noqa: BLE001 — malformed yaml etc.
            log.warning("yaml parse failed, falling back to line parse: %s", e)
            fields = _parse_model_block(text)
        model = (fields.get("default") or model)
        base_url = (fields.get("base_url") or base_url)
        api_key = (fields.get("api_key") or "").strip()
    chat_ready = api_key.lower() not in _PLACEHOLDERS
    return {"chat_ready": chat_ready, "model": model, "base_url": base_url}


async def _hermes_config_set(key: str, value: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        HERMES_BIN, "config", "set", key, value,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"`hermes config set {key}` failed: {out.decode(errors='replace')[:400]}")


async def set_volcano_key(api_key: str, base_url: str | None, model: str | None) -> None:
    """Point hermes chat at Volcengine Ark with the user's key. Chat-only."""
    await _hermes_config_set("model.provider", "custom")
    # Only overwrite base_url/model when actually supplied (user field or env fallback) — otherwise
    # keep whatever hermes already has (the Dockerfile seed), never clobbering it with an empty value.
    if base_url or DEFAULT_BASE_URL:
        await _hermes_config_set("model.base_url", base_url or DEFAULT_BASE_URL)
    if model or DEFAULT_MODEL:
        await _hermes_config_set("model.default", model or DEFAULT_MODEL)
    await _hermes_config_set("model.api_key", api_key)  # value comes from the user, not the repo


# --------------------------------------------------------------------------- #
# HTTP handlers                                                               #
# --------------------------------------------------------------------------- #
async def handle_index(_request: web.Request) -> web.StreamResponse:
    return web.FileResponse(STATIC / "index.html")


async def handle_status(_request: web.Request) -> web.Response:
    info = read_chat_config()
    info["skill"] = CHAT_SKILL
    info["workdir"] = WORKDIR
    info["secure"] = _tls_enabled()   # the UI shows a small warning when this is false
    return web.json_response(info)


def _versions() -> dict:
    """Deployed image versions surfaced in the UI (release-notes button): the lerobot base
    commit (= the base image tag) and this console's commit."""
    img = os.environ.get("LEROBOT_IMAGE", "")
    tag = img.rsplit(":", 1)[-1] if ":" in img else ""
    lerobot = "" if "/" in tag else tag  # guard: a host:port ":" with no tag
    console = os.environ.get("CONSOLE_COMMIT", "").strip()  # from --build-arg CONSOLE_COMMIT
    return {"lerobot": lerobot or "unknown", "console": console or "unknown"}


async def handle_version(_request: web.Request) -> web.Response:
    return web.json_response(_versions())


def _model_choices() -> list:
    models = list(_ENV_MODELS)  # alternatives from the ARK_MODELS env (set in the Dockerfile); may be empty
    cur = read_chat_config().get("model")  # the hermes-configured model — always offered
    if cur and cur not in models:
        models = [cur, *models]
    return models


async def handle_models(_request: web.Request) -> web.Response:
    return web.json_response({"models": _model_choices(), "current": read_chat_config().get("model")})


async def handle_set_model(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    model = (body.get("model") or "").strip()
    if not model:
        return web.json_response({"ok": False, "error": "model is required"}, status=400)
    try:
        await _hermes_config_set("model.default", model)  # chat-only; base_url/key unchanged
    except Exception as e:  # noqa: BLE001
        log.exception("failed to set model")
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    acp: HermesACP = request.app.get("acp")
    if acp:
        try:
            await acp.restart()  # so the next chat turn uses the new model
        except Exception as e:  # noqa: BLE001
            log.warning("acp restart after model change failed: %s", e)
    return web.json_response({"ok": True, **read_chat_config()})


# Whitelisted markdown docs served straight from the lerobot checkout baked into the image
# (LEROBOT_HOME=/lerobot), so the UI links to the local README instead of an external URL.
_LEROBOT_DOCS = {
    "webrtc": "src/lerobot/robots/webrtc_proxy/README.md",
    "webrtc-design": "src/lerobot/robots/webrtc_proxy/DESIGN.md",
}


async def handle_lerobot_doc(request: web.Request) -> web.Response:
    rel = _LEROBOT_DOCS.get(request.match_info.get("name", ""))
    if not rel:
        return web.Response(status=404, text="unknown doc")
    root = os.environ.get("LEROBOT_HOME") or WORKDIR
    try:
        text = (Path(root) / rel).read_text(encoding="utf-8")
    except Exception:
        # Not found locally (e.g. running outside the pod) → render a fallback that points at
        # the bytedance-iaas fork's dev branch.
        text = (
            f"# 文档未在本地找到\n\n镜像里的 lerobot 检出中没有 `{rel}`。\n\n"
            f"在线查看：[bytedance-iaas/lerobot @ dev](https://github.com/bytedance-iaas/lerobot/tree/dev/{rel})\n"
        )
    return web.Response(text=text, content_type="text/markdown")


async def handle_volcano_key(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    api_key = (body.get("api_key") or "").strip()
    if api_key.lower() in _PLACEHOLDERS:
        return web.json_response({"ok": False, "error": "api_key is required"}, status=400)
    try:
        await set_volcano_key(api_key, body.get("base_url"), body.get("model"))
    except Exception as e:  # noqa: BLE001
        log.exception("failed to set volcano key")
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    # Restart the persistent agent so a new session uses the new credentials.
    acp: HermesACP = request.app.get("acp")
    if acp:
        try:
            await acp.restart()
        except Exception as e:  # noqa: BLE001
            log.warning("acp restart after key change failed: %s", e)
    return web.json_response({"ok": True, **read_chat_config()})


# --------------------------------------------------------------------------- #
# Terminal (PTY) websocket — the "ssh console"                                #
# --------------------------------------------------------------------------- #
def _set_winsize(fd: int, rows: int, cols: int) -> None:
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError:
        pass


async def handle_term(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    pid, master_fd = os.forkpty()
    if pid == 0:  # child
        try:
            os.chdir(WORKDIR)
        except OSError:
            pass
        os.environ.setdefault("TERM", "xterm-256color")
        os.execvp(SHELL, [SHELL, "-l"] if SHELL.endswith("bash") else [SHELL])
        os._exit(127)  # unreachable

    log.info("term session opened: pid=%s shell=%s cwd=%s", pid, SHELL, WORKDIR)
    loop = asyncio.get_running_loop()
    os.set_blocking(master_fd, False)

    def _pump_pty() -> None:
        try:
            data = os.read(master_fd, 65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b""
        if not data:
            loop.remove_reader(master_fd)
            asyncio.ensure_future(_close())
            return
        asyncio.ensure_future(ws.send_bytes(data))

    closed = False

    async def _close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            loop.remove_reader(master_fd)
        except Exception:  # noqa: BLE001
            pass
        try:
            os.kill(pid, signal.SIGHUP)
        except ProcessLookupError:
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        if not ws.closed:
            await ws.close()

    loop.add_reader(master_fd, _pump_pty)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                os.write(master_fd, msg.data)
            elif msg.type == WSMsgType.TEXT:
                # control frames are JSON: {"type":"resize","cols":..,"rows":..} or {"type":"input","data":..}
                try:
                    ctrl = json.loads(msg.data)
                except json.JSONDecodeError:
                    os.write(master_fd, msg.data.encode())
                    continue
                if ctrl.get("type") == "resize":
                    _set_winsize(master_fd, int(ctrl.get("rows", 24)), int(ctrl.get("cols", 80)))
                elif ctrl.get("type") == "input":
                    os.write(master_fd, ctrl.get("data", "").encode())
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    finally:
        await _close()
        log.info("term session closed: pid=%s", pid)
    return ws


# --------------------------------------------------------------------------- #
# Chat — one persistent `hermes acp` process, driven over its session API       #
# --------------------------------------------------------------------------- #
# Instead of cold-spawning `hermes chat` per turn (~10s startup each), we launch
# ONE `hermes acp` process together with the server and talk JSON-RPC to it:
#   initialize -> session/new (once) -> session/prompt (per turn, streaming).
# This keeps the agent warm and lets us stream tokens + tool calls live.

class HermesACP:
    """A persistent hermes ACP (Agent Client Protocol) client over stdio."""

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.session_id: str | None = None
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        # Session ids we've already steered with CHAT_DIRECTIVE. New sessions get it
        # on their first prompt; loaded (existing) sessions are marked so we never
        # inject it mid-conversation.
        self._directive_sent: set[str] = set()
        # Per-turn callbacks (single-user console → one active turn at a time):
        self.on_update = None       # async fn(update dict) — stream notifications
        self.on_permission = None   # async fn(params) -> optionId|None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def ensure(self) -> None:
        """Process alive AND a current session selected (creates one if none)."""
        async with self._start_lock:
            if not self.alive:
                await self._spawn()
            if not self.session_id:
                await self._new_locked()

    async def _ensure_proc(self) -> None:
        """Just the process+handshake (no session) — for list/load/new."""
        async with self._start_lock:
            if not self.alive:
                await self._spawn()

    async def restart(self) -> None:
        """Used after the Volcengine key changes (a new session picks it up)."""
        async with self._start_lock:
            if self.alive:
                old = self.proc
                try:
                    old.terminate()
                except ProcessLookupError:
                    pass
                # WAIT for the old process to actually die before spawning the new one.
                # terminate() alone raced: the old process's read-loop ended AFTER the new
                # spawn's initialize was registered, and its exit cleanup failed every
                # pending future — killing the new initialize with "hermes acp exited"
                # (the recurring restart failure). Escalate to SIGKILL if it lingers
                # (a stuck tool can ignore SIGTERM).
                try:
                    await asyncio.wait_for(old.wait(), timeout=5)
                except asyncio.TimeoutError:
                    try:
                        old.kill()
                    except ProcessLookupError:
                        pass
                    await old.wait()
            self.proc = None
            self.session_id = None
            self._directive_sent.clear()
            await self._spawn()

    async def _spawn(self) -> None:
        env = dict(os.environ)
        env["HERMES_ACCEPT_HOOKS"] = "1"
        env.setdefault("NO_COLOR", "1")
        # `--skills <name>` (global flag, before the acp subcommand) PINS the skill into every
        # session's context so the agent always sees it — instead of only auto-discovering it
        # when a message happens to match its description. CHAT_SKILL defaults to robot_sft.
        skill_args = ["--skills", CHAT_SKILL] if CHAT_SKILL else []
        # stderr goes to a log file, NOT DEVNULL: when the acp process dies, its dying
        # words (traceback / fatal error) are the only evidence — DEVNULL made every
        # "hermes acp exited" undebuggable.
        stderr_path = Path(os.environ.get("HERMES_HOME", "/opt/data")) / "logs" / "acp_stderr.log"
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_f = open(stderr_path, "ab")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                HERMES_BIN, *skill_args, "acp", "--accept-hooks",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=stderr_f, cwd=WORKDIR, env=env,
            )
        finally:
            stderr_f.close()  # child holds its own fd copy
        # Fresh pending map PER PROCESS, and the read loop is bound to (proc, its map):
        # an old process's exit cleanup must never touch a newer process's requests.
        self._pending = pending = {}
        asyncio.create_task(self._read_loop(self.proc, pending))
        await self._request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        log.info("hermes acp ready")

    async def _new_locked(self) -> str:
        """Create a fresh session (caller holds _start_lock). Becomes current."""
        res = await self._request("session/new", {"cwd": WORKDIR, "mcpServers": []})
        self.session_id = res.get("sessionId")
        log.info("hermes acp new session=%s", self.session_id)
        return self.session_id

    # ----- session management (new / list / load / delete) -----------------
    async def new_session(self) -> None:
        """Arm a new session WITHOUT creating one: the next prompt does that.

        Creating it here (or at startup) is what littered the store with titleless,
        zero-message ghosts — one per page-open / server-start / API-key change, since
        most of those never get a message. Deferring means a session row exists only
        once there is something in it.
        """
        await self._ensure_proc()
        self.session_id = None

    async def list_sessions(self) -> list[dict]:
        """Read the session store directly (hermes' own SessionDB API, via its venv).

        NOT ACP `session/list`: that server caches the list in memory, so a session
        deleted through the CLI kept coming back until hermes restarted — which is why
        this used to carry a `_deleted` filter on both ends of the wire. Reading the
        store means the list is simply always current.
        """
        proc = await asyncio.create_subprocess_exec(
            HERMES_PY, HERMES_SESSION_API, "list", "--limit", "200",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"session list failed (rc={proc.returncode}): {err.decode(errors='replace')[:400]}"
            )
        return json.loads(out.decode() or "[]")

    async def load_session(self, sid: str, on_update) -> None:
        """Switch to an existing session; replays its history via on_update."""
        await self._ensure_proc()
        self.on_update = on_update
        try:
            await self._request("session/load", {"sessionId": sid, "cwd": WORKDIR, "mcpServers": []})
        finally:
            self.on_update = None
        self.session_id = sid
        self._directive_sent.add(sid)  # existing history → never inject the directive

    async def delete_session(self, sid: str) -> None:
        # Delete via hermes' own CLI (schema-aware: also cleans the FTS index + related
        # tables). This is the supported path — no SQL fallback; a failure is surfaced.
        proc = await asyncio.create_subprocess_exec(
            HERMES_BIN, "sessions", "delete", sid, "--yes",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"`hermes sessions delete {sid}` failed (rc={proc.returncode}): "
                f"{out.decode(errors='replace')[:400]}"
            )
        self._directive_sent.discard(sid)
        if self.session_id == sid:
            self.session_id = None  # next prompt/ensure() makes a fresh one

    async def _write(self, obj: dict) -> None:
        async with self._write_lock:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            await self.proc.stdin.drain()

    async def _request(self, method: str, params: dict):
        self._id += 1
        rid = self._id
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return await fut

    async def _read_loop(self, proc: asyncio.subprocess.Process, pending: dict) -> None:
        # `pending` is THIS process's request map (see _spawn). Never use self._pending
        # here: after a restart that attribute points at the NEW process's map, and the
        # old loop's exit cleanup would fail the new process's in-flight requests.
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                fut = pending.pop(msg["id"], None)
                if fut and not fut.done():
                    fut.set_result(msg.get("result") or {"_error": msg.get("error")})
            elif msg.get("method") == "session/update":
                # Drop stale updates from a superseded process (post-restart stragglers).
                if self.on_update and self.proc is proc:
                    try:
                        await self.on_update(msg["params"].get("update", {}))
                    except Exception:  # noqa: BLE001
                        log.debug("on_update failed", exc_info=True)
            elif msg.get("method") == "session/request_permission":
                if self.proc is proc:
                    await self._reply_permission(msg)
            elif "id" in msg:  # server->client request we don't implement (e.g. fs/*)
                if self.proc is proc:
                    await self._write({"jsonrpc": "2.0", "id": msg["id"],
                                       "error": {"code": -32601, "message": "unsupported"}})
        # process exited — fail ONLY this process's in-flight requests, and only reset
        # live session state if we are still the current process (not superseded by restart).
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("hermes acp exited"))
        pending.clear()
        if self.proc is proc:
            self.session_id = None
        log.warning("hermes acp process exited (rc=%s%s)", proc.returncode,
                    "" if self.proc is proc else ", superseded")

    async def _reply_permission(self, msg: dict) -> None:
        params = msg.get("params", {})
        option_id = None
        if self.on_permission:
            try:
                option_id = await self.on_permission(params)
            except Exception:  # noqa: BLE001
                option_id = None
        outcome = {"outcome": "selected", "optionId": option_id} if option_id else {"outcome": "cancelled"}
        await self._write({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}})

    async def prompt(self, text: str, on_update, on_permission):
        await self.ensure()
        # Steer the answer format once per session, on its first prompt.
        if self.session_id not in self._directive_sent:
            # Append (not prepend) the steering block so the session's auto-title comes
            # from the user's real first words, not "[System] 请用简洁的 Markdown …".
            text = text + "\n\n" + CHAT_DIRECTIVE
            self._directive_sent.add(self.session_id)
        self.on_update = on_update
        self.on_permission = on_permission
        try:
            return await self._request(
                "session/prompt",
                {"sessionId": self.session_id, "prompt": [{"type": "text", "text": text}]},
            )
        finally:
            self.on_update = None
            self.on_permission = None

    async def cancel(self) -> None:
        """Interrupt the current turn (ACP notification, no response expected).

        The in-flight session/prompt then returns with stopReason="cancelled".
        """
        if self.alive and self.session_id:
            await self._write({"jsonrpc": "2.0", "method": "session/cancel",
                               "params": {"sessionId": self.session_id}})


def _chunk_text(update: dict) -> str:
    c = update.get("content") or {}
    return c.get("text", "") if isinstance(c, dict) else ""


_DETAIL_MAX = 4000  # cap per tool card — history replay of long sessions must not ship MBs


def _tool_detail(u: dict, max_chars: int = _DETAIL_MAX) -> str:
    """Human-readable command + output for a tool_call update, so the UI can
    expand the one-line tool card. Best-effort over the ACP shape: `rawInput`
    (the tool's arguments, e.g. a shell `command`) plus any text in `content`
    blocks (the tool's output). Truncated (head+tail) to keep payloads bounded."""
    parts: list[str] = []
    ri = u.get("rawInput")
    if isinstance(ri, dict):
        cmd = ri.get("command") or ri.get("cmd")
        if cmd:
            parts.append(str(cmd))
        elif ri:
            try:
                parts.append(json.dumps(ri, ensure_ascii=False, indent=2))
            except (TypeError, ValueError):
                pass
    elif isinstance(ri, str) and ri.strip():
        parts.append(ri)
    out: list[str] = []
    for blk in u.get("content") or []:
        if not isinstance(blk, dict):
            continue
        c = blk.get("content")
        if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
            out.append(c["text"])
        elif blk.get("type") == "diff" and blk.get("path"):
            out.append(f"[diff] {blk['path']}")
    if out:
        parts.append("\n".join(o for o in out if o))
    s = "\n\n".join(p for p in parts if p).strip()
    if len(s) > max_chars:
        keep = max_chars // 2
        s = s[:keep] + f"\n\n…（省略 {len(s) - max_chars} 字）…\n\n" + s[-keep:]
    return s


def _clean_title(t: str) -> str:
    """Drop a leading '[System] …' steering block from legacy titles (sessions created
    while the directive was prepended). New sessions append it, so their title is clean."""
    t = (t or "").strip()
    if t.startswith("[System]"):
        i = t.find("\n\n")
        t = t[i + 2 :].strip() if i != -1 else ""   # truncated all-directive title → empty
    return t or "新会话"


def _sess_brief(s: dict) -> dict:
    """Store row -> what the sidebar needs. `lastActive` is epoch seconds; the client
    parses an ISO string. Title can be absent (hermes auto-titles lazily, and it does
    lose them), so fall back to the first words of the conversation rather than
    rendering a wall of identical "新会话"."""
    ts = s.get("lastActive") or 0
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
    title = _clean_title(s.get("title")) if s.get("title") else ""
    if not title or title == "新会话":
        preview = _clean_title(s.get("preview") or "")
        title = preview[:40] if preview and preview != "新会话" else "新会话"
    return {"id": s.get("id"), "title": title, "updatedAt": iso}


async def _handle_session_op(ws: web.WebSocketResponse, acp: "HermesACP", op: str, payload: dict) -> None:
    if op == "session_list":
        items = await acp.list_sessions()
        items = sorted(items, key=lambda s: s.get("lastActive") or 0, reverse=True)
        await ws.send_json({"type": "sessions", "items": [_sess_brief(s) for s in items],
                            "current": acp.session_id})
    elif op == "session_new":
        # Deferred: no row exists until the first prompt, so `id` is null here. The client
        # shows an empty chat; the turn-end session_list refresh fills in the real id.
        await acp.new_session()
        await ws.send_json({"type": "session_switched", "id": None, "title": "新会话", "fresh": True})
    elif op == "session_delete":
        sid = payload.get("id") or ""
        await acp.delete_session(sid)
        await ws.send_json({"type": "session_deleted", "id": sid})
    elif op == "session_load":
        sid = payload.get("id") or ""
        await ws.send_json({"type": "history_start", "id": sid})

        async def on_hist(u: dict) -> None:
            kind = u.get("sessionUpdate")
            if kind == "user_message_chunk":
                await ws.send_json({"type": "hist", "role": "user", "text": _chunk_text(u)})
            elif kind == "agent_message_chunk":
                await ws.send_json({"type": "hist", "role": "assistant", "text": _chunk_text(u)})
            elif kind == "agent_thought_chunk":
                await ws.send_json({"type": "hist", "role": "thought", "text": _chunk_text(u)})
            elif kind in ("tool_call", "tool_call_update"):
                st = u.get("status", "")
                # Detail only on the terminal state — intermediate updates would re-ship the
                # (growing) output for the same tool. Include the id so the client dedups.
                await ws.send_json({"type": "hist", "role": "tool", "id": u.get("toolCallId", ""),
                                    "title": u.get("title", ""), "status": st,
                                    "detail": _tool_detail(u) if st in ("completed", "failed") else ""})

        await acp.load_session(sid, on_hist)
        await ws.send_json({"type": "history_done", "id": sid})


async def handle_chat(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    acp: HermesACP = request.app["acp"]
    perm_waiters: dict[str, asyncio.Future] = {}
    turn: dict[str, asyncio.Task | None] = {"task": None}

    async def on_update(u: dict) -> None:
        kind = u.get("sessionUpdate")
        if kind == "agent_message_chunk":
            await ws.send_json({"type": "token", "text": _chunk_text(u)})
        elif kind == "agent_thought_chunk":
            await ws.send_json({"type": "thought", "text": _chunk_text(u)})
        elif kind in ("tool_call", "tool_call_update"):
            await ws.send_json({"type": "tool", "title": u.get("title", ""),
                                "status": u.get("status", ""), "id": u.get("toolCallId", ""),
                                "detail": _tool_detail(u)})

    async def on_permission(params: dict):
        # Ask the user (no --yolo): forward options, await their pick.
        req_id = params.get("toolCall", {}).get("toolCallId") or str(len(perm_waiters))
        fut = asyncio.get_event_loop().create_future()
        perm_waiters[req_id] = fut
        await ws.send_json({"type": "permission", "reqId": req_id,
                            "title": params.get("toolCall", {}).get("title", "请求授权"),
                            "options": params.get("options", [])})
        try:
            return await asyncio.wait_for(fut, timeout=300)
        except asyncio.TimeoutError:
            return None
        finally:
            perm_waiters.pop(req_id, None)

    async def _kill_turn(reselect: bool) -> bool:
        """Reliably end the in-flight turn. Try graceful session/cancel first; if hermes
        doesn't yield quickly (a long-running shell tool can't be interrupted mid-command),
        hard-restart the process so the turn is killed and new work isn't queued behind it
        ("Queued for the next turn"). Returns True if it force-killed (caller must emit 'done'
        since run_turn was cancelled before it could). reselect keeps the user in the session."""
        t = turn["task"]
        turn["task"] = None
        if not (t and not t.done()):
            return False
        sid = acp.session_id
        await acp.cancel()
        try:
            # Give session/cancel a REALISTIC grace period. 1.5s meant nearly every Stop
            # escalated to the process-kill path: interrupting a running tool + winding the
            # turn down takes seconds. (Historically cancel ALSO crashed on interrupt —
            # final_response=None → None.startswith, patched in the hermes venv — so the
            # timeout fired 100% of the time and Stop always nuked the process, losing the
            # in-flight turn's messages.) Restart is the last resort, not the normal path.
            await asyncio.wait_for(asyncio.shield(t), timeout=12.0)
            return False                              # hermes honored cancel; run_turn sent done/error
        except asyncio.TimeoutError:
            log.warning("session/cancel not honored in 12s — hard-restarting hermes acp")
        await acp.restart()                           # force: kill the stuck turn + its tool
        if not t.done():
            t.cancel()
        if reselect and sid:
            async def _sink(_u: dict) -> None:
                return None
            try:
                await acp.load_session(sid, _sink)    # re-select silently (discard the replay)
            except Exception:  # noqa: BLE001
                log.debug("re-select after restart failed", exc_info=True)
        return True

    async def run_turn(text: str) -> None:
        # Booting = hermes not up yet (first turn / after a respawn): the
        # ensure() inside prompt() will spend a few seconds starting it.
        await ws.send_json({"type": "start", "booting": not (acp.alive and acp.session_id)})
        try:
            result = await acp.prompt(text, on_update, on_permission)
            if isinstance(result, dict) and result.get("_error"):
                await ws.send_json({"type": "error", "error": str(result["_error"])})
            else:
                await ws.send_json({"type": "done", "stopReason": (result or {}).get("stopReason")})
        except Exception as e:  # noqa: BLE001
            log.exception("chat turn failed")
            await ws.send_json({"type": "error", "error": str(e)})

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            ptype = payload.get("type")
            if ptype == "permission_response":
                fut = perm_waiters.get(payload.get("reqId"))
                if fut and not fut.done():
                    fut.set_result(payload.get("optionId"))
                continue
            if ptype == "stop":
                # Reliably interrupt (graceful cancel, else hard-restart) so the turn really
                # ends and the UI unblocks — session/cancel alone can't stop a running tool.
                if await _kill_turn(reselect=True):
                    await ws.send_json({"type": "done"})
                continue
            if ptype in ("session_list", "session_new", "session_load", "session_delete"):
                # session_list is read-only — safe to run while a turn streams (so the
                # dropdown can open mid-answer). Creating/loading/deleting changes the
                # active session, so first abandon any in-flight turn.
                busy = bool(turn["task"] and not turn["task"].done())
                if busy and ptype != "session_list":
                    # Switching/creating/deleting changes the active session — end the current
                    # turn reliably first (the op re-establishes the session, so no reselect).
                    await _kill_turn(reselect=False)
                try:
                    await _handle_session_op(ws, acp, ptype, payload)
                except Exception as e:  # noqa: BLE001
                    log.exception("session op failed")
                    await ws.send_json({"type": "error", "error": str(e)})
                continue
            if ptype != "msg":
                continue
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            if turn["task"] and not turn["task"].done():
                continue  # a turn is already running (UI shows stop, not send)
            if not read_chat_config()["chat_ready"]:
                await ws.send_json({"type": "need_key"})
                continue
            # Run the turn as a task so this loop keeps reading (stop / permission).
            turn["task"] = asyncio.create_task(run_turn(text))
    finally:
        t = turn["task"]
        if t and not t.done():
            t.cancel()
        if not ws.closed:
            await ws.close()
    return ws


# --------------------------------------------------------------------------- #
# Service discovery + reverse proxy — the browser-like viewer                  #
# --------------------------------------------------------------------------- #
# Services launched in the console bind to localhost inside the pod; only :8080
# is reachable from outside. So we reverse-proxy them under /proxy/<port>/ and
# the UI shows each in an iframe tab, like browser tabs into pod services.

# Ports we never surface as "services".
_HIDDEN_PORTS = {22}
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length",
}


def _entry(port: int, name: str) -> dict:
    return {"port": port, "proc": name, "url": f"/proxy/{port}/",
            "label": name or f"localhost:{port}"}


def _discover_psutil(found: dict[int, dict]) -> None:
    import psutil

    for c in psutil.net_connections(kind="inet"):
        if c.status != psutil.CONN_LISTEN or not c.laddr:
            continue
        ip, port = c.laddr.ip, c.laddr.port
        if ip not in ("127.0.0.1", "0.0.0.0", "::", "::1"):
            continue
        if port == PORT or port in _HIDDEN_PORTS or port in found:
            continue
        name = ""
        if c.pid:
            try:
                name = psutil.Process(c.pid).name()
            except Exception:  # noqa: BLE001
                pass
        found[port] = _entry(port, name)


def _discover_lsof(found: dict[int, dict]) -> None:
    # Fallback for when psutil.net_connections lacks privileges (e.g. macOS dev).
    import subprocess

    out = subprocess.run(
        ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"],
        capture_output=True, text=True, timeout=5,
    ).stdout
    for line in out.splitlines()[1:]:
        cols = line.split()
        if len(cols) < 9:
            continue
        name, addr = cols[0], cols[8]
        m = re.search(r":(\d+)$", addr)
        if not m:
            continue
        port = int(m.group(1))
        if port == PORT or port in _HIDDEN_PORTS or port in found:
            continue
        found[port] = _entry(port, name)


def discover_services() -> list[dict]:
    """List local TCP LISTEN ports that look like web services (best-effort)."""
    found: dict[int, dict] = {}
    for fn in (_discover_psutil, _discover_lsof):
        try:
            fn(found)
            if found:
                break
        except Exception as e:  # noqa: BLE001 — discovery is best-effort
            log.debug("%s failed: %s", fn.__name__, e)
    return [found[p] for p in sorted(found)]


async def handle_services(_request: web.Request) -> web.Response:
    return web.json_response({"services": discover_services()})


def _rewrite_html(body: bytes, prefix: str) -> bytes:
    """Make a proxied page resolve assets under /proxy/<port>/ in the iframe."""
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return body
    # Rewrite root-absolute attribute URLs (/foo -> /proxy/<port>/foo); skip //host.
    # Do this BEFORE injecting <base> so the base href isn't itself rewritten.
    text = re.sub(r'((?:src|href|action)\s*=\s*["\'])/(?!/)', r"\1" + prefix, text)
    base = f'<base href="{prefix}">'
    if "<head>" in text:
        text = text.replace("<head>", "<head>" + base, 1)
    elif re.search(r"<head[^>]*>", text):
        text = re.sub(r"(<head[^>]*>)", r"\1" + base, text, count=1)
    else:
        text = base + text
    return text.encode("utf-8")


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    port = int(request.match_info["port"])
    tail = request.match_info.get("tail", "")
    prefix = f"/proxy/{port}/"
    qs = request.query_string
    target = f"http://127.0.0.1:{port}/{tail}" + (f"?{qs}" if qs else "")
    session: aiohttp.ClientSession = request.app["proxy_session"]

    # WebSocket upgrade → bridge both directions.
    if request.headers.get("Upgrade", "").lower() == "websocket":
        ws_server = web.WebSocketResponse()
        await ws_server.prepare(request)
        ws_url = "ws://127.0.0.1:%d/%s" % (port, tail) + (f"?{qs}" if qs else "")
        try:
            async with session.ws_connect(ws_url) as ws_client:
                async def s2c() -> None:
                    async for m in ws_server:
                        if m.type == WSMsgType.TEXT:
                            await ws_client.send_str(m.data)
                        elif m.type == WSMsgType.BINARY:
                            await ws_client.send_bytes(m.data)
                        else:
                            break
                async def c2s() -> None:
                    async for m in ws_client:
                        if m.type == WSMsgType.TEXT:
                            await ws_server.send_str(m.data)
                        elif m.type == WSMsgType.BINARY:
                            await ws_server.send_bytes(m.data)
                        else:
                            break
                await asyncio.gather(s2c(), c2s())
        except Exception as e:  # noqa: BLE001
            log.debug("ws proxy to :%d failed: %s", port, e)
        return ws_server

    req_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_BY_HOP and k.lower() != "host"}
    body = await request.read()
    try:
        async with session.request(
            request.method, target, headers=req_headers, data=body or None,
            allow_redirects=False,
        ) as upstream:
            ctype = upstream.headers.get("Content-Type", "").lower()
            resp_headers = {k: v for k, v in upstream.headers.items()
                            if k.lower() not in _HOP_BY_HOP}
            # Keep redirects inside the proxy.
            loc = upstream.headers.get("Location")
            if loc and loc.startswith("/") and not loc.startswith("//"):
                resp_headers["Location"] = prefix + loc[1:]

            # HTML: buffer + rewrite (inject <base>, fix root-absolute URLs).
            if "text/html" in ctype:
                raw = _rewrite_html(await upstream.read(), prefix)
                return web.Response(status=upstream.status, body=raw, headers=resp_headers)

            # Everything else: STREAM chunk-by-chunk so never-ending responses work
            # (MJPEG multipart camera feeds, SSE, large downloads) instead of being
            # buffered to completion — which would hang an infinite stream forever.
            resp = web.StreamResponse(status=upstream.status, headers=resp_headers)
            await resp.prepare(request)
            try:
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
            except (aiohttp.ClientError, ConnectionResetError):
                pass
            await resp.write_eof()
            return resp
    except aiohttp.ClientError as e:
        return web.Response(status=502, text=f"proxy error to localhost:{port}: {e}")


async def _redirect_to_slash(request: web.Request) -> web.Response:
    raise web.HTTPFound(f"/proxy/{request.match_info['port']}/")


# --------------------------------------------------------------------------- #
# Auth (single-user HTTP Basic) + presence channel (no lock)                   #
# --------------------------------------------------------------------------- #
async def handle_health(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


@web.middleware
async def auth_middleware(request: web.Request, handler):
    # /healthz must stay open so LB / k8s health checks don't get 401'd.
    if request.path == "/healthz":
        return await handler(request)
    if AUTH_ENABLED:
        hdr = request.headers.get("Authorization", "")
        ok = False
        if hdr.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(hdr[6:]).decode("utf-8", "replace").partition(":")
                ok = hmac.compare_digest(user, AUTH_USER) and hmac.compare_digest(pw, AUTH_PASS)
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            return web.Response(
                status=401, text="Authentication required",
                headers={"WWW-Authenticate": 'Basic realm="LeRobot Agent Console"'},
            )
    resp = await handler(request)
    # The UI assets (index.html, app.js, style.css, release_note.md) change on every
    # redeploy; without this the browser serves a stale cached bundle and the user has to
    # hard-refresh to see updates. `no-cache` = revalidate on each load (conditional GET →
    # cheap 304 via Last-Modified when unchanged), not `no-store`, so it stays fast.
    if request.path == "/" or request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


async def handle_control(request: web.Request) -> web.WebSocketResponse:
    """Presence channel — no lock.

    Multiple windows/users may be open at once, so this just grants immediately
    and holds the socket open. (Older UIs still open this on load; they now always
    get 'granted' and are never denied/evicted.)
    """
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    await ws.send_json({"type": "granted"})
    try:
        async for msg in ws:
            if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    finally:
        if not ws.closed:
            await ws.close()
    return ws


# --------------------------------------------------------------------------- #
# App                                                                         #
# --------------------------------------------------------------------------- #
async def _on_startup(app: web.Application) -> None:
    app["proxy_session"] = aiohttp.ClientSession(auto_decompress=True)
    app["acp"] = HermesACP()
    # Only start hermes once a Volcengine key exists, so the agent always boots
    # WITH credentials. Warm up in the BACKGROUND so the server starts serving the
    # UI (terminal, viewer) immediately instead of blocking on the ~6s acp boot.
    if read_chat_config()["chat_ready"]:
        async def _warm() -> None:
            try:
                # Process only — NOT ensure(): that also opens a session, and a server that
                # boots without anyone chatting would leave an empty one behind every time.
                await app["acp"]._ensure_proc()
            except Exception as e:  # noqa: BLE001
                log.warning("hermes acp warmup failed (will retry on first chat): %s", e)
        app["acp_warm"] = asyncio.create_task(_warm())
    else:
        log.info("no Volcengine key yet — hermes acp will start when the key is set")


async def _on_cleanup(app: web.Application) -> None:
    await app["proxy_session"].close()
    acp: HermesACP = app.get("acp")
    if acp and acp.alive:
        try:
            acp.proc.terminate()
        except ProcessLookupError:
            pass


def build_app() -> web.Application:
    app = web.Application(middlewares=[auth_middleware])
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_get("/", handle_index)
    app.router.add_get("/healthz", handle_health)   # unauthenticated — for LB/k8s probes
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/version", handle_version)
    app.router.add_get("/api/models", handle_models)
    app.router.add_post("/api/model", handle_set_model)
    app.router.add_get(r"/api/lerobot-doc/{name}", handle_lerobot_doc)
    app.router.add_get("/api/services", handle_services)
    app.router.add_post("/api/volcano-key", handle_volcano_key)
    app.router.add_get("/ws/control", handle_control)
    app.router.add_get("/ws/term", handle_term)
    app.router.add_get("/ws/chat", handle_chat)
    app.router.add_get(r"/proxy/{port:\d+}", _redirect_to_slash)
    app.router.add_route("*", r"/proxy/{port:\d+}/{tail:.*}", handle_proxy)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app


def main() -> None:
    ssl_ctx = None
    if _tls_enabled():
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(TLS_CERT, TLS_KEY)
    scheme = "https" if ssl_ctx else "http"
    log.info("LeRobot Agent Console %s://0.0.0.0:%s  shell=%s  LEROBOT_HOME=%s  hermes=%s",
             scheme, PORT, SHELL, WORKDIR, HERMES_BIN)
    if AUTH_ENABLED:
        log.info("auth: single-user HTTP Basic ENABLED (user=%s)", AUTH_USER)
    else:
        log.warning("auth: DISABLED — set CONSOLE_USER + CONSOLE_PASSWORD to protect the console")
    if not ssl_ctx:
        log.warning("serving plain HTTP (no TLS) — traffic is UNENCRYPTED. Set CONSOLE_TLS_CERT + "
                    "CONSOLE_TLS_KEY to serve HTTPS.")
    web.run_app(build_app(), host="0.0.0.0", port=PORT, ssl_context=ssl_ctx, access_log=None)


if __name__ == "__main__":
    main()
