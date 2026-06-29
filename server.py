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
import fcntl
import json
import logging
import os
import re
import shutil
import signal
import struct
import termios
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agent_console")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

PORT = int(os.environ.get("PORT", "8080"))
SHELL = os.environ.get("CONSOLE_SHELL") or shutil.which("bash") or "/bin/sh"
WORKDIR = os.environ.get("CONSOLE_WORKDIR", os.environ.get("LEROBOT_HOME", os.getcwd()))
HERMES_BIN = os.environ.get("HERMES_BIN") or shutil.which("hermes") or "hermes"
# Skill to preload so the agent knows how to drive LeRobot SFT (requirement f).
CHAT_SKILL = os.environ.get("HERMES_CHAT_SKILL", "robot_sft")

# One-time steer (first turn of a session) so answers render in the left viewer.
HTML_DIRECTIVE = (
    "[System] 请用一个自包含的 HTML 片段作答（可含内联 CSS/JS，不要 markdown 代码围栏），"
    "以便直接在面板中渲染。普通问答用简洁排版即可；涉及数据/表格/状态时尽量结构化展示。\n\n"
)

# Ark / Volcengine OpenAI-compatible endpoint and a sensible default model.
DEFAULT_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
DEFAULT_MODEL = os.environ.get("ARK_MODEL", "doubao-seed-2-0-pro-260215")

# Placeholder values that mean "no real key yet" — treat chat as not-ready.
_PLACEHOLDERS = {"", "your-api-key", "changeme", "<set-me>", "null", "none"}


# --------------------------------------------------------------------------- #
# hermes config helpers (chat only)                                           #
# --------------------------------------------------------------------------- #
def _hermes_config_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    return home / "config.yaml"


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
    await _hermes_config_set("model.base_url", base_url or DEFAULT_BASE_URL)
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
    return web.json_response(info)


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
        self._first_turn = True
        # Per-turn callbacks (single-user console → one active turn at a time):
        self.on_update = None       # async fn(update dict) — stream notifications
        self.on_permission = None   # async fn(params) -> optionId|None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def ensure(self) -> None:
        async with self._start_lock:
            if self.alive and self.session_id:
                return
            await self._spawn()

    async def restart(self) -> None:
        """Used after the Volcengine key changes (a new session picks it up)."""
        async with self._start_lock:
            if self.alive:
                try:
                    self.proc.terminate()
                except ProcessLookupError:
                    pass
            self.proc = None
            self.session_id = None
            self._first_turn = True
            await self._spawn()

    async def _spawn(self) -> None:
        env = dict(os.environ)
        env["HERMES_ACCEPT_HOOKS"] = "1"
        env.setdefault("NO_COLOR", "1")
        self.proc = await asyncio.create_subprocess_exec(
            HERMES_BIN, "acp", "--accept-hooks",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, cwd=WORKDIR, env=env,
        )
        self._pending = {}
        asyncio.create_task(self._read_loop(self.proc))
        await self._request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        res = await self._request("session/new", {"cwd": WORKDIR, "mcpServers": []})
        self.session_id = res.get("sessionId")
        self._first_turn = True
        log.info("hermes acp ready: session=%s", self.session_id)

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

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                fut = self._pending.pop(msg["id"], None)
                if fut and not fut.done():
                    fut.set_result(msg.get("result") or {"_error": msg.get("error")})
            elif msg.get("method") == "session/update":
                if self.on_update:
                    try:
                        await self.on_update(msg["params"].get("update", {}))
                    except Exception:  # noqa: BLE001
                        log.debug("on_update failed", exc_info=True)
            elif msg.get("method") == "session/request_permission":
                await self._reply_permission(msg)
            elif "id" in msg:  # server->client request we don't implement (e.g. fs/*)
                await self._write({"jsonrpc": "2.0", "id": msg["id"],
                                   "error": {"code": -32601, "message": "unsupported"}})
        # process exited — drop the session so the next turn respawns
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("hermes acp exited"))
        self._pending = {}
        self.session_id = None
        log.warning("hermes acp process exited")

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
        # Steer the agent to answer in HTML once at the start of the session.
        if self._first_turn:
            text = HTML_DIRECTIVE + text
            self._first_turn = False
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


def _chunk_text(update: dict) -> str:
    c = update.get("content") or {}
    return c.get("text", "") if isinstance(c, dict) else ""


async def handle_chat(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    acp: HermesACP = request.app["acp"]
    lock = asyncio.Lock()                # one turn at a time per connection
    perm_waiters: dict[str, asyncio.Future] = {}

    async def on_update(u: dict) -> None:
        kind = u.get("sessionUpdate")
        if kind == "agent_message_chunk":
            await ws.send_json({"type": "token", "text": _chunk_text(u)})
        elif kind == "agent_thought_chunk":
            await ws.send_json({"type": "thought", "text": _chunk_text(u)})
        elif kind in ("tool_call", "tool_call_update"):
            await ws.send_json({"type": "tool", "title": u.get("title", ""),
                                "status": u.get("status", ""), "id": u.get("toolCallId", "")})

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
            if ptype != "msg":
                continue
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            if not read_chat_config()["chat_ready"]:
                await ws.send_json({"type": "need_key"})
                continue
            async with lock:
                # Booting = hermes not up yet (first turn / after a respawn): the
                # ensure() inside prompt() will spend a few seconds starting it.
                booting = not (acp.alive and acp.session_id)
                await ws.send_json({"type": "start", "booting": booting})
                try:
                    result = await acp.prompt(text, on_update, on_permission)
                    if isinstance(result, dict) and result.get("_error"):
                        await ws.send_json({"type": "error", "error": str(result["_error"])})
                    else:
                        await ws.send_json({"type": "done", "stopReason": (result or {}).get("stopReason")})
                except Exception as e:  # noqa: BLE001
                    log.exception("chat turn failed")
                    await ws.send_json({"type": "error", "error": str(e)})
    finally:
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
            raw = await upstream.read()
            ctype = upstream.headers.get("Content-Type", "")
            if "text/html" in ctype.lower():
                raw = _rewrite_html(raw, prefix)
            resp_headers = {k: v for k, v in upstream.headers.items()
                            if k.lower() not in _HOP_BY_HOP}
            # Keep redirects inside the proxy.
            loc = upstream.headers.get("Location")
            if loc and loc.startswith("/") and not loc.startswith("//"):
                resp_headers["Location"] = prefix + loc[1:]
            return web.Response(status=upstream.status, body=raw, headers=resp_headers)
    except aiohttp.ClientError as e:
        return web.Response(status=502, text=f"proxy error to localhost:{port}: {e}")


async def _redirect_to_slash(request: web.Request) -> web.Response:
    raise web.HTTPFound(f"/proxy/{request.match_info['port']}/")


# --------------------------------------------------------------------------- #
# App                                                                         #
# --------------------------------------------------------------------------- #
async def _on_startup(app: web.Application) -> None:
    app["proxy_session"] = aiohttp.ClientSession(auto_decompress=True)
    app["acp"] = HermesACP()
    # Only start hermes once a Volcengine key exists, so the agent always boots
    # WITH credentials. If there's no key yet, we wait: the key submission in chat
    # (POST /api/volcano-key) starts it. If a key is already configured, warm up now.
    if read_chat_config()["chat_ready"]:
        try:
            await app["acp"].ensure()
        except Exception as e:  # noqa: BLE001
            log.warning("hermes acp warmup failed (will retry on first chat): %s", e)
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
    app = web.Application()
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/services", handle_services)
    app.router.add_post("/api/volcano-key", handle_volcano_key)
    app.router.add_get("/ws/term", handle_term)
    app.router.add_get("/ws/chat", handle_chat)
    app.router.add_get(r"/proxy/{port:\d+}", _redirect_to_slash)
    app.router.add_route("*", r"/proxy/{port:\d+}/{tail:.*}", handle_proxy)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app


def main() -> None:
    log.info("LeRobot Agent Console on :%s  shell=%s  workdir=%s  hermes=%s", PORT, SHELL, WORKDIR, HERMES_BIN)
    web.run_app(build_app(), host="0.0.0.0", port=PORT, access_log=None)


if __name__ == "__main__":
    main()
