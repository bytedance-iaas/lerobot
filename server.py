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

# The console is single-user: we keep one rolling hermes session id so chat
# context carries across turns. hermes prints `session_id: <id>` to stderr in
# quiet mode; we capture it on the first turn and `--resume` it afterwards.
_SESSION_RE = re.compile(r"session_id:\s*(\S+)")
_CHAT_STATE: dict[str, str | None] = {"session_id": None}
# Tri-state cache for "is CHAT_SKILL installed?": None=unknown, True/False once checked.
# Passing `-s <skill>` for an uninstalled skill makes hermes hard-error
# ("Unknown skill(s): ...") instead of answering, so we only pass it when present.
_SKILL_OK: list[bool | None] = [None]
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


def read_chat_config() -> dict:
    """Return {chat_ready, model, base_url} by reading hermes' config.yaml."""
    cfg_path = _hermes_config_path()
    model = DEFAULT_MODEL
    base_url = DEFAULT_BASE_URL
    api_key = ""
    try:
        import yaml

        data = yaml.safe_load(cfg_path.read_text()) or {}
        m = data.get("model")
        if isinstance(m, dict):
            model = m.get("default") or model
            base_url = m.get("base_url") or base_url
            api_key = (m.get("api_key") or "").strip()
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001 — config is best-effort
        log.warning("could not read hermes config: %s", e)
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
# Chat websocket — bridge to the hermes agent                                 #
# --------------------------------------------------------------------------- #
async def _skill_available() -> bool:
    """Whether CHAT_SKILL is installed (cached). Empty CHAT_SKILL => disabled."""
    if not CHAT_SKILL:
        return False
    if _SKILL_OK[0] is None:
        try:
            proc = await asyncio.create_subprocess_exec(
                HERMES_BIN, "skills", "list",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            _SKILL_OK[0] = CHAT_SKILL.lower() in out.decode(errors="replace").lower()
        except Exception:  # noqa: BLE001
            _SKILL_OK[0] = False
        if not _SKILL_OK[0]:
            log.warning("hermes skill %r not installed — chat runs without it", CHAT_SKILL)
    return bool(_SKILL_OK[0])


def _build_chat_cmd(text: str, resume: str | None, with_skill: bool) -> list[str]:
    cmd = [
        HERMES_BIN, "chat",
        "-q", text,
        "-Q",          # quiet: final response on stdout, `session_id:` on stderr
    ]
    if resume:
        cmd += ["--resume", resume]  # carry context across turns
    if with_skill and CHAT_SKILL:
        cmd += ["-s", CHAT_SKILL]
    return cmd


async def _spawn_hermes(text: str, resume: str | None, with_skill: bool) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["HERMES_ACCEPT_HOOKS"] = "1"
    env.setdefault("NO_COLOR", "1")
    proc = await asyncio.create_subprocess_exec(
        *_build_chat_cmd(text, resume, with_skill), cwd=WORKDIR, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def _run_hermes_turn(ws: web.WebSocketResponse, text: str) -> None:
    if not read_chat_config()["chat_ready"]:
        await ws.send_json({"type": "need_key"})
        return

    await ws.send_json({"type": "start"})
    resume = _CHAT_STATE["session_id"]
    with_skill = await _skill_available()
    # First turn of a session: steer the agent to answer in self-contained HTML so
    # the left viewer can render it. With --resume the steer persists, so set once.
    prompt = text if resume else (HTML_DIRECTIVE + text)
    try:
        rc, out, err = await _spawn_hermes(prompt, resume, with_skill)
    except FileNotFoundError:
        await ws.send_json({"type": "error", "error": f"hermes not found ({HERMES_BIN})"})
        return

    # Skill vanished between check and run (or list was stale) — retry without it.
    if with_skill and "Unknown skill" in (out + err):
        _SKILL_OK[0] = False
        with_skill = False
        rc, out, err = await _spawn_hermes(prompt, resume, False)

    # A stale/pruned session id can't be resumed — retry once as a fresh session
    # (now fresh, so re-apply the HTML steer).
    if resume and rc != 0 and "No session found" in (out + err):
        _CHAT_STATE["session_id"] = None
        rc, out, err = await _spawn_hermes(HTML_DIRECTIVE + text, None, with_skill)

    m = _SESSION_RE.search(err) or _SESSION_RE.search(out)
    if m:
        _CHAT_STATE["session_id"] = m.group(1)

    answer = out.strip()
    if rc != 0 and not answer:
        await ws.send_json({"type": "error", "error": err.strip() or f"hermes exited with code {rc}"})
        return
    await ws.send_json({"type": "answer", "text": answer})
    await ws.send_json({"type": "done"})


async def handle_chat(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    lock = asyncio.Lock()  # serialize turns so the named session stays consistent
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "msg":
                text = (payload.get("text") or "").strip()
                if not text:
                    continue
                async with lock:
                    try:
                        await _run_hermes_turn(ws, text)
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


async def _on_cleanup(app: web.Application) -> None:
    await app["proxy_session"].close()


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
