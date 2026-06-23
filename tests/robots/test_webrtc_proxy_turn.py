# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TURN support: relay-minted coturn REST/HMAC creds + ICE config distribution.

The signaling relay hands each peer, on connect, an ICE-server list (STUN + TURN with a
freshly-minted, time-limited credential). The aiortc transport merges it into the peer
connection. Here we test the credential math (no network), the ICE-config builder, and
the real relay->client distribution over a WebSocket.
"""

import base64
import hashlib
import hmac
import time

from lerobot.robots.webrtc_proxy.signaling_server import IceConfig, _coturn_credentials


def test_coturn_credentials_match_rest_api_format():
    secret = "s3cr3t"
    before = int(time.time())
    username, credential = _coturn_credentials(secret, ttl_s=600, name="sess:robot")

    expiry_str, name = username.split(":", 1)
    expiry = int(expiry_str)
    assert name == "sess:robot"
    # expiry ~ now + ttl
    assert before + 600 <= expiry <= before + 601 + 1
    # credential = base64(HMAC_SHA1(secret, username)) — exactly coturn's check.
    expected = base64.b64encode(hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()).decode()
    assert credential == expected


def test_ice_config_empty_when_unconfigured():
    ice = IceConfig()
    assert ice.enabled is False
    assert ice.for_peer("sess:robot") == []


def test_ice_config_stun_only_needs_no_secret():
    ice = IceConfig(stun_urls=["stun:stun.l.google.com:19302"])
    assert ice.enabled is True
    servers = ice.for_peer("sess:controller")
    assert servers == [{"urls": ["stun:stun.l.google.com:19302"]}]


def test_ice_config_turn_requires_secret():
    # TURN urls without a secret can't be credentialed -> not emitted, not "enabled".
    ice = IceConfig(turn_urls=["turn:turn.example.com:3478"])
    assert ice.enabled is False
    assert ice.for_peer("sess:robot") == []


def test_ice_config_turn_gets_fresh_credentials():
    ice = IceConfig(
        stun_urls=["stun:stun.example.com:3478"],
        turn_urls=["turn:turn.example.com:3478?transport=udp", "turns:turn.example.com:5349?transport=tcp"],
        turn_secret="shared",
        turn_ttl_s=300,
    )
    servers = ice.for_peer("sess:robot")
    assert {"urls": ["stun:stun.example.com:3478"]} in servers
    turn = next(s for s in servers if "username" in s)
    assert turn["urls"] == [
        "turn:turn.example.com:3478?transport=udp",
        "turns:turn.example.com:5349?transport=tcp",
    ]
    # credential verifies against the same secret.
    expected = base64.b64encode(
        hmac.new(b"shared", turn["username"].encode(), hashlib.sha1).digest()
    ).decode()
    assert turn["credential"] == expected


def test_to_ice_server_handles_str_and_dict():
    import pytest

    pytest.importorskip("aiortc", reason="needs the lerobot[webrtc] extra (aiortc)")
    from lerobot.robots.webrtc_proxy.transport import AiortcTransport

    plain = AiortcTransport._to_ice_server("stun:stun.example.com:3478")
    assert plain.username is None
    creds = AiortcTransport._to_ice_server(
        {"urls": ["turn:turn.example.com:3478"], "username": "u", "credential": "p"}
    )
    assert creds.username == "u"
    assert creds.credential == "p"


# --- relay -> client ICE distribution over a real WebSocket --------------------
import asyncio  # noqa: E402
import threading  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("aiohttp", reason="signaling needs aiohttp (lerobot[webrtc])")

from lerobot.robots.webrtc_proxy.signaling import WebSocketSignaling  # noqa: E402
from lerobot.robots.webrtc_proxy.signaling_server import start_relay  # noqa: E402


class _Loop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(
            target=lambda: (asyncio.set_event_loop(self.loop), self.loop.run_forever()), daemon=True
        ).start()

    def run(self, coro, timeout=5):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


def test_relay_hands_turn_creds_to_a_client():
    ice = IceConfig(
        stun_urls=["stun:stun.example.com:3478"],
        turn_urls=["turn:turn.example.com:3478?transport=udp"],
        turn_secret="shared",
        turn_ttl_s=300,
    )
    lt = _Loop()
    _, port = lt.run(start_relay("127.0.0.1", 0, ice=ice))
    url = f"ws://127.0.0.1:{port}/ws"

    async def _open_and_read():
        sig = WebSocketSignaling(url, "sess", role="robot")
        await sig.open()  # consumes the ICE message the relay pushes on join
        servers = sig.ice_servers
        await sig.close()
        return servers

    try:
        servers = lt.run(_open_and_read())
    finally:
        lt.stop()

    assert {"urls": ["stun:stun.example.com:3478"]} in servers
    turn = next(s for s in servers if "username" in s)
    expected = base64.b64encode(
        hmac.new(b"shared", turn["username"].encode(), hashlib.sha1).digest()
    ).decode()
    assert turn["credential"] == expected
    assert turn["username"].endswith(":sess:robot")


def test_relay_hands_empty_list_when_no_turn_configured():
    lt = _Loop()
    _, port = lt.run(start_relay("127.0.0.1", 0))  # no IceConfig
    url = f"ws://127.0.0.1:{port}/ws"

    async def _open_and_read():
        sig = WebSocketSignaling(url, "sess", role="controller")
        await sig.open()
        servers = sig.ice_servers
        await sig.close()
        return servers

    try:
        assert lt.run(_open_and_read()) == []
    finally:
        lt.stop()
