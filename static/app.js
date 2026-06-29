/* LeRobot Agent Console — front-end glue.
 *   - terminal: xterm.js <-> WS /ws/term  (a real PTY, the "ssh console")
 *   - chat:     WS /ws/chat <-> hermes agent
 *   - first chat use prompts for the Volcengine (Ark) api key (chat only)
 */
(() => {
  "use strict";

  const wsURL = (path) => (location.protocol === "https:" ? "wss://" : "ws://") + location.host + path;

  // ----------------------------------------------------------------- terminal
  const term = new Terminal({
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: 13.5,
    theme: { background: "#0b0f17", foreground: "#d6deeb", cursor: "#5aa7ff" },
    cursorBlink: true,
    convertEol: true,
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(document.getElementById("terminal"));
  fit.fit();

  let termWS;
  function connectTerm() {
    termWS = new WebSocket(wsURL("/ws/term"));
    termWS.binaryType = "arraybuffer";
    termWS.onopen = () => sendResize();
    termWS.onmessage = (e) => {
      if (typeof e.data === "string") term.write(e.data);
      else term.write(new Uint8Array(e.data));
    };
    termWS.onclose = () => {
      term.write("\r\n\x1b[33m[console disconnected — reconnecting…]\x1b[0m\r\n");
      setTimeout(connectTerm, 1500);
    };
  }
  function sendResize() {
    if (termWS && termWS.readyState === 1)
      termWS.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
  }
  term.onData((d) => {
    if (termWS && termWS.readyState === 1) termWS.send(JSON.stringify({ type: "input", data: d }));
  });
  window.addEventListener("resize", () => { fit.fit(); sendResize(); });
  document.getElementById("term-clear").onclick = () => term.clear();
  connectTerm();

  // --------------------------------------------------------------------- chat
  const body = document.getElementById("chat-body");
  const textEl = document.getElementById("chat-text");
  const sendBtn = document.getElementById("chat-send");
  let chatWS, busy = false, thinkingEl = null, chatReady = false;

  function addMsg(role, text) {
    const wrap = document.createElement("div");
    wrap.className = "msg " + (role === "user" ? "msg-user" : "msg-bot");
    const ava = document.createElement("span");
    ava.className = "msg-ava";
    ava.textContent = role === "user" ? "你" : "L";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.append(ava, bubble);
    body.appendChild(wrap);
    body.scrollTop = body.scrollHeight;
    return bubble;
  }
  function showThinking() {
    thinkingEl = addMsg("bot", "Agent 正在思考");
    thinkingEl.classList.add("thinking");
  }
  function clearThinking() {
    if (thinkingEl) { thinkingEl.closest(".msg").remove(); thinkingEl = null; }
  }
  function setBusy(b) {
    busy = b;
    sendBtn.disabled = b;
  }

  function connectChat() {
    chatWS = new WebSocket(wsURL("/ws/chat"));
    chatWS.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === "start") { showThinking(); }
      else if (m.type === "answer") { clearThinking(); addMsg("bot", m.text || "(空响应)"); }
      else if (m.type === "done") { setBusy(false); }
      else if (m.type === "error") { clearThinking(); addMsg("bot", "⚠️ " + m.error); setBusy(false); }
      else if (m.type === "need_key") { clearThinking(); setBusy(false); openKeyModal(); }
    };
    chatWS.onclose = () => setTimeout(connectChat, 1500);
  }
  connectChat();

  async function send() {
    const text = textEl.value.trim();
    if (!text || busy) return;
    if (!chatReady) { pendingText = text; openKeyModal(); return; }
    addMsg("user", text);
    textEl.value = ""; textEl.style.height = "auto";
    setBusy(true);
    if (chatWS.readyState !== 1) connectChat();
    chatWS.send(JSON.stringify({ type: "msg", text }));
  }
  sendBtn.onclick = send;
  textEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  textEl.addEventListener("input", () => {
    textEl.style.height = "auto";
    textEl.style.height = Math.min(textEl.scrollHeight, 140) + "px";
  });
  document.getElementById("chat-quick").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-q]");
    if (!b) return;
    textEl.value = b.dataset.q;
    textEl.dispatchEvent(new Event("input"));
    send();
  });

  // ------------------------------------------------------------- key handling
  let pendingText = null;
  const modal = document.getElementById("key-modal");
  const keyInput = document.getElementById("key-input");
  const keyErr = document.getElementById("key-err");
  const saveBtn = document.getElementById("key-save");

  function openKeyModal() { modal.hidden = false; keyErr.hidden = true; keyInput.focus(); }
  function closeKeyModal() { modal.hidden = true; }
  document.getElementById("chat-gear").onclick = openKeyModal;
  document.getElementById("key-cancel").onclick = closeKeyModal;

  saveBtn.onclick = async () => {
    const api_key = keyInput.value.trim();
    if (!api_key) { keyErr.textContent = "请输入 API Key"; keyErr.hidden = false; return; }
    saveBtn.disabled = true; saveBtn.textContent = "保存中…";
    try {
      const r = await fetch("/api/volcano-key", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key,
          base_url: document.getElementById("key-baseurl").value.trim() || undefined,
          model: document.getElementById("key-model").value.trim() || undefined,
        }),
      });
      const data = await r.json();
      if (!data.ok) throw new Error(data.error || "保存失败");
      chatReady = true;
      if (data.model) document.getElementById("chat-model").textContent = data.model;
      closeKeyModal();
      if (pendingText) { textEl.value = pendingText; pendingText = null; send(); }
    } catch (err) {
      keyErr.textContent = "⚠️ " + err.message; keyErr.hidden = false;
    } finally {
      saveBtn.disabled = false; saveBtn.textContent = "保存并开始";
    }
  };

  // initial status: know whether chat is ready and which model
  fetch("/api/status").then((r) => r.json()).then((s) => {
    chatReady = !!s.chat_ready;
    if (s.model) document.getElementById("chat-model").textContent = s.model;
    document.getElementById("chat-status").style.background = chatReady ? "" : "#c2c7d2";
  }).catch(() => {});
})();
