/* LeRobot Agent Console — front-end glue.
 *   - terminal: xterm.js  <-> WS /ws/term  (a real PTY, the "ssh console")
 *   - chat:     WS /ws/chat <-> hermes agent; HTML answers render in the viewer
 *   - viewer:   browser-like tabs over /proxy/<port> services + agent HTML output
 *   - splitters: drag to resize panels
 *   - first chat use prompts for the Volcengine (Ark) api key (chat only)
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const wsURL = (p) => (location.protocol === "https:" ? "wss://" : "ws://") + location.host + p;

  /* ----------------------------------------------------------------- terminal */
  const term = new Terminal({
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: 13.5,
    theme: { background: "#0b0f17", foreground: "#d6deeb", cursor: "#5aa7ff" },
    cursorBlink: true, convertEol: true,
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open($("terminal"));
  const refit = () => { try { fit.fit(); sendResize(); } catch (_) {} };
  refit();

  let termWS;
  function connectTerm() {
    termWS = new WebSocket(wsURL("/ws/term"));
    termWS.binaryType = "arraybuffer";
    termWS.onopen = () => sendResize();
    termWS.onmessage = (e) => term.write(typeof e.data === "string" ? e.data : new Uint8Array(e.data));
    termWS.onclose = () => {
      term.write("\r\n\x1b[33m[console disconnected — reconnecting…]\x1b[0m\r\n");
      setTimeout(connectTerm, 1500);
    };
  }
  function sendResize() {
    if (termWS && termWS.readyState === 1)
      termWS.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
  }
  term.onData((d) => { if (termWS && termWS.readyState === 1) termWS.send(JSON.stringify({ type: "input", data: d })); });
  window.addEventListener("resize", refit);
  $("term-clear").onclick = () => term.clear();
  connectTerm();

  /* ------------------------------------------------------------------- viewer */
  const viewerTabs = $("viewer-tabs"), viewerBody = $("viewer-body"), viewerEmpty = $("viewer-empty");
  const tabs = new Map();             // id -> { tabEl, paneEl, kind, port }
  let tabSeq = 0;

  // The viewer is a pure inline browser: no default tab. When empty, show the
  // "new tab" placeholder; otherwise the active tab's iframe fills the area.
  function activate(id) {
    if (!tabs.has(id)) id = null;
    for (const [tid, t] of tabs) {
      const on = tid === id;
      t.tabEl.classList.toggle("active", on);
      t.paneEl.classList.toggle("active", on);
    }
    viewerEmpty.hidden = tabs.size > 0;
  }

  function makeTab(id, label, { closeable = true, live = false } = {}) {
    const tabEl = document.createElement("button");
    tabEl.className = "vtab";
    tabEl.dataset.tab = id;
    tabEl.innerHTML =
      (live ? '<span class="vt-dot"></span>' : "") +
      `<span class="vt-label"></span>` +
      (closeable ? '<span class="vt-close" title="关闭">✕</span>' : "");
    tabEl.querySelector(".vt-label").textContent = label;
    viewerTabs.appendChild(tabEl);
    return tabEl;
  }

  function addHtmlTab(html, label) {
    const id = "t" + ++tabSeq;
    const tabEl = makeTab(id, label || "AI 输出 " + tabSeq);
    const pane = document.createElement("div");
    pane.className = "vpane";
    pane.dataset.pane = id;
    const iframe = document.createElement("iframe");
    iframe.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms allow-popups allow-modals");
    iframe.srcdoc = html;
    pane.appendChild(iframe);
    viewerBody.appendChild(pane);
    tabs.set(id, { tabEl, paneEl: pane, kind: "html" });
    activate(id);
    return id;
  }

  function addServiceTab(port, label) {
    for (const [tid, t] of tabs) if (t.kind === "service" && t.port === port) { activate(tid); return tid; }
    const id = "t" + ++tabSeq;
    const tabEl = makeTab(id, label || ("localhost:" + port), { live: true });
    const pane = document.createElement("div");
    pane.className = "vpane";
    pane.dataset.pane = id;
    const iframe = document.createElement("iframe");
    iframe.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-downloads");
    iframe.src = "/proxy/" + port + "/";
    pane.appendChild(iframe);
    viewerBody.appendChild(pane);
    tabs.set(id, { tabEl, paneEl: pane, kind: "service", port });
    activate(id);
    return id;
  }

  function addExternalTab(url) {
    const id = "t" + ++tabSeq;
    const tabEl = makeTab(id, url.replace(/^https?:\/\//, "").slice(0, 28));
    const pane = document.createElement("div");
    pane.className = "vpane";
    pane.dataset.pane = id;
    const iframe = document.createElement("iframe");
    iframe.src = url;
    pane.appendChild(iframe);
    viewerBody.appendChild(pane);
    tabs.set(id, { tabEl, paneEl: pane, kind: "external" });
    activate(id);
    return id;
  }

  function closeTab(id) {
    const t = tabs.get(id);
    if (!t) return;
    const wasActive = t.tabEl.classList.contains("active");
    t.tabEl.remove(); t.paneEl.remove(); tabs.delete(id);
    if (wasActive) activate([...tabs.keys()].pop() || null);
  }

  viewerTabs.addEventListener("click", (e) => {
    const closeBtn = e.target.closest(".vt-close");
    const tab = e.target.closest(".vtab");
    if (!tab) return;
    if (closeBtn) { closeTab(tab.dataset.tab); return; }
    activate(tab.dataset.tab);
  });

  // open <url-or-port> heuristics
  function openTarget(raw) {
    const s = (raw || "").trim();
    if (!s) return;
    let m;
    if (/^\d+$/.test(s)) return addServiceTab(parseInt(s, 10));
    if ((m = s.match(/^(?:https?:\/\/)?(?:localhost|127\.0\.0\.1):(\d+)/i))) return addServiceTab(parseInt(m[1], 10));
    if (s.startsWith("/proxy/")) { const p = s.split("/")[2]; return addServiceTab(parseInt(p, 10)); }
    if (/^https?:\/\//i.test(s)) return addExternalTab(s);
    return addExternalTab("http://" + s);
  }

  /* --------------------------------------------------------------- open menu */
  const openMenu = $("open-menu"), omList = $("om-list");
  function toggleOpenMenu(force) {
    const show = force !== undefined ? force : openMenu.hidden;
    openMenu.hidden = !show;
    if (show) {
      const r = $("viewer-add").getBoundingClientRect();
      openMenu.style.top = r.bottom + 6 + "px";
      openMenu.style.left = Math.max(8, r.right - 320) + "px";
      refreshServices();
    }
  }
  $("viewer-add").onclick = (e) => { e.stopPropagation(); toggleOpenMenu(); };
  $("viewer-empty-open").onclick = (e) => { e.stopPropagation(); toggleOpenMenu(true); };
  document.addEventListener("click", (e) => {
    if (!openMenu.hidden && !openMenu.contains(e.target) && e.target !== $("viewer-add")) toggleOpenMenu(false);
  });
  $("om-form").addEventListener("submit", (e) => {
    e.preventDefault();
    openTarget($("om-url").value);
    $("om-url").value = "";
    toggleOpenMenu(false);
  });
  async function refreshServices() {
    try {
      const { services } = await (await fetch("/api/services")).json();
      if (!services.length) { omList.innerHTML = '<div class="om-empty">未发现本地服务</div>'; return; }
      omList.innerHTML = "";
      services.forEach((s) => {
        const row = document.createElement("div");
        row.className = "om-item";
        row.innerHTML = `<span class="om-port">:${s.port}</span><span class="om-svc"></span><span class="om-proc"></span>`;
        row.querySelector(".om-svc").textContent = s.label;
        row.querySelector(".om-proc").textContent = s.proc || "";
        row.onclick = () => { addServiceTab(s.port, ":" + s.port + " " + (s.proc || "")); toggleOpenMenu(false); };
        omList.appendChild(row);
      });
    } catch (_) { omList.innerHTML = '<div class="om-empty">读取失败</div>'; }
  }

  /* --------------------------------------------------------------------- chat */
  const body = $("chat-body"), textEl = $("chat-text"), sendBtn = $("chat-send");
  let chatWS, busy = false, chatReady = false, pendingText = null;

  const looksHtml = (s) =>
    /<(html|body|div|table|section|article|main|header|h[1-6]|ul|ol|li|p|span|canvas|svg|img|pre|code|style|script|button|form|iframe)[\s>/]/i.test(s);

  function htmlTitle(html) {
    const t = html.match(/<title[^>]*>([^<]+)<\/title>/i) || html.match(/<h[1-3][^>]*>([^<]+)<\/h[1-3]>/i);
    return t ? t[1].trim().slice(0, 26) : null;
  }

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
  function addRenderedNote(label) {
    const b = addMsg("bot", "已在左侧面板渲染：" + label + "  ↖");
    b.style.color = "#2563eb";
    b.style.cursor = "default";
  }
  function setBusy(b) { busy = b; sendBtn.disabled = b; }

  // streaming-turn state
  let curBubble = null, curText = "", toolEls = {};
  function rm(el) { if (el) el.closest(".msg").remove(); }

  function startTurn() {
    curText = ""; toolEls = {};
    curBubble = addMsg("bot", "");
    curBubble.classList.add("thinking");
    curBubble.textContent = "思考中";
  }
  function appendToken(t) {
    if (!curBubble) startTurn();
    curBubble.classList.remove("thinking");
    curText += t;
    curBubble.textContent = curText;
    body.scrollTop = body.scrollHeight;
  }
  function addToolLine(u) {
    const id = u.id || Math.random();
    let t = toolEls[id];
    if (!t) {
      const wrap = document.createElement("div");
      wrap.className = "msg msg-bot";
      wrap.innerHTML = '<span class="msg-ava tool-ava">⚙</span><div class="bubble tool-bubble"></div>';
      body.appendChild(wrap);
      t = toolEls[id] = { el: wrap.querySelector(".tool-bubble"), title: "" };
    }
    if (u.title) t.title = u.title;             // tool_call has the title; updates may not
    t.el.textContent = "🔧 " + (t.title || "工具") + (u.status ? " · " + u.status : "");
    t.el.classList.toggle("tool-done", u.status === "completed" || u.status === "failed");
    body.scrollTop = body.scrollHeight;
  }
  function addPermission(m) {
    const bubble = addMsg("bot", "");
    bubble.classList.add("perm");
    bubble.textContent = "⚠️ 需要授权：" + (m.title || "");
    const box = document.createElement("div");
    box.className = "perm-btns";
    (m.options || []).forEach((o) => {
      const b = document.createElement("button");
      b.className = "perm-btn" + (/reject|deny/i.test(o.kind || o.optionId || "") ? " perm-deny" : "");
      b.textContent = o.name || o.optionId;
      b.onclick = () => {
        chatWS.send(JSON.stringify({ type: "permission_response", reqId: m.reqId, optionId: o.optionId }));
        box.querySelectorAll("button").forEach((x) => (x.disabled = true));
        b.classList.add("chosen");
      };
      box.appendChild(b);
    });
    bubble.appendChild(box);
    body.scrollTop = body.scrollHeight;
  }
  function finishTurn() {
    const txt = curText.trim();
    if (txt && looksHtml(txt)) {
      const label = htmlTitle(txt) || "AI 输出";
      addHtmlTab(txt, label);
      rm(curBubble);                 // don't show raw HTML in chat
      addRenderedNote(label);
    } else if (!txt) {
      rm(curBubble);                 // tool-only turn: drop the empty answer bubble
    } else {
      curBubble.classList.remove("thinking");
    }
    curBubble = null;
    setBusy(false);
  }

  function connectChat() {
    chatWS = new WebSocket(wsURL("/ws/chat"));
    chatWS.onmessage = (e) => {
      const m = JSON.parse(e.data);
      switch (m.type) {
        case "start": startTurn(); break;
        case "thought": if (curBubble && !curText) curBubble.textContent = "思考中…"; break;
        case "token": appendToken(m.text || ""); break;
        case "tool": addToolLine(m); break;
        case "permission": addPermission(m); break;
        case "done": finishTurn(); break;
        case "error": rm(curBubble); curBubble = null; addMsg("bot", "⚠️ " + m.error); setBusy(false); break;
        case "need_key": rm(curBubble); curBubble = null; setBusy(false); openKeyModal(); break;
      }
    };
    chatWS.onclose = () => setTimeout(connectChat, 1500);
  }
  connectChat();

  function send() {
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
  textEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  textEl.addEventListener("input", () => { textEl.style.height = "auto"; textEl.style.height = Math.min(textEl.scrollHeight, 140) + "px"; });
  $("chat-quick").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-q]");
    if (!b) return;
    textEl.value = b.dataset.q; textEl.dispatchEvent(new Event("input")); send();
  });

  /* ------------------------------------------------------------- key handling */
  const modal = $("key-modal"), keyInput = $("key-input"), keyErr = $("key-err"), saveBtn = $("key-save");
  function openKeyModal() { modal.hidden = false; keyErr.hidden = true; keyInput.focus(); }
  function closeKeyModal() { modal.hidden = true; }
  $("chat-gear").onclick = openKeyModal;
  $("key-cancel").onclick = closeKeyModal;
  saveBtn.onclick = async () => {
    const api_key = keyInput.value.trim();
    if (!api_key) { keyErr.textContent = "请输入 API Key"; keyErr.hidden = false; return; }
    saveBtn.disabled = true; saveBtn.textContent = "保存中…";
    try {
      const r = await fetch("/api/volcano-key", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key,
          base_url: $("key-baseurl").value.trim() || undefined,
          model: $("key-model").value.trim() || undefined,
        }),
      });
      const data = await r.json();
      if (!data.ok) throw new Error(data.error || "保存失败");
      chatReady = true;
      if (data.model) $("chat-model").textContent = data.model;
      $("chat-status").style.background = "";
      closeKeyModal();
      if (pendingText) { textEl.value = pendingText; pendingText = null; send(); }
    } catch (err) { keyErr.textContent = "⚠️ " + err.message; keyErr.hidden = false; }
    finally { saveBtn.disabled = false; saveBtn.textContent = "保存并开始"; }
  };

  /* ---------------------------------------------------------------- splitters */
  function dragSplit(handle, onMove) {
    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      document.body.classList.add("splitting");
      const move = (ev) => onMove(ev);
      const up = () => {
        document.body.classList.remove("splitting");
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        refit();
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
  }
  const grid = $("grid"), viewer = $("viewer"), colLeft = $("col-left");
  dragSplit($("splitter-v"), (e) => {
    const r = grid.getBoundingClientRect();
    const left = Math.max(280, Math.min(e.clientX - r.left, r.width - 340));
    grid.style.setProperty("--left", left + "px");
  });
  dragSplit($("splitter-h"), (e) => {
    const r = colLeft.getBoundingClientRect();
    const h = Math.max(90, Math.min(e.clientY - r.top, r.height - 140));
    viewer.style.setProperty("--viewer-h", h + "px");
    refit();
  });

  /* ------------------------------------------------------------------ status */
  fetch("/api/status").then((r) => r.json()).then((s) => {
    chatReady = !!s.chat_ready;
    if (s.model) $("chat-model").textContent = s.model;
    $("chat-status").style.background = chatReady ? "" : "#c2c7d2";
  }).catch(() => {});
})();
