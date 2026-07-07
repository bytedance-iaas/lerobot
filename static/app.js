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
  let sessionActive = false;   // single-session lock: only reconnect WS while we hold it

  /* ------------------------------------------------------- terminal (tabbed) */
  // Each tab is its own PTY: a /ws/term connection forks a fresh shell server-side,
  // so tabs are fully independent. The manager owns xterm instances + their sockets.
  const TERM = (() => {
    const stack = $("term-stack"), tabBar = $("term-tabs");
    const TERM_OPTS = {
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
      fontSize: 13.5,
      theme: { background: "#0b0f17", foreground: "#d6deeb", cursor: "#5aa7ff" },
      cursorBlink: true, convertEol: true,
    };
    const tabs = new Map();   // id -> { id, term, fit, ws, el, tabEl }
    let active = null, seq = 0;

    function sendResize(t) {
      if (t.ws && t.ws.readyState === 1)
        t.ws.send(JSON.stringify({ type: "resize", cols: t.term.cols, rows: t.term.rows }));
    }
    function refit() {
      const t = tabs.get(active);
      if (t) try { t.fit.fit(); sendResize(t); } catch (_) {}
    }
    function connect(t) {
      t.ws = new WebSocket(wsURL("/ws/term"));
      t.ws.binaryType = "arraybuffer";
      t.ws.onopen = () => sendResize(t);
      t.ws.onmessage = (e) => t.term.write(typeof e.data === "string" ? e.data : new Uint8Array(e.data));
      t.ws.onclose = () => {
        if (!sessionActive || !tabs.has(t.id)) return;
        t.term.write("\r\n\x1b[33m[console disconnected — reconnecting…]\x1b[0m\r\n");
        setTimeout(() => { if (sessionActive && tabs.has(t.id)) connect(t); }, 1500);
      };
    }
    function activate(id) {
      if (!tabs.has(id)) return;
      active = id;
      for (const [tid, t] of tabs) {
        const on = tid === id;
        t.el.classList.toggle("active", on);
        t.tabEl.classList.toggle("active", on);
      }
      refit();
      const t = tabs.get(id);
      if (t) setTimeout(() => t.term.focus(), 0);
    }
    function open() {
      const id = "term" + ++seq;
      const el = document.createElement("div");
      el.className = "term-pane";
      stack.appendChild(el);
      const term = new Terminal(TERM_OPTS);
      const fit = new FitAddon.FitAddon();
      term.loadAddon(fit);
      term.open(el);
      term.onData((d) => {
        const t = tabs.get(id);
        if (t && t.ws && t.ws.readyState === 1) t.ws.send(JSON.stringify({ type: "input", data: d }));
      });
      const tabEl = document.createElement("button");
      tabEl.className = "term-tab";
      tabEl.dataset.id = id;
      tabEl.innerHTML = `<span class="tt-label"></span><span class="tt-close" title="关闭">✕</span>`;
      tabEl.querySelector(".tt-label").textContent = "终端 " + seq;
      tabBar.appendChild(tabEl);
      const t = { id, term, fit, ws: null, el, tabEl };
      tabs.set(id, t);
      connect(t);
      activate(id);
      return id;
    }
    function close(id) {
      const t = tabs.get(id);
      if (!t || tabs.size <= 1) return;   // always keep at least one terminal
      try { t.ws && t.ws.close(); } catch (_) {}
      try { t.term.dispose(); } catch (_) {}
      t.el.remove(); t.tabEl.remove(); tabs.delete(id);
      if (active === id) activate([...tabs.keys()].pop());
    }
    tabBar.addEventListener("click", (e) => {
      const tab = e.target.closest(".term-tab");
      if (!tab) return;
      if (e.target.closest(".tt-close")) close(tab.dataset.id);
      else activate(tab.dataset.id);
    });
    $("term-add").onclick = () => open();
    $("term-clear").onclick = () => { const t = tabs.get(active); if (t) t.term.clear(); };
    window.addEventListener("resize", refit);

    return {
      refit,
      // (re)connect sockets when our session lock is granted; open one if none yet.
      start() { if (tabs.size === 0) open(); else for (const t of tabs.values()) if (!t.ws || t.ws.readyState > 1) connect(t); },
      stop() { for (const t of tabs.values()) try { t.ws && t.ws.close(); } catch (_) {} },
    };
  })();
  const refit = () => TERM.refit();

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

  // A "doc" tab renders a markdown file (e.g. the Release Notes welcome page) into a
  // scrollable pane — no iframe, just fetch + a tiny markdown -> HTML pass.
  // Links in chat/artifacts open in a NEW TAB — a plain <a> would navigate the whole
  // single-page console away and lose the session. Also harden with rel=noopener.
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A" && node.getAttribute("href")) {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
  // Render markdown (and any inline HTML the model emits) to safe HTML via marked + DOMPurify.
  function renderMD(src) {
    return DOMPurify.sanitize(marked.parse(src || "", { gfm: true, breaks: true }));
  }

  function addDocTab(id, label, url) {
    if (tabs.has(id)) { activate(id); return id; }
    const tabEl = makeTab(id, label);
    const pane = document.createElement("div");
    pane.className = "vpane vpane-doc";
    pane.dataset.pane = id;
    const doc = document.createElement("div");
    doc.className = "md-body";
    doc.innerHTML = '<p class="md-loading">加载中…</p>';
    pane.appendChild(doc);
    viewerBody.appendChild(pane);
    tabs.set(id, { tabEl, paneEl: pane, kind: "doc" });
    // The welcome/release-notes doc also gets a live deployed-version banner (fetched
    // server-side, so it's correct regardless of any stale browser cache).
    const wantVer = id === "welcome";
    Promise.all([
      fetch(url).then((r) => r.text()).catch(() => "# " + label + "\n\n(无法加载)"),
      wantVer
        ? fetch("/api/version", { cache: "no-store" }).then((r) => r.json()).catch(() => ({}))
        : Promise.resolve(null),
    ]).then(([md, ver]) => {
      let banner = "";
      if (ver) {
        const short = (s) => (s || "unknown").slice(0, 12);
        banner =
          '<div class="md-ver">当前部署版本 · lerobot <code>' + short(ver.lerobot) +
          '</code> · console <code>' + short(ver.console) + "</code></div>";
      }
      doc.innerHTML = banner + renderMD(md);
    });
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
  function setBusy(b) {
    busy = b;
    sendBtn.disabled = false;                 // stays clickable so it can stop
    sendBtn.classList.toggle("is-stop", b);
    sendBtn.textContent = b ? "■" : "➤";
    sendBtn.title = b ? "停止" : "发送";
  }
  function stopTurn() {
    if (chatWS && chatWS.readyState === 1) chatWS.send(JSON.stringify({ type: "stop" }));
  }

  // streaming-turn state
  let curBubble = null, curText = "", toolEls = {}, curThought = null;
  function rm(el) { if (el) el.closest(".msg").remove(); }

  function startTurn(booting) {
    curText = ""; toolEls = {}; curThought = null;
    curBubble = addMsg("bot", "");
    curBubble.classList.add("thinking");
    curBubble.textContent = booting ? "正在启动 Agent" : "思考中";
  }
  // The answer bubble (blinking while pending) must stay at the BOTTOM — reasoning
  // and tool cards stack above it in arrival order. Re-append it after each new card.
  function keepAnswerLast() { if (curBubble) body.appendChild(curBubble.closest(".msg")); }
  // Stream the agent's reasoning into its own collapsible 💭 bubble, placed
  // ABOVE the answer bubble. Expanded while streaming (so it's visible live),
  // collapsed when the turn ends. One thought bubble accumulates the whole turn.
  function addThought(text) {
    if (!text) return;
    if (!curBubble) startTurn();
    if (!curThought) {
      const wrap = document.createElement("div");
      wrap.className = "msg msg-bot";
      wrap.innerHTML = '<span class="msg-ava tool-ava">💭</span><div class="bubble tool-bubble think-bubble"></div>';
      body.appendChild(wrap);                                // reasoning stacks in arrival order
      const bubble = wrap.querySelector(".tool-bubble");
      curThought = { bubble, refs: buildToolBubble(bubble), text: "" };
      curThought.refs.detail.hidden = false;                 // stream expanded
      curThought.refs.caret.textContent = "▾";
      curBubble.textContent = "";                            // drop the redundant "思考中" (blinking cursor stays)
      keepAnswerLast();                                      // keep the blinking answer bubble at the bottom
    }
    curThought.text += text;
    curThought.refs.detail.textContent = curThought.text;
    curThought.refs.label.textContent = "💭 思考中…";
    curThought.bubble.classList.remove("no-detail");
    body.scrollTop = body.scrollHeight;
  }
  function finalizeThought() {
    if (!curThought) return;
    curThought.refs.label.textContent = "💭 思考";
    curThought.refs.detail.hidden = true;                    // collapse when done (click to re-expand)
    curThought.refs.caret.textContent = "▸";
    curThought = null;
  }
  // Dock the reasoning card directly ABOVE the answer bubble and collapse it, so "思考"
  // reads as part of the answer (grouped) and is closed by default — click to reopen.
  function dockThoughtToAnswer() {
    if (!curThought || !curBubble) return;
    const answerMsg = curBubble.closest(".msg");
    const thoughtMsg = curThought.bubble.closest(".msg");
    thoughtMsg.classList.add("think-docked");                // tighten the gap to the answer
    body.insertBefore(thoughtMsg, answerMsg);
    finalizeThought();                                       // collapse + relabel 💭 思考
  }
  // Render a bubble's raw text: full HTML artifacts via sanitize, otherwise markdown — so
  // **bold** / `code` / lists render *live* while streaming, not as raw text that only snaps
  // to rendered at the end of the turn.
  function renderBubble(bubble, raw) {
    bubble.classList.add("md-chat");
    bubble.innerHTML = renderMD(raw);   // marked renders markdown + passes inline HTML; DOMPurify sanitizes
  }
  // Coalesce streamed tokens to one render per frame (~60fps) so we don't re-parse the whole
  // bubble on every token.
  let _renderPending = false;
  function scheduleRender() {
    if (_renderPending) return;
    _renderPending = true;
    requestAnimationFrame(() => {
      _renderPending = false;
      if (!curBubble) return;
      renderBubble(curBubble, curText);
      body.scrollTop = body.scrollHeight;
    });
  }
  function appendToken(t) {
    if (!curBubble) startTurn();
    if (!curText && curThought) dockThoughtToAnswer();   // first answer token → group 思考 with the answer
    curBubble.classList.remove("thinking");
    curText += t;
    scheduleRender();
  }
  // A tool card: clickable header (🔧 title · status) + a collapsible detail
  // pane (the full command + output). Header toggles; caret shows the state.
  function buildToolBubble(bubble) {
    bubble.innerHTML =
      '<button class="tool-head" type="button">' +
      '<span class="tool-caret">▸</span><span class="tool-label"></span></button>' +
      '<pre class="tool-detail" hidden></pre>';
    const head = bubble.querySelector(".tool-head");
    const detail = bubble.querySelector(".tool-detail");
    const caret = bubble.querySelector(".tool-caret");
    head.onclick = () => {
      if (bubble.classList.contains("no-detail")) return;   // nothing to show
      const open = detail.hidden;
      detail.hidden = !open;
      caret.textContent = open ? "▾" : "▸";
    };
    return { head, detail, caret, label: bubble.querySelector(".tool-label") };
  }
  function setToolBubble(bubble, refs, title, status, detail) {
    refs.label.textContent = "🔧 " + (title || "工具") + (status ? " · " + status : "");
    bubble.classList.toggle("tool-done", status === "completed" || status === "failed");
    if (detail != null && detail !== "") {
      refs.detail.textContent = detail;                     // latest non-empty wins
      bubble.classList.remove("no-detail");
    } else if (!refs.detail.textContent) {
      bubble.classList.add("no-detail");                    // no detail yet → not expandable
    }
  }
  function addToolLine(u) {
    const id = u.id || Math.random();
    let t = toolEls[id];
    if (!t) {
      const wrap = document.createElement("div");
      wrap.className = "msg msg-bot";
      wrap.innerHTML = '<span class="msg-ava tool-ava">⚙</span><div class="bubble tool-bubble"></div>';
      body.appendChild(wrap);
      const bubble = wrap.querySelector(".tool-bubble");
      t = toolEls[id] = { el: bubble, refs: buildToolBubble(bubble), title: "" };
      keepAnswerLast();                         // keep the blinking answer bubble at the bottom
    }
    if (u.title) t.title = u.title;             // tool_call has the title; updates may not
    setToolBubble(t.el, t.refs, t.title, u.status, u.detail);
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
    finalizeThought();
    const txt = curText.trim();
    if (!txt) {
      // No final text. If the agent ran tools, show a subtle "done" marker so the
      // turn doesn't look frozen; otherwise drop the empty bubble.
      if (Object.keys(toolEls).length) {
        curBubble.classList.remove("thinking");
        curBubble.textContent = "（已完成）";
        curBubble.style.color = "#8a90a0";
      } else {
        rm(curBubble);
      }
    } else {
      curBubble.classList.remove("thinking");
      renderBubble(curBubble, curText.trim());   // same render path as live streaming
    }
    curBubble = null;
    setBusy(false);
  }

  /* ---------------------------------------------------------- chat sessions */
  const sessMenu = $("sess-menu"), sessList = $("sess-list"), sessTitle = $("chat-session-title"), sessFoot = $("sess-foot");
  let curSession = null;

  function clearChat() {
    body.innerHTML = "";
    curBubble = null; curText = ""; toolEls = {}; curThought = null;
  }
  function wsSend(o) { if (chatWS && chatWS.readyState === 1) chatWS.send(JSON.stringify(o)); }

  function setTitle(t) { sessTitle.textContent = (t && t.trim()) || "新会话"; }

  function newSession() {
    toggleSessMenu(false);
    if (busy) stopTurn();                      // abandon any in-flight turn, then switch
    wsSend({ type: "session_new" });           // server replies session_switched(fresh)
  }
  function loadSession(id, title) {
    if (id === curSession) { toggleSessMenu(false); return; }
    toggleSessMenu(false);
    if (busy) stopTurn();                      // abandon any in-flight turn, then switch
    setTitle(title);                           // optimistic title only; curSession is set when the
    wsSend({ type: "session_load", id });       // server actually starts streaming (history_start).
  }                                             // Setting it early left a stale guard that ate the 1st click.
  function deleteSession(id) {
    if (busy) return;
    wsSend({ type: "session_delete", id });
  }
  function refreshSessions() { wsSend({ type: "session_list" }); }

  function onSwitched(id, title, fresh) {
    curSession = id;
    setTitle(title);
    clearChat();
    if (fresh) addMsg("bot", "新会话已就绪 👋 直接发消息开始吧。");
    setBusy(false);
  }
  function onDeleted(id) {
    if (id === curSession) {
      // Deleted the open session: clear it but DON'T auto-create a new one —
      // wait for the user to click 新建会话.
      curSession = null;
      clearChat();
      setTitle("");
      setBusy(false);
      addMsg("bot", "会话已删除。点上方「＋ 新建会话」开始新的对话。");
    }
    refreshSessions();
  }

  function renderSessions(items, current) {
    curSession = current || curSession;
    const cur = items.find((s) => s.id === current);
    if (cur) setTitle(cur.title);
    sessList.innerHTML = "";
    sessFoot.textContent = "";
    if (!items.length) { sessList.innerHTML = '<div class="sess-empty">还没有会话</div>'; return; }
    const now = Date.now(), DAY = 864e5;
    const bucket = (iso) => {
      const age = now - (Date.parse(iso) || 0);
      return age <= DAY ? "1天内" : age <= 7 * DAY ? "一周内" : age <= 30 * DAY ? "一个月内" : "更早";
    };
    let lastGroup = null;
    items.forEach((s) => {
      const g = bucket(s.updatedAt);
      if (g !== lastGroup) {
        lastGroup = g;
        const h = document.createElement("div");
        h.className = "sess-group"; h.textContent = g;
        sessList.appendChild(h);
      }
      const row = document.createElement("div");
      row.className = "sess-item" + (s.id === current ? " active" : "");
      row.innerHTML = '<div class="si-main"><div class="si-title"></div></div><button class="si-del" title="删除会话">🗑</button>';
      row.querySelector(".si-title").textContent = s.title || "新会话";
      row.querySelector(".si-main").onclick = () => loadSession(s.id, s.title);
      row.querySelector(".si-del").onclick = (e) => {
        e.stopPropagation();
        if (confirm("删除会话「" + (s.title || "新会话") + "」？此操作不可撤销。")) deleteSession(s.id);
      };
      sessList.appendChild(row);
    });
    sessFoot.textContent = "已全部加载完成";
  }

  // ----- history replay (session_load) -----
  let histAcc = null;  // { role, text } accumulator; consecutive same-role chunks merge
  let histTools = {};  // tool cards by id — dedup so one tool isn't rendered per update event
  function histStart(id) { clearChat(); setBusy(true); histAcc = null; histTools = {}; if (id) curSession = id; }
  // The first user message carries a "[System] …" steering block. It may be appended (new
  // sessions) or prepended (legacy) — strip it either way so it never shows in the bubble.
  function stripSystemPrefix(s) {
    return s
      .replace(/^\s*\[System\][\s\S]*?\n\n/, "")   // legacy: prepended
      .replace(/\n\n\[System\][\s\S]*$/, "")        // new: appended
      .trim();
  }
  function histFlush() {
    if (!histAcc) return;
    const { role, text } = histAcc; histAcc = null;
    if (role === "user") { const u = stripSystemPrefix(text).trim(); if (u) addMsg("user", u); return; }
    const t = text.trim();
    if (!t) return;
    if (role === "thought") {                          // replayed reasoning → collapsed 💭 card
      const wrap = document.createElement("div");
      wrap.className = "msg msg-bot";
      wrap.innerHTML = '<span class="msg-ava tool-ava">💭</span><div class="bubble tool-bubble think-bubble"></div>';
      const bubble = wrap.querySelector(".tool-bubble");
      const refs = buildToolBubble(bubble);
      refs.label.textContent = "💭 思考";
      refs.detail.textContent = t;                     // collapsed by default; click to expand
      body.appendChild(wrap);
      return;
    }
    renderBubble(addMsg("bot", ""), t);
  }
  function histChunk(m) {
    if (m.role === "tool") {
      histFlush();
      const id = m.id || Math.random();
      let t = histTools[id];
      if (!t) {                                    // one card per tool id, not per update event
        const wrap = document.createElement("div");
        wrap.className = "msg msg-bot";
        wrap.innerHTML = '<span class="msg-ava tool-ava">⚙</span><div class="bubble tool-bubble"></div>';
        const bubble = wrap.querySelector(".tool-bubble");
        t = histTools[id] = { el: bubble, refs: buildToolBubble(bubble), title: "" };
        body.appendChild(wrap);
      }
      if (m.title) t.title = m.title;
      setToolBubble(t.el, t.refs, t.title, m.status, m.detail);
      return;
    }
    if (!histAcc || histAcc.role !== m.role) { histFlush(); histAcc = { role: m.role, text: "" }; }
    histAcc.text += m.text || "";
  }
  function histDone() {
    histFlush();
    if (!body.children.length) addMsg("bot", "（这个会话还没有消息）");
    setBusy(false);
    body.scrollTop = body.scrollHeight;
  }

  function toggleSessMenu(force) {
    const show = force !== undefined ? force : sessMenu.hidden;
    sessMenu.hidden = !show;
    if (show) {
      // drop the panel just under the header (left-aligned to the session button)
      const head = $("chat").querySelector(".chat-head");
      sessMenu.style.top = head.offsetHeight + 4 + "px";
      refreshSessions();
    }
  }
  $("sess-toggle").onclick = (e) => { e.stopPropagation(); toggleSessMenu(); };
  $("sess-new").onclick = () => newSession();
  document.addEventListener("click", (e) => {
    if (!sessMenu.hidden && !sessMenu.contains(e.target) && !$("sess-toggle").contains(e.target)) toggleSessMenu(false);
  });

  function connectChat() {
    chatWS = new WebSocket(wsURL("/ws/chat"));
    chatWS.onmessage = (e) => {
      const m = JSON.parse(e.data);
      switch (m.type) {
        case "start": startTurn(m.booting); break;
        case "thought": addThought(m.text || ""); break;
        case "token": appendToken(m.text || ""); break;
        case "tool": addToolLine(m); break;
        case "permission": addPermission(m); break;
        case "done": finishTurn(); setTimeout(() => { if (!busy) refreshSessions(); }, 800); break;
        case "error": rm(curBubble); curBubble = null; addMsg("bot", "⚠️ " + m.error); setBusy(false); break;
        case "need_key": rm(curBubble); curBubble = null; setBusy(false); openKeyModal(); break;
        case "sessions": renderSessions(m.items || [], m.current); break;
        case "session_switched": onSwitched(m.id, m.title, m.fresh); break;
        case "session_deleted": onDeleted(m.id); break;
        case "history_start": histStart(m.id); break;
        case "hist": histChunk(m); break;
        case "history_done": histDone(); break;
      }
    };
    chatWS.onclose = () => { if (sessionActive) setTimeout(() => sessionActive && connectChat(), 1500); };
  }

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
  sendBtn.onclick = () => { if (busy) stopTurn(); else send(); };
  textEl.addEventListener("keydown", (e) => {
    // Ignore Enter while an IME is composing — that Enter confirms a candidate (CJK/pinyin),
    // it must not send, or the box keeps residue. isComposing is the standard flag; keyCode
    // 229 is the legacy fallback for older browsers.
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      send();
    }
  });
  textEl.addEventListener("input", () => { textEl.style.height = "auto"; textEl.style.height = Math.min(textEl.scrollHeight, 140) + "px"; });
  // Release-notes button: always (re)open the doc so clicking gives a visible reaction even
  // when the tab is already the active one.
  $("q-release").onclick = () => {
    if (tabs.has("welcome")) closeTab("welcome");
    addDocTab("welcome", "更新说明", "/static/release_note.md");
  };
  // In-app markdown links like [text](doc:webrtc) open the local lerobot doc (served from
  // /lerobot in the image) as a doc tab, instead of navigating to an external URL.
  viewerBody.addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="doc:"]');
    if (!a) return;
    e.preventDefault();
    const name = a.getAttribute("href").slice(4);
    addDocTab("doc-" + name, (a.textContent || name).trim(), "/api/lerobot-doc/" + name);
  });
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
    saveBtn.disabled = true; saveBtn.textContent = "正在启动 Agent…";
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
    // Resize the chat (right) column; the left stays 1fr so the app always
    // fills the viewport at any window width.
    const r = grid.getBoundingClientRect();
    const right = Math.max(300, Math.min(r.right - e.clientX, r.width - 380));
    grid.style.setProperty("--right", right + "px");
  });
  dragSplit($("splitter-h"), (e) => {
    const r = colLeft.getBoundingClientRect();
    const h = Math.max(90, Math.min(e.clientY - r.top, r.height - 140));
    viewer.style.setProperty("--viewer-h", h + "px");
    refit();
  });

  /* ------------------------------------------------------------------ status */
  function fetchStatus() {
    fetch("/api/status").then((r) => r.json()).then((s) => {
      chatReady = !!s.chat_ready;
      if (s.model) $("chat-model").value = s.model;
      $("chat-status").style.background = chatReady ? "" : "#c2c7d2";
      // Small warning when the console is served over plain HTTP (unencrypted).
      // Prefer the server's answer; fall back to the page protocol on older servers.
      const insecure = s.secure === false || (s.secure === undefined && location.protocol !== "https:");
      $("http-warn").hidden = !insecure;
    }).catch(() => {});
  }

  /* -------------------------------------------------------------- start up */
  // No presence lock — multiple windows/users may be open at once.
  // Model dropdown: populate from /api/models, switch model on change.
  function loadModels() {
    fetch("/api/models").then((r) => r.json()).then((d) => {
      const sel = $("chat-model");
      sel.innerHTML = "";
      (d.models || []).forEach((m) => {
        const o = document.createElement("option");
        o.value = m;
        o.textContent = m;
        sel.appendChild(o);
      });
      if (d.current) sel.value = d.current;
    }).catch(() => {});
  }
  $("chat-model").addEventListener("change", (e) => {
    fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: e.target.value }),
    }).then((r) => r.json()).then((d) => { if (d.model) $("chat-model").value = d.model; }).catch(() => {});
  });

  function startApp() {
    sessionActive = true;
    TERM.start();
    if (!chatWS || chatWS.readyState > 1) connectChat();
    fetchStatus();
    loadModels();
  }
  startApp();
})();
