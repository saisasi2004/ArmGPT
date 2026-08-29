/* ArmGPT frontend. Vanilla - no build step, no CDN. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const el = {
    app: document.querySelector('.app'),
    messages: $('messages'),
    welcome: $('welcome'),
    composer: $('composer'),
    input: $('input'),
    send: $('send'),
    hintText: $('hintText'),
    chatTitle: $('chatTitle'),
    sessionList: $('sessionList'),
    newChat: $('newChat'),
    videoFeed: $('videoFeed'),
    videoFrame: document.querySelector('.video-frame'),
    videoFallback: $('videoFallback'),
    previewMode: $('previewMode'),
    previewNote: $('previewNote'),
    paramsPanel: $('paramsPanel'),
  };

  let sessionId = null;
  let busy = false;

  /* ───────────────────────────── helpers ───────────────────────────── */

  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

  // Escape first, then promote `backticks` to <code>. Order matters: doing it
  // the other way round would let message text inject markup.
  const renderText = (s) =>
    escapeHtml(s).replace(/`([^`\n]+)`/g, '<code>$1</code>');

  async function api(url, options = {}) {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).error || detail; } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  const scrollDown = () => {
    el.messages.scrollTop = el.messages.scrollHeight;
  };

  /* ──────────────────────────── messages ───────────────────────────── */

  function hideWelcome() {
    if (el.welcome) { el.welcome.remove(); el.welcome = null; }
  }

  function addUserMessage(text) {
    hideWelcome();
    const node = document.createElement('div');
    node.className = 'msg user';
    node.innerHTML = `
      <div class="msg-avatar">You</div>
      <div class="msg-body">
        <div class="msg-role">You</div>
        <div class="msg-text"></div>
      </div>`;
    node.querySelector('.msg-text').textContent = text;
    el.messages.appendChild(node);
    scrollDown();
  }

  function addThinking() {
    const node = document.createElement('div');
    node.className = 'msg assistant';
    node.id = 'thinking';
    node.innerHTML = `
      <div class="msg-avatar"><box-icon name='bot' color='currentColor' size='sm'></box-icon></div>
      <div class="msg-body">
        <div class="msg-role">ArmGPT</div>
        <div class="thinking">
          <span>parsing command</span>
          <span class="thinking-dots"><i></i><i></i><i></i></span>
        </div>
      </div>`;
    el.messages.appendChild(node);
    scrollDown();
    return node;
  }

  const PILL_TEXT = {
    ok: 'executed',
    ambiguous: 'needs clarification',
    blocked: 'blocked - hand detected',
    not_found: 'nothing found',
    error: 'error',
  };

  function addAssistantMessage(result) {
    hideWelcome();
    const node = document.createElement('div');
    node.className = 'msg assistant';

    const body = document.createElement('div');
    body.className = 'msg-body';

    const role = document.createElement('div');
    role.className = 'msg-role';
    role.textContent = result.elapsed_ms
      ? `ArmGPT · ${(result.elapsed_ms / 1000).toFixed(1)}s`
      : 'ArmGPT';
    body.appendChild(role);

    // Status pill - omitted for plain chat replies, which have no execution.
    const isChat = result.intent && result.intent.action === 'chat';
    if (result.status && !(result.status === 'ok' && isChat)) {
      const pill = document.createElement('span');
      pill.className = `pill ${result.status}`;
      pill.textContent = PILL_TEXT[result.status] || result.status;
      body.appendChild(pill);
    }

    const text = document.createElement('div');
    text.className = 'msg-text';
    text.innerHTML = renderText(result.reply || '');
    body.appendChild(text);

    if (result.snapshot) {
      const wrap = document.createElement('div');
      wrap.className = 'snapshot';
      const img = document.createElement('img');
      img.src = result.snapshot;
      img.alt = 'Annotated camera frame';
      img.addEventListener('click', () => openLightbox(img.src));
      wrap.appendChild(img);
      body.appendChild(wrap);
    }

    if (result.intent) {
      const trace = document.createElement('details');
      trace.className = 'trace';
      const summary = document.createElement('summary');
      summary.innerHTML = '<box-icon name=\'cog\' color=\'currentColor\' size=\'xs\'></box-icon> execution trace';
      trace.appendChild(summary);
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify({
        intent: result.intent,
        detections: result.detections,
        robot: result.robot,
      }, null, 2);
      trace.appendChild(pre);
      body.appendChild(trace);
    }

    node.innerHTML = '<div class="msg-avatar"><box-icon name=\'bot\' color=\'currentColor\' size=\'sm\'></box-icon></div>';
    node.appendChild(body);
    el.messages.appendChild(node);
    scrollDown();
  }

  /* ───────────────────────────── sending ───────────────────────────── */

  async function send(text) {
    if (busy || !text.trim()) return;
    busy = true;
    el.send.disabled = true;

    addUserMessage(text);
    el.input.value = '';
    autosize();
    const thinking = addThinking();

    try {
      const result = await api('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });
      thinking.remove();
      if (result.session_id && result.session_id !== sessionId) {
        sessionId = result.session_id;
      }
      addAssistantMessage(result);
      loadSessions();
    } catch (err) {
      thinking.remove();
      addAssistantMessage({
        status: 'error',
        reply: `Request failed: ${err.message}`,
      });
    } finally {
      busy = false;
      el.send.disabled = false;
      el.input.focus();
    }
  }

  function autosize() {
    el.input.style.height = 'auto';
    el.input.style.height = Math.min(el.input.scrollHeight, 180) + 'px';
  }

  el.composer.addEventListener('submit', (e) => {
    e.preventDefault();
    send(el.input.value);
  });

  el.input.addEventListener('input', autosize);
  el.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(el.input.value);
    }
  });

  document.addEventListener('click', (e) => {
    const suggestion = e.target.closest('.suggestion');
    if (suggestion) send(suggestion.dataset.q);
  });

  /* ──────────────────────────── sessions ───────────────────────────── */

  async function loadSessions() {
    try {
      const { sessions, persistent } = await api('/api/sessions');
      el.sessionList.innerHTML = '';

      if (!persistent) {
        el.sessionList.innerHTML =
          '<div class="empty-hint">MongoDB isn\'t reachable, so history isn\'t ' +
          'being saved. Chat still works - this session just won\'t persist.</div>';
        return;
      }
      if (!sessions.length) {
        el.sessionList.innerHTML =
          '<div class="empty-hint">No conversations yet.</div>';
        return;
      }

      for (const s of sessions) {
        const row = document.createElement('div');
        row.className = 'session' + (s.id === sessionId ? ' active' : '');

        const title = document.createElement('div');
        title.className = 'session-title';
        title.textContent = s.title;
        title.addEventListener('click', () => openSession(s.id, s.title));

        const del = document.createElement('button');
        del.className = 'session-del';
        del.textContent = '×';
        del.title = 'Delete';
        del.addEventListener('click', async (e) => {
          e.stopPropagation();
          await api(`/api/sessions/${s.id}`, { method: 'DELETE' });
          if (s.id === sessionId) newChat();
          loadSessions();
        });

        row.append(title, del);
        el.sessionList.appendChild(row);
        if (s.id === sessionId) el.chatTitle.textContent = s.title;
      }
    } catch (_) {
      /* sidebar is non-critical - never break chat over it */
    }
  }

  async function openSession(id, title) {
    sessionId = id;
    el.chatTitle.textContent = title;
    el.messages.innerHTML = '';
    el.welcome = null;
    try {
      const { messages } = await api(`/api/sessions/${id}/messages`);
      for (const m of messages) {
        if (m.role === 'user') addUserMessage(m.content);
        else addAssistantMessage({
          reply: m.content,
          status: m.meta?.status,
          intent: m.meta?.intent,
          detections: m.meta?.detections,
          robot: m.meta?.robot,
          // Snapshots aren't persisted - a base64 JPEG per turn would bloat
          // the collection fast, and the frame is stale on replay anyway.
        });
      }
    } catch (err) {
      console.error(err);
    }
    loadSessions();
    scrollDown();
  }

  function newChat() {
    sessionId = null;
    el.chatTitle.textContent = 'New chat';
    el.messages.innerHTML = `
      <div class="welcome" id="welcome">
        <div class="welcome-mark"><box-icon name='bot' color='currentColor' size='60px'></box-icon></div>
        <h1>Tell the arm what to do</h1>
        <p>I'll parse your command with a local LLM, find the objects with the
           overhead camera, and send the pixel coordinates to the controller.</p>
        <div class="suggestions">
          <button class="suggestion" data-q="place the red object on the blue plate">
            <strong>place the red object on the blue plate</strong>
            <span>pick &amp; place by color</span>
          </button>
          <button class="suggestion" data-q="where is the cup?">
            <strong>where is the cup?</strong>
            <span>locate a COCO object</span>
          </button>
          <button class="suggestion" data-q="how many circles do you see?">
            <strong>how many circles do you see?</strong>
            <span>count by shape</span>
          </button>
          <button class="suggestion" data-q="pick up marker 3 and put it on the green block">
            <strong>pick up marker 3 and put it on the green block</strong>
            <span>ArUco → color</span>
          </button>
        </div>
      </div>`;
    el.welcome = $('welcome');
    loadSessions();
    el.input.focus();
  }

  el.newChat.addEventListener('click', newChat);

  /* ───────────────────────────── vision ────────────────────────────── */

  async function loadDetectors() {
    try {
      const { detectors } = await api('/api/detectors');
      for (const d of detectors) {
        const opt = document.createElement('option');
        opt.value = d.key;
        opt.textContent = d.name;
        opt.title = d.hint;
        el.previewMode.appendChild(opt);
      }
    } catch (err) {
      console.error(err);
    }
  }

  const camSource = $('cameraSource');
  const camNote = $('cameraNote');

  // `force` is the rescan button. A scan has to take each device to test it,
  // including the one on screen, so the feed stutters for a few seconds - fine
  // when asked for, not something to do on page load.
  async function loadCameras(force = false) {
    try {
      const { active, devices } = await api(
        '/api/camera/devices' + (force ? '?refresh=1' : ''));
      camSource.innerHTML = '';
      if (!devices.length) {
        camSource.innerHTML = '<option value="">No cameras found</option>';
        return;
      }
      for (const d of devices) {
        const opt = document.createElement('option');
        opt.value = d.index;
        opt.textContent = `Camera ${d.index}`
          + (d.dark ? ' (no image - depth/IR?)' : '')
          + (d.active ? ' - active' : '');
        if (d.index === active) opt.selected = true;
        camSource.appendChild(opt);
      }
    } catch (err) {
      camSource.innerHTML = '<option value="">Error listing cameras</option>';
      console.error(err);
    }
  }

  camSource.addEventListener('change', async () => {
    const index = camSource.value;
    if (index === '') return;
    camNote.classList.remove('error');
    camNote.textContent = `Switching to camera ${index}…`;
    try {
      await api('/api/camera/switch', {
        method: 'POST', body: JSON.stringify({ index: Number(index) }),
      });
      // Nudge the <img> to reconnect so the new feed shows immediately rather
      // than waiting for the browser to notice the multipart stream changed.
      setTimeout(() => {
        el.videoFeed.src = '/video_feed?t=' + Date.now();
        camNote.textContent = `Now showing camera ${index}.`;
        loadCameras();
        pollStatus();
      }, 1200);
    } catch (err) {
      camNote.textContent = err.message;
      camNote.classList.add('error');
    }
  });

  $('refreshCams').addEventListener('click', async () => {
    camNote.classList.remove('error');
    camNote.textContent = 'Scanning for cameras — the feed pauses briefly…';
    await loadCameras(true);
    camNote.textContent = 'Scan complete.';
    el.videoFeed.src = '/video_feed?t=' + Date.now();
  });

  $('retryCam').addEventListener('click', async () => {
    camNote.classList.remove('error');
    camNote.textContent = 'Reconnecting…';
    try {
      await api('/api/camera/retry', { method: 'POST' });
      setTimeout(() => {
        el.videoFeed.src = '/video_feed?t=' + Date.now();
        camNote.textContent = 'Reconnect requested.';
        pollStatus();
      }, 1500);
    } catch (err) {
      camNote.textContent = err.message;
      camNote.classList.add('error');
    }
  });

  el.previewMode.addEventListener('change', async () => {
    const mode = el.previewMode.value;
    el.previewNote.classList.remove('error');
    el.previewNote.textContent = mode === 'none'
      ? 'The chat picks its own detector per command. This only changes what you see here.'
      : 'Loading…';
    try {
      const res = await api('/api/preview', {
        method: 'POST',
        body: JSON.stringify({ mode }),
      });
      buildParams(mode, res.params || [], res.values || {});
      if (mode !== 'none') {
        el.previewNote.textContent =
          'Overlay only. Chat commands still choose their own detector.';
      }
    } catch (err) {
      el.previewNote.textContent = err.message;
      el.previewNote.classList.add('error');
      el.previewMode.value = 'none';
      buildParams('none', [], {});
    }
  });

  function buildParams(mode, specs, values) {
    el.paramsPanel.innerHTML = '';
    if (!specs.length) {
      el.paramsPanel.innerHTML =
        '<div class="params-empty">Select an overlay to tune its parameters live.</div>';
      return;
    }

    const label = document.createElement('label');
    label.className = 'field-label';
    label.textContent = 'Parameters';
    el.paramsPanel.appendChild(label);

    const push = (key, value) => api('/api/preview/param', {
      method: 'POST',
      body: JSON.stringify({ mode, key, value }),
    }).catch(console.error);

    for (const spec of specs) {
      const wrap = document.createElement('div');
      wrap.className = 'param';
      const current = values[spec.key] ?? spec.default;

      if (spec.type === 'check') {
        const lbl = document.createElement('label');
        lbl.className = 'param-check';
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = Boolean(current);
        box.addEventListener('change', () => push(spec.key, box.checked));
        const span = document.createElement('span');
        span.textContent = spec.label;
        lbl.append(box, span);
        wrap.appendChild(lbl);

      } else if (spec.type === 'combo') {
        const head = document.createElement('div');
        head.className = 'param-head';
        head.innerHTML = `<span class="param-name"></span>`;
        head.querySelector('.param-name').textContent = spec.label;
        wrap.appendChild(head);

        const sel = document.createElement('select');
        for (const o of spec.options) {
          const opt = document.createElement('option');
          opt.value = o; opt.textContent = o;
          if (o === current) opt.selected = true;
          sel.appendChild(opt);
        }
        sel.addEventListener('change', () => push(spec.key, sel.value));
        wrap.appendChild(sel);

      } else { // slider
        const head = document.createElement('div');
        head.className = 'param-head';
        const name = document.createElement('span');
        name.className = 'param-name';
        name.textContent = spec.label;
        const val = document.createElement('span');
        val.className = 'param-val';
        val.textContent = current;
        head.append(name, val);
        wrap.appendChild(head);

        const range = document.createElement('input');
        range.type = 'range';
        range.min = spec.min; range.max = spec.max; range.value = current;
        range.addEventListener('input', () => { val.textContent = range.value; });
        range.addEventListener('change', () => push(spec.key, Number(range.value)));
        wrap.appendChild(range);
      }
      el.paramsPanel.appendChild(wrap);
    }
  }

  /* ────────────────────────── robot tcp tab ────────────────────────── */

  const rb = {
    mode: $('robotMode'),
    addrLabel: $('addrLabel'),
    host: $('robotHost'),
    port: $('robotPort'),
    timeout: $('robotTimeout'),
    timeoutWrap: $('timeoutWrap'),
    dryRun: $('dryRun'),
    dryRunTitle: $('dryRunTitle'),
    dryRunSub: $('dryRunSub'),
    liveWarn: $('liveWarn'),
    banner: $('connBanner'),
    connText: $('connText'),
    connNote: $('connNote'),
    testBtn: $('testConn'),
    saveBtn: $('saveConn'),
    manualLine: $('manualLine'),
    manualSend: $('manualSend'),
    manualNote: $('manualNote'),
    clientList: $('clientList'),
    log: $('trafficLog'),
    clearLog: $('clearLog'),
    tabDot: $('tabDot'),
  };

  const isServer = () => rb.mode.value === 'server';

  function paintDryRun() {
    const dry = rb.dryRun.checked;
    const server = isServer();
    rb.dryRunTitle.textContent = dry ? 'Dry run' : 'Live - sending coordinates';
    rb.dryRunSub.textContent = dry
      ? 'Commands are formatted and logged, never sent.'
      : (server ? 'Commands are broadcast to every connected client.'
                : 'Commands are written to the controller socket.');
    rb.liveWarn.hidden = dry;
  }

  // Swap every mode-dependent label/control between server and client.
  function applyModeUI() {
    const server = isServer();
    rb.addrLabel.textContent = server ? 'Listen address' : 'Controller address';
    rb.timeoutWrap.hidden = server;           // timeout is client-only
    rb.testBtn.textContent = server ? 'Check server' : 'Test connection';
    rb.clientList.hidden = !server;
    rb.connNote.classList.remove('error');
    rb.connNote.textContent = server
      ? 'ArmGPT is listening. Point Hercules (TCP Client) at this address and '
        + 'hit Connect, then send a command below.'
      : 'Test opens a socket and closes it - it sends no data, so it can\'t '
        + 'move the arm.';
    paintDryRun();
  }

  function setBanner(state, text) {
    rb.banner.dataset.state = state;
    rb.connText.textContent = text;
    rb.tabDot.dataset.state = state;
  }

  function renderLog(history) {
    if (!history || !history.length) {
      rb.log.innerHTML = '<div class="params-empty">Nothing sent yet.</div>';
      return;
    }
    rb.log.innerHTML = '';
    // newest first - the thing you just did is the thing you want to see
    for (const h of [...history].reverse()) {
      const cls = h.error ? 'failed' : (h.dry_run ? 'dry' : 'sent');
      const tag = h.error ? 'fail' : (h.dry_run ? 'dry' : 'sent');
      const row = document.createElement('div');
      row.className = `log-row ${cls}`;

      const ts = document.createElement('span');
      ts.className = 'log-ts'; ts.textContent = h.ts || '';
      const line = document.createElement('span');
      line.className = 'log-line'; line.textContent = h.line;
      const tagEl = document.createElement('span');
      tagEl.className = 'log-tag';
      let tagText = tag;
      if (!h.error && h.ms != null) {
        // server sends report how many clients received it; client sends the RTT
        tagText = (h.mode === 'server' && h.clients != null)
          ? `${tag} ${h.clients}cl ${h.ms}ms`
          : `${tag} ${h.ms}ms`;
      }
      tagEl.textContent = tagText;

      row.append(ts, line, tagEl);
      if (h.error) {
        const err = document.createElement('span');
        err.className = 'log-err'; err.textContent = h.error;
        row.appendChild(err);
      } else if (h.reply) {
        const rep = document.createElement('span');
        rep.className = 'log-err';
        rep.style.color = 'var(--ok)';
        rep.textContent = '← ' + h.reply;
        row.appendChild(rep);
      }
      rb.log.appendChild(row);
    }
  }

  function renderClients(addrs) {
    if (!isServer()) { rb.clientList.hidden = true; return; }
    rb.clientList.hidden = false;
    rb.clientList.innerHTML = '';
    if (!addrs || !addrs.length) {
      const empty = document.createElement('div');
      empty.className = 'client-empty';
      empty.textContent = 'No clients connected yet.';
      rb.clientList.appendChild(empty);
      return;
    }
    for (const a of addrs) {
      const row = document.createElement('div');
      row.className = 'client-row';
      row.textContent = a;
      rb.clientList.appendChild(row);
    }
  }

  async function loadRobotConfig() {
    try {
      const c = await api('/api/robot/config');
      rb.mode.value = c.mode || 'server';
      rb.host.value = c.host;
      rb.port.value = c.port;
      rb.timeout.value = c.timeout;
      rb.dryRun.checked = c.dry_run;
      applyModeUI();
      renderLog(c.history);
    } catch (err) {
      console.error(err);
    }
  }

  rb.mode.addEventListener('change', async () => {
    applyModeUI();
    try {
      await api('/api/robot/config', {
        method: 'POST', body: JSON.stringify({ mode: rb.mode.value }),
      });
      rb.connNote.textContent = isServer()
        ? `Server mode - listening on ${rb.host.value}:${rb.port.value}.`
        : `Client mode - will dial out to ${rb.host.value}:${rb.port.value}.`;
      pollStatus();
    } catch (err) {
      rb.connNote.textContent = err.message;
      rb.connNote.classList.add('error');
    }
  });

  rb.dryRun.addEventListener('change', async () => {
    paintDryRun();
    try {
      await api('/api/robot/config', {
        method: 'POST',
        body: JSON.stringify({ dry_run: rb.dryRun.checked }),
      });
      pollStatus();
    } catch (err) {
      rb.connNote.textContent = err.message;
      rb.connNote.classList.add('error');
      rb.dryRun.checked = !rb.dryRun.checked;  // roll back the UI
      paintDryRun();
    }
  });

  rb.saveBtn.addEventListener('click', async () => {
    rb.saveBtn.disabled = true;
    rb.connNote.classList.remove('error');
    try {
      const body = {
        mode: rb.mode.value,
        host: rb.host.value,
        port: Number(rb.port.value),
      };
      if (!isServer()) body.timeout = Number(rb.timeout.value);
      await api('/api/robot/config', {
        method: 'POST', body: JSON.stringify(body),
      });
      rb.connNote.classList.remove('error');
      rb.connNote.textContent = isServer()
        ? `Saved - now listening on ${rb.host.value}:${rb.port.value}.`
        : `Saved - ${rb.host.value}:${rb.port.value}.`;
      pollStatus();
    } catch (err) {
      rb.connNote.textContent = err.message;
      rb.connNote.classList.add('error');
    } finally {
      rb.saveBtn.disabled = false;
    }
  });

  rb.testBtn.addEventListener('click', async () => {
    rb.testBtn.disabled = true;
    const server = isServer();
    setBanner('unknown', server ? 'Checking server…' : 'Connecting…');
    rb.connNote.classList.remove('error');
    try {
      // Client mode: test the values in the boxes so an address can be tried
      // before committing. Server mode ignores them and reports its own state.
      const r = await api('/api/robot/test', {
        method: 'POST',
        body: JSON.stringify({ host: rb.host.value, port: Number(rb.port.value) }),
      });
      if (server) {
        if (r.ok) {
          const n = r.clients || 0;
          setBanner(n > 0 ? 'ok' : 'warn',
            `Listening on ${r.host}:${r.port} - ${n} client${n === 1 ? '' : 's'}`);
          rb.connNote.textContent = n > 0
            ? 'A client is connected and ready to receive commands.'
            : 'Listening, but no client yet. Connect Hercules to this address.';
          renderClients(r.client_addrs);
        } else {
          setBanner('error', 'Server not listening');
          rb.connNote.textContent = r.error;
          rb.connNote.classList.add('error');
        }
      } else if (r.ok) {
        setBanner('ok', `${r.host}:${r.port} accepted in ${r.ms}ms`);
        rb.connNote.textContent = 'Controller is listening.';
      } else {
        setBanner('error', `${r.host}:${r.port} unreachable`);
        rb.connNote.textContent = r.error;
        rb.connNote.classList.add('error');
      }
    } catch (err) {
      setBanner('error', 'Test failed');
      rb.connNote.textContent = err.message;
      rb.connNote.classList.add('error');
    } finally {
      rb.testBtn.disabled = false;
    }
  });

  rb.manualSend.addEventListener('click', async () => {
    const line = rb.manualLine.value.trim();
    if (!line) return;
    // Live mode moves real hardware from a hand-typed string - make the user
    // say yes first. Dry run sends nothing, so it goes straight through.
    const dest = isServer() ? 'connected client(s)'
                            : `${rb.host.value}:${rb.port.value}`;
    if (!rb.dryRun.checked &&
        !confirm(`Send "${line}" to ${dest}?\n\n` +
                 `Dry run is OFF - this will move the arm.`)) {
      return;
    }
    rb.manualSend.disabled = true;
    rb.manualNote.classList.remove('error');
    try {
      const r = await api('/api/robot/send', {
        method: 'POST', body: JSON.stringify({ line }),
      });
      let msg;
      if (r.error) msg = r.error;
      else if (r.dry_run) msg = 'Dry run - formatted but not sent.';
      else if (r.mode === 'server')
        msg = `Broadcast to ${r.clients} client${r.clients === 1 ? '' : 's'} in ${r.ms}ms.`;
      else msg = `Sent in ${r.ms}ms.` + (r.reply ? ` Reply: ${r.reply}` : '');
      rb.manualNote.textContent = msg;
      if (r.error) rb.manualNote.classList.add('error');
      loadRobotConfig();
    } catch (err) {
      rb.manualNote.textContent = err.message;
      rb.manualNote.classList.add('error');
    } finally {
      rb.manualSend.disabled = false;
    }
  });

  rb.manualLine.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); rb.manualSend.click(); }
  });

  rb.clearLog.addEventListener('click', async () => {
    await api('/api/robot/history', { method: 'DELETE' }).catch(console.error);
    renderLog([]);
  });

  // tab switching
  document.querySelectorAll('.vtab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.vtab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const want = tab.dataset.tab;
      $('pane-camera').hidden = want !== 'camera';
      $('pane-robot').hidden = want !== 'robot';
      if (want === 'robot') loadRobotConfig();
    });
  });

  /* ───────────────────────────── status ────────────────────────────── */

  function setDot(key, state) {
    const row = document.querySelector(`.status-row[data-key="${key}"] .dot`);
    if (row) row.dataset.state = state;
  }

  async function pollStatus() {
    try {
      const s = await api('/api/status');

      // model
      if (s.llm.ok) {
        $('statusLlm').textContent = s.llm.model;
        setDot('llm', 'ok');
      } else {
        $('statusLlm').textContent = 'unavailable';
        $('statusLlm').title = s.llm.error || '';
        setDot('llm', 'error');
      }

      // camera. Reconnecting is its own state: the feed is down but the app is
      // actively retrying, so it's a warning, not an error the user must act on.
      const cam = s.camera;
      if (cam.running && cam.has_frame && !cam.reconnecting && cam.dark) {
        // A live stream of black frames. Not an error, but reporting it as
        // healthy is worse than useless — the arm would be looking at nothing.
        $('statusCamera').textContent = `index ${cam.index} — no image`;
        $('statusCamera').title =
          'The camera is streaming, but every frame is black. It may be a '
          + 'depth/IR sensor, a covered lens, or a device another app is holding.';
        setDot('camera', 'warn');
        el.videoFrame.classList.add('live');
      } else if (cam.running && cam.has_frame && !cam.reconnecting) {
        $('statusCamera').textContent = `index ${cam.index}`;
        $('statusCamera').title = '';
        setDot('camera', 'ok');
        el.videoFrame.classList.add('live');
      } else if (cam.reconnecting) {
        $('statusCamera').textContent = 'reconnecting…';
        $('statusCamera').title = cam.error || '';
        setDot('camera', 'warn');
        el.videoFrame.classList.remove('live');
        el.videoFallback.textContent =
          `Lost camera ${cam.index}. Retrying — free the device and it will come back.`;
      } else {
        $('statusCamera').textContent = cam.error ? 'error' : 'starting…';
        $('statusCamera').title = cam.error || '';
        setDot('camera', cam.error ? 'error' : 'warn');
        el.videoFrame.classList.remove('live');
        el.videoFallback.textContent = cam.error || 'Connecting to camera…';
      }
      $('retryCam').hidden = !(cam.error || cam.reconnecting || cam.dark);

      // robot - same signal drives the sidebar row, the tab dot and the banner
      const r = s.robot;
      const server = r.mode === 'server';
      if (r.dry_run) {
        $('statusRobot').textContent = server ? 'dry run (server)' : 'dry run';
        $('statusRobot').title = 'Dry run - commands are formatted but not sent';
        setDot('robot', 'warn');
        setBanner('warn', server
          ? `Dry run - would broadcast on ${r.host}:${r.port}`
          : `Dry run - ${r.host}:${r.port} not contacted`);
      } else if (server) {
        const n = r.clients || 0;
        if (!r.listening) {
          $('statusRobot').textContent = 'bind failed';
          $('statusRobot').title = r.error || '';
          setDot('robot', 'error');
          setBanner('error', `Not listening - ${r.error || r.host + ':' + r.port}`);
        } else if (n > 0) {
          $('statusRobot').textContent = `${n} client${n === 1 ? '' : 's'}`;
          setDot('robot', 'ok');
          setBanner('ok', `Listening on ${r.host}:${r.port} - ${n} client${n === 1 ? '' : 's'}`);
        } else {
          $('statusRobot').textContent = 'listening';
          $('statusRobot').title = `Listening on ${r.host}:${r.port}, no client yet`;
          setDot('robot', 'warn');
          setBanner('warn', `Listening on ${r.host}:${r.port} - no client yet`);
        }
        // keep the client list live while the tab is open
        if (!$('pane-robot').hidden) renderClients(r.client_addrs);
      } else if (r.reachable) {
        $('statusRobot').textContent = `${r.host}:${r.port}`;
        setDot('robot', 'ok');
        setBanner('ok', `Connected - ${r.host}:${r.port}`);
      } else {
        $('statusRobot').textContent = 'unreachable';
        $('statusRobot').title = `${r.host}:${r.port} not accepting connections`;
        setDot('robot', 'error');
        setBanner('error', `Unreachable - ${r.host}:${r.port}`);
      }

      // history. "off" was misleading - without Mongo the chat still works and
      // still remembers, it just forgets when the server stops.
      $('statusMongo').textContent = s.mongo.available ? s.mongo.db : 'in memory';
      $('statusMongo').title = s.mongo.available
        ? `Persisted to ${s.mongo.uri}/${s.mongo.db}`
        : `MongoDB not in use (${s.mongo.reason || 'unreachable'}). `
          + 'History is kept in memory and lost when the server stops.';
      setDot('mongo', s.mongo.available ? 'ok' : 'warn');

      el.hintText.textContent = s.safety_check
        ? 'Enter to send · Shift+Enter for a new line · hand-detection interlock on'
        : 'Enter to send · Shift+Enter for a new line';
    } catch (_) {
      ['llm', 'camera', 'robot', 'mongo'].forEach((k) => setDot(k, 'error'));
    }
  }

  /* ──────────────────────────── lightbox ───────────────────────────── */

  let lightbox = null;
  function openLightbox(src) {
    if (!lightbox) {
      lightbox = document.createElement('div');
      lightbox.className = 'lightbox';
      lightbox.innerHTML = '<img alt="Annotated frame, enlarged">';
      lightbox.addEventListener('click', () => lightbox.setAttribute('hidden', ''));
      document.body.appendChild(lightbox);
    }
    lightbox.querySelector('img').src = src;
    lightbox.removeAttribute('hidden');
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox) lightbox.setAttribute('hidden', '');
  });

  /* ──────────────────────────── panels ─────────────────────────────── */

  $('collapseSidebar').addEventListener('click', () => el.app.classList.add('no-sidebar'));
  $('showSidebar').addEventListener('click', () => el.app.classList.remove('no-sidebar'));
  $('toggleVision').addEventListener('click', () => el.app.classList.toggle('no-vision'));

  /* ───────────────────────────── boot ──────────────────────────────── */

  el.videoFeed.addEventListener('error', () => el.videoFrame.classList.remove('live'));
  el.videoFeed.src = '/video_feed';

  loadDetectors();
  loadCameras();
  loadSessions();
  loadRobotConfig();
  buildParams('none', [], {});
  pollStatus();
  setInterval(pollStatus, 5000);
  el.input.focus();
})();
