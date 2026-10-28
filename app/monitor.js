/* Read-only telemetry. Command execution uses app.js's existing dispatcher. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let status = {}, points = [], received = 0, envelopeReceived = 0, busy = false;
  const classes = ['open', 'close', 'next', 'music', 'louder', 'photo', 'previous', 'quieter', 'start', 'stop', 'rest'];
  async function api(path, data) {
    const response = await fetch('/api/board/' + path, data === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || '设备请求失败');
    return body;
  }
  function error(text) { $('board-error').textContent = text || ''; $('board-error').hidden = !text; }
  function controls() {
    const active = status.connected || status.connecting;
    $('connect-board').disabled = busy || active;
    $('disconnect-board').disabled = busy || !active;
    $('pause-board').disabled = busy || !status.connected || !status.verified || status.stale || !!(status.latest && status.latest.flags);
    for (const id of ['board-port', 'board-baud', 'board-transport', 'refresh-ports']) $(id).disabled = busy || active;
  }
  function render(data) {
    status = data;
    const live = data.connected && data.verified && !data.stale;
    const fault = data.latest && data.latest.flags;
    $('board-status').textContent = data.connecting ? '正在连接设备…' : !data.connected ? '未连接设备' : fault ? (fault & 4 ? 'UART 故障 · 请复位 STM32' : '板端故障 · 控制已阻止') : !data.verified ? '正在验证板端模型…' : data.stale ? '等待新推理数据…' : data.enabled ? '实时控制已启用' : '仅监测 · 控制已暂停';
    $('board-status').classList.toggle('is-live', !!live && data.enabled && !fault);
    controls();
    $('pause-board').textContent = data.enabled ? '暂停控制' : '启用控制';
    if (data.connected || data.connecting) {
      $('board-transport').value = data.transport || 'serial';
      $('refresh-ports').textContent = data.transport === 'ble' ? '扫描 BT24' : '刷新';
      if (data.port && ![...$('board-port').options].some(p => p.value === data.port)) $('board-port').replaceChildren(new Option(data.port, data.port));
    }
    if (data.connected && data.baudrate) $('board-baud').value = String(data.baudrate);
    if (data.port && [...$('board-port').options].some(p => p.value === data.port)) $('board-port').value = data.port;
    error(data.error);
    const r = live && data.latest;
    $('board-candidate').textContent = r ? `${classes[r.top]} · ${(r.probabilities[r.top] * 100).toFixed(1)}%` : '—';
    $('board-timing').textContent = r ? `${(r.inference_us / 1000).toFixed(2)} / ${(r.pipeline_us / 1000).toFixed(2)} ms` : '—';
    $('board-gate').textContent = r ? (r.armed ? '已解锁 · 等待命令' : '等待连续静息解锁') : '等待设备';
    $('board-confirmed').textContent = data.last_command ? `${data.last_command.word} · ${new Date(data.last_command.sent_at).toLocaleTimeString()}` : '—';
    $('board-model').textContent = data.info ? `板端模型 ${data.info.model} · SHA ${data.info.sha} · ${data.verified ? '已验证' : '未验证'} · 100 ms 步进 · ${data.transport === 'ble' ? 'BLE / ' : ''}UART ${data.baudrate || 230400}` : '等待模型身份。请选择设备连接，并关闭其他软件对同一设备的占用。';
    if (r) $('board-health').textContent = `每步最大 ${(r.max_pipeline_us / 1000).toFixed(2)} ms · ≥100 ms ${r.over_100ms} 次 · ≥200 ms ${r.over_200ms} 次 · 跳窗 ${r.skipped_windows} · 板端丢包 ${r.tx_dropped} · 序号异常 ${data.sequence_gaps} · CRC ${data.crc_errors} · 故障 ${r.flags} · ADC削顶 ${r.clipped_samples}`;
    else $('board-health').textContent = '等待有效板端推理数据。仅确认命令驱动照片与音乐。';
    if (!data.connected) { points = []; envelopeReceived = 0; draw(); }
  }
  function draw() {
    const canvas = $('board-envelope'), ctx = canvas.getContext('2d');
    canvas.width = Math.max(240, Math.round(canvas.clientWidth));
    const w = canvas.width, h = canvas.height, left = 42, right = w - 14, top = 16, bottom = h - 26;
    ctx.clearRect(0, 0, w, h); ctx.font = '13px sans-serif';
    for (const v of [0, 250, 500]) {
      const y = bottom - v / 500 * (bottom - top);
      ctx.strokeStyle = '#e0e5df'; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.fillStyle = '#526157'; ctx.fillText(String(v), 5, y + 4);
    }
    ctx.fillText('−5 s', left, h - 5); ctx.fillText('现在', right - 30, h - 5);
    if (!points.length) { $('board-envelope-values').textContent = '无实时包络'; return; }
    const end = points[points.length - 1][0], latest = points[points.length - 1];
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    [1, 2].forEach((ch, index) => {
      ctx.strokeStyle = index ? '#d66b12' : '#14834b'; ctx.lineWidth = 2; ctx.beginPath();
      points.forEach((p, i) => {
        const x = right + (p[0] - end) / 10000 * (right - left), y = bottom - p[ch] / 500 * (bottom - top);
        if (!i || p[0] - points[i - 1][0] !== 20) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }); ctx.stroke();
    }); ctx.restore();
    const peak = Math.max(...points.map(p => Math.max(p[1], p[2])));
    $('board-envelope-values').textContent = `CH0 ${latest[1].toFixed(1)} · CH1 ${latest[2].toFixed(1)}${peak > 500 ? ` · 超出量程（5秒峰值 ${peak.toFixed(0)}）` : ''}`;
  }
  async function refresh() {
    const ble = $('board-transport').value === 'ble';
    const selected = $('board-port').value, data = await api(ble ? 'scan-ble' : 'ports', ble ? {} : undefined);
    $('board-port').replaceChildren(...data.ports.map(p => {
      const option = document.createElement('option'); option.value = p.device;
      option.textContent = `${p.device} · ${p.description}`; return option;
    }));
    if (!data.ports.length) { const option = document.createElement('option'); option.value = ''; option.textContent = ble ? '未发现 BT24，请检查供电及其他设备占用' : '未发现串口'; $('board-port').append(option); }
    if (data.ports.some(p => p.device === selected)) $('board-port').value = selected;
    else if (data.ports.some(p => p.device === 'COM7')) $('board-port').value = 'COM7';
  }
  async function action(callback) {
    busy = true; render(status);
    try { await callback(); } catch (e) { error(e.message); }
    finally { busy = false; controls(); }
  }
  $('board-transport').onchange = () => {
    const ble = $('board-transport').value === 'ble';
    $('refresh-ports').textContent = ble ? '扫描 BT24' : '刷新';
    $('board-port').replaceChildren(new Option(ble ? '请先扫描 BT24' : '请刷新串口', ''));
    error('');
  };
  $('refresh-ports').onclick = () => action(refresh);
  $('connect-board').onclick = () => action(async () => { points = []; draw(); render(await api('connect', {port: $('board-port').value, baudrate: Number($('board-baud').value), transport: $('board-transport').value})); });
  $('disconnect-board').onclick = () => action(async () => render(await api('disconnect', {})));
  $('pause-board').onclick = () => action(async () => render(await api('control', {enabled: !status.enabled})));
  window.addEventListener('quietly-board', e => { received = Date.now(); render(e.detail); });
  window.addEventListener('quietly-envelope', e => {
    if (Date.now() - e.detail.sent_at > 1000) return;
    for (const p of e.detail.points) {
      if (points.length && p[0] <= points[points.length - 1][0]) points = [];
      points.push(p);
    }
    points = points.slice(-501); envelopeReceived = Date.now(); draw();
  });
  setInterval(() => {
    if (received && Date.now() - received > 1500 && status.connected) render({...status, stale: true});
    if (envelopeReceived && Date.now() - envelopeReceived > 1000) { points = []; envelopeReceived = 0; draw(); }
  }, 500);
  draw();
  window.addEventListener('resize', draw);
  if (location.protocol === 'file:') {
    for (const id of ['connect-board', 'refresh-ports']) $(id).disabled = true;
    error('请运行 app/start.cmd，直接打开 HTML 只能手动预览。');
  } else action(async () => { await refresh(); const initial = await api('status'); received = Date.now(); render(initial); });
})();
