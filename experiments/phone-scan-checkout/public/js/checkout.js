// 노트북 계산대 화면
(() => {
  const $ = (id) => document.getElementById(id);
  const won = (n) => n.toLocaleString('ko-KR') + '원';
  let session = null;
  let es = null;

  async function api(method, url, body) {
    const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(data.message || data.error || r.statusText), { status: r.status, data });
    return data;
  }

  function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  function setStatus(status) {
    const el = $('status');
    el.className = 'badge ' + status;
    el.textContent = { open: '열림', closed: '종료됨' }[status] || status;
  }

  function render(cart) {
    session = cart;
    setStatus(cart.status);
    const tbody = $('items');
    tbody.innerHTML = '';
    for (const it of cart.items) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${esc(it.name)}<div class="muted mono">${esc(it.barcode)}</div></td>
        <td class="num">${won(it.price)}</td>
        <td class="num">
          <span class="qty">
            <button class="sm" data-act="dec" data-bc="${esc(it.barcode)}">−</button>
            <input type="number" min="0" max="999" value="${it.qty}" data-act="set" data-bc="${esc(it.barcode)}">
            <button class="sm" data-act="inc" data-bc="${esc(it.barcode)}">+</button>
          </span>
        </td>
        <td class="num"><b>${won(it.price * it.qty)}</b></td>
        <td class="num"><button class="sm danger" data-act="del" data-bc="${esc(it.barcode)}">삭제</button></td>`;
      tbody.appendChild(tr);
    }
    $('empty').style.display = cart.items.length ? 'none' : '';
    $('count').textContent = cart.count;
    $('total').textContent = won(cart.total);
    const open = cart.status === 'open';
    $('pay').disabled = !open || cart.items.length === 0;
    $('clearCart').disabled = !open || cart.items.length === 0;
    const ev = cart.lastEvent;
    const msg = ev && {
      added: `✔ "${ev.name}" 담김`,
      qty: `수량 변경: ${ev.name} → ${ev.qty}`,
      removed: `삭제: ${ev.name}`,
      paid: `결제 완료 #${ev.orderId} · ${won(ev.total || 0)} — 다음 손님 상품을 스캔하세요`,
      cleared: '장바구니를 비웠습니다',
    }[ev.type];
    $('event').textContent = msg || '';
    for (const b of document.querySelectorAll('[data-act]')) b.disabled = !open;
  }

  async function loadQr() {
    const q = await api('GET', `/api/sessions/${session.sessionId}/qr`);
    $('qr').src = q.qrDataUrl;
    $('sid').textContent = q.sessionId;
    $('scanUrl').textContent = q.scanUrl;
    $('qrWarn').innerHTML = q.warnings.map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join('');
  }

  async function loadPublicUrlSetting() {
    const s = await api('GET', '/api/settings/public-url');
    $('publicUrl').value = s.value;
    const src = { setting: '저장된 설정', env: '환경변수 PUBLIC_URL', auto: '자동(LAN IP)' }[s.source];
    $('publicUrlInfo').textContent = `현재 적용: ${s.effective} (${src})` + (s.lanIp ? ` · 이 PC의 LAN IP: ${s.lanIp}` : '');
  }

  async function loadOrders() {
    const orders = await api('GET', '/api/orders?limit=20');
    $('orderCount').textContent = orders.length ? `(최근 ${orders.length}건)` : '';
    $('ordersEmpty').style.display = orders.length ? 'none' : '';
    $('orders').innerHTML = orders.map((o) => `
      <tr>
        <td class="mono">#${o.id}</td>
        <td class="muted">${esc(o.paidAt)}</td>
        <td>${o.items.map((it) => `${esc(it.name)} × ${it.qty}`).join(', ')}</td>
        <td class="num">${o.count}</td>
        <td class="num"><b>${won(o.total)}</b></td>
      </tr>`).join('');
  }

  function connectEvents() {
    if (es) es.close();
    es = new EventSource(`/api/sessions/${session.sessionId}/events`);
    es.addEventListener('cart', (e) => render(JSON.parse(e.data)));
    es.addEventListener('paid', (e) => { showReceipt(JSON.parse(e.data)); loadOrders().catch(() => {}); });
    es.addEventListener('closed', () => { es.close(); $('closedOverlay').classList.add('show'); });
    es.onerror = () => { /* 브라우저가 자동 재연결 */ };
  }

  function showReceipt(r) {
    $('orderId').textContent = r.orderId;
    $('paidAt').textContent = new Date(r.paidAt).toLocaleString('ko-KR');
    $('receiptItems').innerHTML = r.items.map((it) => `<tr><td>${esc(it.name)}</td><td class="num">${it.qty}</td><td class="num">${won(it.price * it.qty)}</td></tr>`).join('');
    $('receiptTotal').textContent = won(r.total);
    $('paidOverlay').classList.add('show');
  }

  async function openSession(cart) {
    render(cart);
    await loadQr();
    connectEvents();
  }

  // 수량 조작
  document.addEventListener('click', async (e) => {
    const b = e.target.closest('[data-act]');
    if (!b || b.tagName === 'INPUT') return;
    const bc = b.dataset.bc;
    const it = session.items.find((x) => x.barcode === bc);
    if (!it) return;
    try {
      if (b.dataset.act === 'inc') await api('PATCH', `/api/sessions/${session.sessionId}/items/${bc}`, { qty: it.qty + 1 });
      if (b.dataset.act === 'dec') await api('PATCH', `/api/sessions/${session.sessionId}/items/${bc}`, { qty: it.qty - 1 });
      if (b.dataset.act === 'del') await api('DELETE', `/api/sessions/${session.sessionId}/items/${bc}`);
    } catch (err) { alert(err.message); }
  });
  document.addEventListener('change', async (e) => {
    const i = e.target;
    if (i.dataset.act !== 'set') return;
    const qty = Math.max(0, Math.min(999, parseInt(i.value, 10) || 0));
    try { await api('PATCH', `/api/sessions/${session.sessionId}/items/${i.dataset.bc}`, { qty }); }
    catch (err) { alert(err.message); }
  });

  $('pay').addEventListener('click', async () => {
    try {
      const r = await api('POST', `/api/sessions/${session.sessionId}/checkout`);
      showReceipt(r);
      await loadOrders();
    } catch (err) { alert(err.message); }
  });
  $('nextCart').addEventListener('click', () => $('paidOverlay').classList.remove('show'));
  $('clearCart').addEventListener('click', async () => {
    if (!confirm('장바구니를 비울까요? (결제 기록은 남지 않습니다)')) return;
    try { await api('POST', `/api/sessions/${session.sessionId}/clear`); } catch (err) { alert(err.message); }
  });
  $('newCart').addEventListener('click', async () => {
    if (!confirm('새 QR을 발급하고 연결된 휴대폰을 모두 끊을까요? 현재 장바구니도 비워집니다.')) return;
    $('closedOverlay').classList.remove('show');
    await openSession(await api('POST', '/api/sessions'));
  });
  $('reloadCurrent').addEventListener('click', async () => {
    $('closedOverlay').classList.remove('show');
    await openSession(await api('GET', '/api/sessions/current'));
  });
  $('copyUrl').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText($('scanUrl').textContent); $('copyUrl').textContent = '복사됨'; setTimeout(() => ($('copyUrl').textContent = '주소 복사'), 1500); }
    catch { alert($('scanUrl').textContent); }
  });
  $('savePublicUrl').addEventListener('click', async () => {
    try {
      await api('PUT', '/api/settings/public-url', { value: $('publicUrl').value });
      await loadPublicUrlSetting();
      await loadQr();
    } catch (err) { alert(err.message); }
  });

  (async () => {
    await loadPublicUrlSetting();
    await openSession(await api('GET', '/api/sessions/current'));
    await loadOrders();
  })().catch((e) => { alert('서버에 연결할 수 없습니다: ' + e.message); });
})();
