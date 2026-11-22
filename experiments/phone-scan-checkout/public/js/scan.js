// 휴대폰 스캔 화면
(() => {
  const $ = (id) => document.getElementById(id);
  const won = (n) => n.toLocaleString('ko-KR') + '원';
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sessionId = new URLSearchParams(location.search).get('s');
  const gate = createScanGate({ gapMs: 1500 });
  let scanner = null;
  let scanning = false;
  let status = 'open';
  let toastTimer = null;
  let toastCls = null;
  let cartItems = new Map(); // barcode → item (SSE로 받은 현재 장바구니)

  // 이미 담긴 상품의 반복 안내(주황)가 방금 뜬 "담김"(초록) 안내를 덮지 않게 한다
  function toast(text, cls) {
    const t = $('toast');
    if (cls === 'warn' && toastCls === 'ok' && t.classList.contains('show')) return;
    t.textContent = text; t.className = `toast show ${cls}`;
    toastCls = cls;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 1800);
  }

  // 서버의 바코드 매칭과 같게: 공백·하이픈 제거, UPC-A(12자리) ↔ EAN-13(앞자리 0) 동일 취급
  function cartItemFor(raw) {
    const s = String(raw).trim().replace(/[\s-]/g, '');
    const cands = [s];
    if (/^\d{12}$/.test(s)) cands.push('0' + s);
    if (/^\d{13}$/.test(s) && s.startsWith('0')) cands.push(s.slice(1));
    for (const c of cands) if (cartItems.has(c)) return cartItems.get(c);
    return null;
  }

  function setStatus(s) {
    status = s;
    const el = $('status');
    el.className = 'badge ' + s;
    el.textContent = { open: '연결됨', closed: '연결 종료' }[s] || s;
    const blocked = s !== 'open';
    $('manual').disabled = blocked;
    $('start').disabled = blocked;
    if (blocked && scanning) stopScan();
    if (s === 'closed') $('camWarn').innerHTML = '<div class="err">이 계산대 연결은 종료되었습니다. 노트북 화면의 새 QR을 다시 찍어주세요.</div>';
  }

  // 결제 완료: 연결은 유지하고 안내만 잠시 보여준 뒤 다음 손님 스캔을 계속한다
  let paidTimer = null;
  function onPaid(receipt) {
    gate.reset();
    toast(`결제 완료 ${won(receipt.total)} · 다음 손님 스캔 가능`, 'ok');
    if (navigator.vibrate) navigator.vibrate([80, 60, 80]);
    $('camWarn').innerHTML = `<div class="ok">결제 완료: ${won(receipt.total)} (주문 #${receipt.orderId}). 다음 손님 상품을 바로 스캔하세요.</div>`;
    $('last').textContent = '';
    clearTimeout(paidTimer);
    paidTimer = setTimeout(() => { if (status === 'open') $('camWarn').innerHTML = ''; }, 5000);
  }

  function renderCart(cart) {
    cartItems = new Map(cart.items.map((it) => [it.barcode, it]));
    $('count').textContent = cart.count;
    $('total').textContent = won(cart.total);
    $('list').innerHTML = cart.items.map((it) => `<div><span>${esc(it.name)} × ${it.qty}</span><span>${won(it.price * it.qty)}</span></div>`).join('');
    setStatus(cart.status);
  }

  async function api(method, url, body) {
    const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(data.message || data.error || r.statusText), { status: r.status, data });
    return data;
  }

  function showDuplicate(barcode, name) {
    toast(`이미 담긴 상품입니다: ${name}`, 'warn');
    $('last').textContent = `마지막 스캔: ${barcode} → ${name} (이미 담김, 수량은 계산대에서 변경)`;
    if (navigator.vibrate) navigator.vibrate([40, 40, 40]);
  }

  // 인식된 바코드는 모두 이 대기열을 거친다. 요청 중이라고 버리지 않는다.
  const queue = createScanQueue({
    send: sendScan,
    isInCart: (code) => !!cartItemFor(code),
    onLocalDuplicate: (code) => { const it = cartItemFor(code); if (it) showDuplicate(code, it.name); },
  });
  function addBarcode(barcode, source) {
    if (status !== 'open') return;
    queue.push(barcode, source);
  }

  async function sendScan(barcode, source) {
    try {
      const r = await api('POST', `/api/sessions/${sessionId}/scan`, { barcode, source });
      if (r.duplicate) {
        showDuplicate(barcode, r.item.name);
      } else {
        toast(`✔ ${r.item.name} 담김`, 'ok');
        $('last').textContent = `마지막 스캔: ${barcode} → ${r.item.name}`;
        if (navigator.vibrate) navigator.vibrate(80);
      }
      renderCart(r.cart);
    } catch (err) {
      if (err.data && err.data.error === 'unregistered') {
        toast(`미등록 상품입니다 (${barcode})`, 'err');
        $('last').textContent = `마지막 스캔: ${barcode} → 미등록`;
        if (navigator.vibrate) navigator.vibrate([60, 60, 60]);
      } else if (err.status === 410) { setStatus('closed'); toast('연결이 종료되었습니다', 'err'); }
      else if (err.status === 404) { showInvalid(err.message); }
      else toast(err.message, 'err');
    }
  }

  // ---------- 카메라 ----------
  const FORMATS = (() => {
    const F = window.Html5QrcodeSupportedFormats;
    return F ? [F.EAN_13, F.EAN_8, F.UPC_A, F.UPC_E, F.CODE_128, F.CODE_39, F.QR_CODE] : undefined;
  })();

  async function startScan() {
    if (!window.isSecureContext || !navigator.mediaDevices) {
      $('camWarn').innerHTML = '<div class="err">이 주소는 HTTPS가 아니어서 브라우저가 카메라를 막습니다. 아래 "바코드 직접 입력"을 사용하거나 README의 HTTPS 접속 방법을 따라 주세요.</div>';
      return;
    }
    if (!window.Html5Qrcode) { $('camWarn').innerHTML = '<div class="err">스캔 라이브러리를 불러오지 못했습니다. 페이지를 새로고침하세요.</div>'; return; }
    try {
      scanner = scanner || new Html5Qrcode('reader', { formatsToSupport: FORMATS, verbose: false, experimentalFeatures: { useBarCodeDetectorIfSupported: true } });
      // 인식 영역을 화면 대부분으로 넓힌다. 손님은 상품을 정확히 가운데에 대지 않는다.
      const box = (w, h) => ({ width: Math.max(50, Math.floor(w * 0.92)), height: Math.max(50, Math.floor(h * 0.75)) });
      const onDecoded = (text) => { if (gate.check(text)) addBarcode(text, 'camera'); };
      const onMiss = () => { /* 프레임마다 인식 실패는 정상 */ };
      // 전면 카메라('user')를 기본으로 사용. 전면은 좌우 반전 영상이 올 수 있어 disableFlip을 끈다.
      // 전면 카메라는 기본 해상도가 낮아 작은 바코드를 놓치므로 1080p를 먼저 요청하고, 안 되면 기본값으로 다시 연다.
      const base = { fps: 10, qrbox: box, disableFlip: false };
      try {
        await scanner.start({ facingMode: 'user' },
          { ...base, videoConstraints: { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1080 } } },
          onDecoded, onMiss);
      } catch (hiResErr) {
        if (/NotAllowed|Permission|denied/i.test(String(hiResErr && (hiResErr.message || hiResErr)))) throw hiResErr;
        await scanner.start({ facingMode: 'user' }, { ...base, aspectRatio: 1.333 }, onDecoded, onMiss);
      }
      scanning = true;
      gate.reset();
      $('start').style.display = 'none';
      $('stop').style.display = '';
      $('camWarn').innerHTML = '';
    } catch (err) {
      const m = String(err && (err.message || err));
      let hint = m;
      if (/NotAllowed|Permission|denied/i.test(m)) hint = '카메라 권한이 거부되었습니다. 브라우저 주소창의 자물쇠 아이콘에서 카메라를 허용하거나, 아래 직접 입력을 사용하세요.';
      else if (/NotFound|no camera|Requested device not found/i.test(m)) hint = '사용할 수 있는 카메라가 없습니다. 아래 직접 입력을 사용하세요.';
      $('camWarn').innerHTML = `<div class="err">${esc(hint)}</div>`;
    }
  }

  async function stopScan() {
    if (!scanner || !scanning) return;
    try { await scanner.stop(); } catch { /* ignore */ }
    scanning = false;
    $('start').style.display = '';
    $('stop').style.display = 'none';
  }

  $('start').addEventListener('click', startScan);
  $('stop').addEventListener('click', stopScan);

  // ---------- 직접 입력 ----------
  $('manualForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const v = $('manual').value.trim();
    if (!v) return;
    addBarcode(v, 'manual');
    $('manual').value = '';
  });

  // ---------- 세션 연결 ----------
  function showInvalid(text) {
    $('invalid').style.display = '';
    $('invalid').textContent = text || '유효하지 않은 세션입니다. 노트북 화면의 QR을 다시 찍어주세요.';
    $('scanCard').style.display = 'none';
    $('manual').disabled = true;
    setStatus('closed');
  }

  function connectEvents() {
    const es = new EventSource(`/api/sessions/${sessionId}/events`);
    es.addEventListener('cart', (e) => renderCart(JSON.parse(e.data)));
    es.addEventListener('paid', (e) => onPaid(JSON.parse(e.data)));
    es.addEventListener('closed', () => { setStatus('closed'); es.close(); });
  }

  (async () => {
    if (!sessionId) return showInvalid('세션 정보가 없습니다. 노트북 화면의 QR을 찍어서 접속하세요.');
    try {
      renderCart(await api('GET', `/api/sessions/${sessionId}`));
      connectEvents();
    } catch (err) { showInvalid(err.message); }
  })();
})();
