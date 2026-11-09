'use strict';
const express = require('express');
const path = require('path');
const os = require('os');
const fs = require('fs');
const http = require('http');
const https = require('https');
const crypto = require('crypto');
const QRCode = require('qrcode');
const store = require('./db');

const PORT = Number(process.env.PORT || 3000);
const HTTPS_PORT = Number(process.env.HTTPS_PORT || 3443);

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use('/vendor/html5-qrcode', express.static(path.join(__dirname, 'node_modules/html5-qrcode')));
app.use('/vendor/jsbarcode', express.static(path.join(__dirname, 'node_modules/jsbarcode/dist')));

// 페이지 라우트
app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'public/index.html')));
app.get('/admin', (_req, res) => res.sendFile(path.join(__dirname, 'public/admin.html')));
app.get('/scan', (_req, res) => res.sendFile(path.join(__dirname, 'public/scan.html')));
app.get('/samples', (_req, res) => res.sendFile(path.join(__dirname, 'public/samples.html')));

// ---------- 계산대 세션 (메모리) ----------
// 장바구니는 시연용이므로 메모리에만 둔다. 상품/설정/주문 기록은 SQLite에 남는다.
const sessions = new Map();
let currentSessionId = null;

const ID_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789';
function shortId(len) {
  const bytes = crypto.randomBytes(len);
  let out = '';
  for (let i = 0; i < len; i++) out += ID_ALPHABET[bytes[i] % ID_ALPHABET.length];
  return out;
}

function newSession() {
  const id = shortId(8); // QR에 넣기 좋은 짧은 ID (0/O, 1/l/I 처럼 헷갈리는 문자는 제외)
  const s = { id, status: 'open', createdAt: new Date().toISOString(), items: new Map(), clients: new Set(), lastEvent: null, orderCount: 0 };
  sessions.set(id, s);
  return s;
}

function cartSnapshot(s) {
  const items = [...s.items.values()];
  const total = items.reduce((sum, it) => sum + it.price * it.qty, 0);
  const count = items.reduce((sum, it) => sum + it.qty, 0);
  return { sessionId: s.id, status: s.status, items, total, count, orderCount: s.orderCount || 0, lastEvent: s.lastEvent, updatedAt: new Date().toISOString() };
}

function broadcast(s, event, data) {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of s.clients) {
    try { res.write(payload); } catch { s.clients.delete(res); }
  }
}

function closeSession(s, reason) {
  if (s.status === 'closed') return;
  s.status = 'closed';
  broadcast(s, 'closed', { sessionId: s.id, reason });
  for (const res of s.clients) { try { res.end(); } catch { /* ignore */ } }
  s.clients.clear();
  // 종료된 세션은 잠시 후 메모리에서 제거 (휴대폰이 늦게 요청해도 "종료됨" 안내를 받도록 5분 유지)
  setTimeout(() => sessions.delete(s.id), 5 * 60 * 1000).unref();
}

function startNewCart(reason) {
  if (currentSessionId && sessions.has(currentSessionId)) closeSession(sessions.get(currentSessionId), reason);
  const s = newSession();
  currentSessionId = s.id;
  return s;
}

function getSessionOr404(req, res) {
  const s = sessions.get(req.params.id);
  if (!s) {
    res.status(404).json({ error: 'not_found', message: '존재하지 않거나 만료된 계산대 세션입니다. 노트북 화면의 QR을 다시 찍어주세요.' });
    return null;
  }
  return s;
}

// ---------- 공개 접속 주소 ----------
// 실제 Wi-Fi/이더넷 주소를 고른다. WSL·Hyper-V·VirtualBox 같은 가상 어댑터와
// 169.254.x.x(자동 사설 주소)는 휴대폰에서 접속할 수 없으므로 뒤로 미룬다.
const VIRTUAL_IFACE = /vEthernet|WSL|Hyper-V|VirtualBox|VMware|docker|Loopback|Tailscale|ZeroTier|Bluetooth/i;
const PREFERRED_IFACE = /Wi-?Fi|무선|Ethernet|이더넷|^en\d|^eth\d|^wl/i;
function lanIp() {
  const candidates = [];
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const i of ifaces[name]) {
      if (i.family !== 'IPv4' || i.internal || i.address.startsWith('169.254.')) continue;
      const score = VIRTUAL_IFACE.test(name) ? 2 : PREFERRED_IFACE.test(name) ? 0 : 1;
      candidates.push({ score, address: i.address });
    }
  }
  candidates.sort((a, b) => a.score - b.score);
  return candidates.length ? candidates[0].address : null;
}

function resolvePublicBase() {
  const fromSetting = (store.getSetting('public_url') || '').trim();
  const fromEnv = (process.env.PUBLIC_URL || '').trim();
  let base = fromSetting || fromEnv;
  const source = fromSetting ? 'setting' : fromEnv ? 'env' : 'auto';
  if (!base) {
    const ip = lanIp();
    base = ip ? `http://${ip}:${PORT}` : `http://localhost:${PORT}`;
  }
  base = base.replace(/\/+$/, '');
  const warnings = [];
  if (/localhost|127\.0\.0\.1/.test(base)) {
    warnings.push('주소에 localhost가 포함되어 있어 휴대폰에서 열리지 않습니다. 공개 주소(터널 HTTPS 주소)를 설정하세요.');
  } else if (base.startsWith('http://')) {
    warnings.push('HTTP 주소입니다. 휴대폰 브라우저는 HTTPS가 아니면 카메라를 막으므로 바코드 직접 입력만 가능합니다. README의 HTTPS 설정을 참고하세요.');
  }
  return { base, source, warnings };
}

// ---------- API: 세션 ----------
app.get('/api/sessions/current', (_req, res) => {
  let s = currentSessionId && sessions.get(currentSessionId);
  if (!s || s.status === 'closed') s = startNewCart('restart');
  res.json(cartSnapshot(s));
});

app.post('/api/sessions', (_req, res) => {
  const s = startNewCart('new_cart');
  res.status(201).json(cartSnapshot(s));
});

app.get('/api/sessions/:id', (req, res) => {
  const s = getSessionOr404(req, res); if (!s) return;
  res.json(cartSnapshot(s));
});

app.get('/api/sessions/:id/qr', async (req, res) => {
  const s = getSessionOr404(req, res); if (!s) return;
  const pub = resolvePublicBase();
  const scanUrl = `${pub.base}/scan?s=${s.id}`;
  const qrDataUrl = await QRCode.toDataURL(scanUrl, { width: 320, margin: 1, errorCorrectionLevel: 'M' });
  res.json({ sessionId: s.id, scanUrl, qrDataUrl, publicBase: pub.base, source: pub.source, warnings: pub.warnings });
});

// SSE: 노트북(그리고 휴대폰)이 장바구니 변화를 실시간으로 받는다
app.get('/api/sessions/:id/events', (req, res) => {
  const s = getSessionOr404(req, res); if (!s) return;
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write(`event: cart\ndata: ${JSON.stringify(cartSnapshot(s))}\n\n`);
  if (s.status === 'closed') {
    res.write(`event: closed\ndata: ${JSON.stringify({ sessionId: s.id })}\n\n`);
    res.end();
    return;
  }
  s.clients.add(res);
  const hb = setInterval(() => { try { res.write(': ping\n\n'); } catch { /* ignore */ } }, 25000);
  req.on('close', () => { clearInterval(hb); s.clients.delete(res); });
});

// 휴대폰 스캔 → 장바구니 담기
app.post('/api/sessions/:id/scan', (req, res) => {
  const s = getSessionOr404(req, res); if (!s) return;
  if (s.status === 'closed') return res.status(410).json({ error: 'session_closed', message: '이 계산대 세션은 종료되었습니다. 노트북의 새 QR을 다시 찍어주세요.' });
  const barcode = store.normalizeBarcode(req.body && req.body.barcode);
  if (!barcode) return res.status(400).json({ error: 'bad_request', message: '바코드 번호가 비어 있습니다.' });
  const p = store.findProductByBarcode(barcode);
  if (!p) return res.status(404).json({ error: 'unregistered', barcode, message: `미등록 바코드입니다: ${barcode}` });
  // 같은 바코드를 다시 찍어도 수량을 올리지 않는다. 수량은 노트북 계산대에서만 바꾼다.
  const existing = s.items.get(p.barcode);
  if (existing) {
    return res.status(200).json({ ok: false, duplicate: true, item: existing, cart: cartSnapshot(s), message: `이미 담긴 상품입니다: ${existing.name} (수량은 계산대에서 변경)` });
  }
  const it = { barcode: p.barcode, name: p.name, price: p.price, qty: 1 };
  s.items.set(p.barcode, it);
  s.lastEvent = { type: 'added', barcode: it.barcode, name: it.name, qty: it.qty, source: (req.body && req.body.source) || 'camera', at: new Date().toISOString() };
  const cart = cartSnapshot(s);
  broadcast(s, 'cart', cart);
  res.json({ ok: true, item: it, cart });
});

// 노트북: 수량 수정 / 삭제
app.patch('/api/sessions/:id/items/:barcode', (req, res) => {
  const s = getSessionOr404(req, res); if (!s) return;
  if (s.status !== 'open') return res.status(409).json({ error: 'not_open', message: '열려 있는 장바구니가 아닙니다.' });
  const it = s.items.get(req.params.barcode);
  if (!it) return res.status(404).json({ error: 'item_not_found', message: '장바구니에 없는 상품입니다.' });
  const qty = Number(req.body && req.body.qty);
  if (!Number.isInteger(qty) || qty < 0 || qty > 999) return res.status(400).json({ error: 'bad_qty', message: '수량은 0~999 사이 정수여야 합니다.' });
  if (qty === 0) s.items.delete(it.barcode); else it.qty = qty;
  s.lastEvent = { type: qty === 0 ? 'removed' : 'qty', barcode: it.barcode, name: it.name, qty, at: new Date().toISOString() };
  const cart = cartSnapshot(s);
  broadcast(s, 'cart', cart);
  res.json(cart);
});

app.delete('/api/sessions/:id/items/:barcode', (req, res) => {
  const s = getSessionOr404(req, res); if (!s) return;
  if (s.status !== 'open') return res.status(409).json({ error: 'not_open', message: '열려 있는 장바구니가 아닙니다.' });
  const it = s.items.get(req.params.barcode);
  if (!it) return res.status(404).json({ error: 'item_not_found', message: '장바구니에 없는 상품입니다.' });
  s.items.delete(it.barcode);
  s.lastEvent = { type: 'removed', barcode: it.barcode, name: it.name, qty: 0, at: new Date().toISOString() };
  const cart = cartSnapshot(s);
  broadcast(s, 'cart', cart);
  res.json(cart);
});

// 모의 결제: 카드 정보 입력/청구 없이 품목·총액을 주문 기록(SQLite)으로 남기고 장바구니를 비운다.
// 계산대 세션과 휴대폰 연결은 그대로 유지되어 다음 손님을 바로 스캔할 수 있다.
app.post('/api/sessions/:id/checkout', (req, res) => {
  const s = getSessionOr404(req, res); if (!s) return;
  if (s.status !== 'open') return res.status(409).json({ error: 'not_open', message: '열려 있는 계산대가 아닙니다.' });
  if (s.items.size === 0) return res.status(400).json({ error: 'empty_cart', message: '장바구니가 비어 있습니다.' });
  const cart = cartSnapshot(s);
  const orderId = store.recordOrder(s.id, cart.total, cart.items);
  const receipt = { orderId, sessionId: s.id, items: cart.items, total: cart.total, count: cart.count, paidAt: new Date().toISOString(), method: 'TEST (모의 결제)' };
  s.items.clear();
  s.orderCount = (s.orderCount || 0) + 1;
  s.lastEvent = { type: 'paid', orderId, total: receipt.total, count: receipt.count, at: receipt.paidAt };
  broadcast(s, 'paid', receipt);
  broadcast(s, 'cart', cartSnapshot(s));
  res.json(receipt);
});

// 결제 없이 장바구니만 비우기 (손님이 구매를 취소했을 때)
app.post('/api/sessions/:id/clear', (req, res) => {
  const s = getSessionOr404(req, res); if (!s) return;
  if (s.status !== 'open') return res.status(409).json({ error: 'not_open', message: '열려 있는 계산대가 아닙니다.' });
  s.items.clear();
  s.lastEvent = { type: 'cleared', at: new Date().toISOString() };
  const cart = cartSnapshot(s);
  broadcast(s, 'cart', cart);
  res.json(cart);
});

// ---------- API: 주문(결제) 기록 ----------
app.get('/api/orders', (req, res) => {
  const limit = Math.min(200, Math.max(1, Number(req.query.limit) || 20));
  res.json(store.listOrders(limit));
});

// ---------- API: 상품 ----------
app.get('/api/products', (_req, res) => res.json(store.listProducts()));
app.get('/api/products/lookup/:barcode', (req, res) => {
  const p = store.findProductByBarcode(req.params.barcode);
  if (!p) return res.status(404).json({ error: 'unregistered' });
  res.json(p);
});
app.post('/api/products', (req, res) => {
  const r = store.createProduct(req.body || {});
  res.status(r.status).json(r.error ? { error: r.error } : r.product);
});
app.put('/api/products/:id', (req, res) => {
  const r = store.updateProduct(Number(req.params.id), req.body || {});
  res.status(r.status).json(r.error ? { error: r.error } : r.product);
});
app.delete('/api/products/:id', (req, res) => {
  const r = store.deleteProduct(Number(req.params.id));
  res.status(r.status).json(r.error ? { error: r.error } : { ok: true });
});

// ---------- API: 설정 ----------
app.get('/api/settings/public-url', (_req, res) => {
  const pub = resolvePublicBase();
  res.json({ value: store.getSetting('public_url') || '', env: process.env.PUBLIC_URL || '', effective: pub.base, source: pub.source, warnings: pub.warnings, lanIp: lanIp() });
});
app.put('/api/settings/public-url', (req, res) => {
  const v = String((req.body && req.body.value) || '').trim().replace(/\/+$/, '');
  if (v && !/^https?:\/\/[^\s/]+/.test(v)) return res.status(400).json({ error: 'http:// 또는 https:// 로 시작하는 주소를 입력하세요.' });
  store.setSetting('public_url', v);
  const pub = resolvePublicBase();
  res.json({ value: v, effective: pub.base, source: pub.source, warnings: pub.warnings });
});

app.use('/api', (_req, res) => res.status(404).json({ error: 'not_found' }));

// ---------- 시작 ----------
function start({ port = PORT, host = '0.0.0.0', log = true } = {}) {
  if (store.seedIfEmpty() && log) console.log('샘플 상품 6개를 등록했습니다. (최초 실행)');
  const server = http.createServer(app);
  return new Promise((resolve) => {
    server.listen(port, host, () => {
      const actual = server.address().port;
      if (log) {
        const ip = lanIp();
        console.log('\n셀프 계산대 서버 실행 중');
        console.log(`  노트북(계산대):   http://localhost:${actual}/`);
        console.log(`  상품 등록:        http://localhost:${actual}/admin`);
        console.log(`  샘플 바코드 보기: http://localhost:${actual}/samples`);
        if (ip) console.log(`  같은 Wi-Fi 접속:  http://${ip}:${actual}/  (카메라는 HTTPS 필요 → README 참고)`);
        const pub = resolvePublicBase();
        console.log(`  QR에 들어갈 주소: ${pub.base}  [${pub.source}]`);
        for (const w of pub.warnings) console.log(`  ! ${w}`);
        console.log(`  DB 파일: ${store.DB_PATH}\n`);
      }
      resolve({ server, port: actual });
    });
  }).then((r) => {
    // 선택: certs/key.pem + certs/cert.pem 이 있으면 HTTPS도 함께 연다 (자체 서명 인증서용)
    const keyPath = path.join(__dirname, 'certs/key.pem');
    const certPath = path.join(__dirname, 'certs/cert.pem');
    if (fs.existsSync(keyPath) && fs.existsSync(certPath) && port === PORT) {
      const hs = https.createServer({ key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }, app);
      hs.listen(HTTPS_PORT, host, () => { if (log) console.log(`  HTTPS(자체 서명): https://${lanIp() || 'localhost'}:${HTTPS_PORT}/\n`); });
      r.httpsServer = hs;
    }
    return r;
  });
}

if (require.main === module) {
  start().catch((e) => { console.error(e); process.exit(1); });
}

module.exports = { app, start, _internal: { sessions, startNewCart, resolvePublicBase } };
