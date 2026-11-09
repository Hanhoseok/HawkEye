// 서버 전체 흐름 테스트: 임시 DB로 서버를 띄우고 세션 생성 → 스캔 → 수량 변경 → 결제 → 새 장바구니까지 확인
const test = require('node:test');
const assert = require('node:assert');
const os = require('os');
const path = require('path');
const fs = require('fs');

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-test-'));
process.env.DB_PATH = path.join(tmp, 'test.db');
process.env.PUBLIC_URL = 'https://demo.example.test';

const { start } = require('../server');

let base, server;
const j = async (method, url, body) => {
  const r = await fetch(base + url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  return { status: r.status, data: await r.json() };
};

test.before(async () => { ({ server } = await start({ port: 0, host: '127.0.0.1', log: false })); base = `http://127.0.0.1:${server.address().port}`; });
test.after(() => { server.close(); try { require('../db').db.close(); } catch {}; try { fs.rmSync(tmp, { recursive: true, force: true }); } catch {} });

test('샘플 상품이 자동 등록되고 상품 CRUD가 동작한다', async () => {
  let r = await j('GET', '/api/products');
  assert.equal(r.status, 200);
  assert.equal(r.data.length, 6);
  assert.match(r.data[0].barcode, /^\d{13}$/);

  r = await j('POST', '/api/products', { barcode: '4902430 735247', name: '테스트 상품', price: 2500 });
  assert.equal(r.status, 201);
  assert.equal(r.data.barcode, '4902430735247', '공백은 제거되어 저장');
  const id = r.data.id;

  r = await j('POST', '/api/products', { barcode: '4902430735247', name: '중복', price: 1 });
  assert.equal(r.status, 409);

  r = await j('POST', '/api/products', { barcode: '123', name: '', price: 1 });
  assert.equal(r.status, 400);
  r = await j('POST', '/api/products', { barcode: '1234567', name: '음수', price: -1 });
  assert.equal(r.status, 400);

  r = await j('PUT', `/api/products/${id}`, { barcode: '4902430735247', name: '테스트 상품(수정)', price: 2700 });
  assert.equal(r.status, 200);
  assert.equal(r.data.name, '테스트 상품(수정)');
  assert.equal(r.data.price, 2700);
});

test('장바구니 전체 흐름: 스캔 → 수량 → 삭제 → 결제(연결 유지) → QR 재발급', async () => {
  let r = await j('GET', '/api/sessions/current');
  assert.equal(r.status, 200);
  const sid = r.data.sessionId;
  assert.equal(r.data.status, 'open');

  // QR 주소에 localhost가 없고 세션 ID가 들어간다
  r = await j('GET', `/api/sessions/${sid}/qr`);
  assert.equal(r.data.scanUrl, `https://demo.example.test/scan?s=${sid}`);
  assert.ok(r.data.qrDataUrl.startsWith('data:image/png;base64,'));
  assert.equal(r.data.warnings.length, 0);

  // SSE 연결: 첫 이벤트로 장바구니 스냅샷을 받는다
  const ac = new AbortController();
  const sse = await fetch(`${base}/api/sessions/${sid}/events`, { signal: ac.signal });
  assert.equal(sse.headers.get('content-type'), 'text/event-stream; charset=utf-8');
  const reader = sse.body.getReader();
  // 한 청크에 이벤트가 여러 개 올 수 있으므로(결제 시 paid + cart) 버퍼에 쌓고 빈 줄 단위로 잘라 읽는다
  let buf = '';
  const readEvent = async () => {
    for (;;) {
      const idx = buf.indexOf('\n\n');
      if (idx >= 0) {
        const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const m = block.match(/event: (\w+)\ndata: (.*)/);
        if (m) return { event: m[1], data: JSON.parse(m[2]) };
        continue; // 주석(ping) 블록은 건너뜀
      }
      const { value, done } = await reader.read();
      if (done) throw new Error('SSE stream ended');
      buf += Buffer.from(value).toString('utf8');
    }
  };
  let ev = await readEvent();
  assert.equal(ev.event, 'cart');
  assert.equal(ev.data.items.length, 0);

  const products = (await j('GET', '/api/products')).data;
  const p1 = products[0], p2 = products[1];

  // 등록 상품 스캔 → 담김, SSE로 노트북에 전달
  r = await j('POST', `/api/sessions/${sid}/scan`, { barcode: p1.barcode });
  assert.equal(r.status, 200);
  assert.equal(r.data.item.qty, 1);
  ev = await readEvent();
  assert.equal(ev.event, 'cart');
  assert.equal(ev.data.items[0].name, p1.name);
  assert.equal(ev.data.total, p1.price);
  assert.equal(ev.data.lastEvent.type, 'added');

  // 같은 상품 재스캔 → 담지 않고 "이미 담김" 응답, 수량은 그대로 1
  r = await j('POST', `/api/sessions/${sid}/scan`, { barcode: p1.barcode });
  assert.equal(r.status, 200);
  assert.equal(r.data.duplicate, true);
  assert.equal(r.data.item.qty, 1);
  assert.equal(r.data.cart.count, 1);

  // 미등록 바코드 → 404, 장바구니 변화 없음
  r = await j('POST', `/api/sessions/${sid}/scan`, { barcode: '9999999999999' });
  assert.equal(r.status, 404);
  assert.equal(r.data.error, 'unregistered');
  r = await j('GET', `/api/sessions/${sid}`);
  assert.equal(r.data.count, 1);

  // 두 번째 상품
  r = await j('POST', `/api/sessions/${sid}/scan`, { barcode: p2.barcode, source: 'manual' });
  assert.equal(r.status, 200);
  await readEvent();

  // 수량 수정
  r = await j('PATCH', `/api/sessions/${sid}/items/${p1.barcode}`, { qty: 5 });
  assert.equal(r.status, 200);
  assert.equal(r.data.items.find((i) => i.barcode === p1.barcode).qty, 5);
  assert.equal(r.data.total, p1.price * 5 + p2.price);
  await readEvent();

  // 수량 0 → 삭제, DELETE → 삭제
  r = await j('PATCH', `/api/sessions/${sid}/items/${p2.barcode}`, { qty: 0 });
  assert.equal(r.data.items.length, 1);
  await readEvent();
  r = await j('POST', `/api/sessions/${sid}/scan`, { barcode: p2.barcode });
  await readEvent();
  r = await j('DELETE', `/api/sessions/${sid}/items/${p2.barcode}`);
  assert.equal(r.status, 200);
  assert.equal(r.data.items.length, 1);
  await readEvent();

  // 모의 결제: 품목·총액이 주문 기록으로 저장되고, 장바구니는 비워지며, 세션(휴대폰 연결)은 유지된다
  r = await j('POST', `/api/sessions/${sid}/checkout`);
  assert.equal(r.status, 200);
  assert.equal(r.data.total, p1.price * 5);
  assert.equal(r.data.items.length, 1);
  assert.ok(r.data.orderId >= 1);
  const orderId = r.data.orderId;
  ev = await readEvent();
  assert.equal(ev.event, 'paid');
  assert.equal(ev.data.total, p1.price * 5);
  ev = await readEvent();
  assert.equal(ev.event, 'cart', '결제 직후 빈 장바구니가 전달된다');
  assert.equal(ev.data.items.length, 0);
  assert.equal(ev.data.status, 'open');
  assert.equal(ev.data.orderCount, 1);
  assert.equal(ev.data.lastEvent.type, 'paid');

  // 서버에 주문 기록이 남는다
  r = await j('GET', '/api/orders');
  assert.equal(r.status, 200);
  assert.equal(r.data[0].id, orderId);
  assert.equal(r.data[0].total, p1.price * 5);
  assert.equal(r.data[0].count, 5);
  assert.equal(r.data[0].items[0].name, p1.name);

  // 결제 후에도 같은 QR/세션으로 다음 손님 상품을 바로 담을 수 있다
  r = await j('POST', `/api/sessions/${sid}/scan`, { barcode: p1.barcode });
  assert.equal(r.status, 200);
  assert.equal(r.data.ok, true);
  assert.equal(r.data.item.qty, 1);
  await readEvent();

  // 결제 없이 비우기
  r = await j('POST', `/api/sessions/${sid}/clear`);
  assert.equal(r.status, 200);
  assert.equal(r.data.items.length, 0);
  ev = await readEvent();
  assert.equal(ev.data.lastEvent.type, 'cleared');
  assert.equal((await j('GET', '/api/orders')).data.length, 1, '비우기는 주문 기록을 남기지 않는다');

  // QR 재발급 → 새 ID, 이전 세션은 종료되고 SSE에 closed 전달
  r = await j('POST', '/api/sessions');
  assert.equal(r.status, 201);
  const sid2 = r.data.sessionId;
  assert.notEqual(sid2, sid);
  ev = await readEvent();
  assert.equal(ev.event, 'closed');
  ac.abort();

  r = await j('POST', `/api/sessions/${sid}/scan`, { barcode: p1.barcode });
  assert.equal(r.status, 410, '이전 세션으로 스캔하면 종료 안내');
  r = await j('GET', `/api/sessions/${sid2}/qr`);
  assert.equal(r.data.scanUrl, `https://demo.example.test/scan?s=${sid2}`);

  r = await j('POST', `/api/sessions/${sid2}/checkout`);
  assert.equal(r.status, 400, '빈 장바구니는 결제 불가');

  r = await j('GET', '/api/sessions/does-not-exist');
  assert.equal(r.status, 404);
});

test('UPC-A 12자리와 EAN-13 앞자리 0 표기를 서로 찾는다', async () => {
  await j('POST', '/api/products', { barcode: '012345678905', name: 'UPC 상품', price: 100 });
  let r = await j('GET', '/api/products/lookup/0012345678905');
  assert.equal(r.status, 200);
  await j('POST', '/api/products', { barcode: '0712345678911', name: 'EAN 0 상품', price: 100 });
  r = await j('GET', '/api/products/lookup/712345678911');
  assert.equal(r.status, 200);
});

test('공개 주소 설정이 저장되고 QR 주소에 반영된다', async () => {
  let r = await j('PUT', '/api/settings/public-url', { value: 'https://abc.trycloudflare.com/' });
  assert.equal(r.status, 200);
  assert.equal(r.data.effective, 'https://abc.trycloudflare.com');
  const sid = (await j('GET', '/api/sessions/current')).data.sessionId;
  r = await j('GET', `/api/sessions/${sid}/qr`);
  assert.equal(r.data.scanUrl, `https://abc.trycloudflare.com/scan?s=${sid}`);
  assert.equal(r.data.source, 'setting');

  r = await j('PUT', '/api/settings/public-url', { value: 'http://localhost:3000' });
  assert.equal((await j('GET', `/api/sessions/${sid}/qr`)).data.warnings.length, 1, 'localhost 경고');
  r = await j('PUT', '/api/settings/public-url', { value: 'not a url' });
  assert.equal(r.status, 400);
  await j('PUT', '/api/settings/public-url', { value: '' });
});
