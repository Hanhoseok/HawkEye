const test = require('node:test');
const assert = require('node:assert');
const createScanQueue = require('../public/js/scan-queue.js');
const createScanGate = require('../public/js/dedupe.js');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 서버 흉내: latency 뒤에 장바구니에 담는다 (이미 있으면 그대로)
function fakeServer(latencyMs) {
  const cart = new Set();
  const sent = [];
  return {
    cart, sent,
    async send(code) { sent.push(code); await sleep(latencyMs); cart.add(code); },
  };
}

test('요청 중에 들어온 새 바코드를 버리지 않고 뒤이어 보낸다', async () => {
  const srv = fakeServer(30);
  const q = createScanQueue({ send: srv.send, isInCart: (c) => srv.cart.has(c) });
  assert.equal(q.push('B'), 'sent');
  assert.equal(q.push('C'), 'queued');   // 예전 코드는 여기서 C를 버렸다
  await q.idle();
  assert.deepEqual(srv.sent, ['B', 'C']);
  assert.ok(srv.cart.has('C'));
});

test('이미 담긴 바코드는 서버에 보내지 않고 바로 이미 담김으로 처리한다', async () => {
  const srv = fakeServer(5);
  const local = [];
  const q = createScanQueue({ send: srv.send, isInCart: (c) => srv.cart.has(c), onLocalDuplicate: (c) => local.push(c) });
  q.push('B');
  await q.idle();
  assert.equal(q.push('B'), 'in-cart');
  await q.idle();
  assert.deepEqual(srv.sent, ['B']);
  assert.deepEqual(local, ['B']);
});

test('같은 바코드가 줄에 있거나 보내는 중이면 한 번만 보낸다', async () => {
  const srv = fakeServer(30);
  const q = createScanQueue({ send: srv.send, isInCart: (c) => srv.cart.has(c) });
  q.push('B');
  assert.equal(q.push('B'), 'duplicate');
  q.push('C');
  assert.equal(q.push('C'), 'duplicate');
  await q.idle();
  assert.deepEqual(srv.sent, ['B', 'C']);
});

test('보내기가 실패해도 다음 바코드는 계속 보낸다', async () => {
  const sent = [];
  const q = createScanQueue({ send: async (c) => { sent.push(c); await sleep(5); if (c === 'X') throw new Error('net'); } });
  q.push('X'); q.push('Y');
  await q.idle();
  assert.deepEqual(sent, ['X', 'Y']);
});

// C05 재현: 이미 담긴 병(B)과 새 캔(C)을 함께 비춰 인식 결과가 B,C,B,C... 로 번갈아 나오는 경우.
// 예전 코드는 응답이 프레임 간격보다 느리면 캔을 끝까지 담지 못했다.
for (const latency of [15, 60]) {
  test(`병·캔이 번갈아 인식돼도 캔이 담긴다 (프레임 10ms, 응답 ${latency}ms)`, async () => {
    const srv = fakeServer(latency);
    srv.cart.add('B'); // 병은 이미 담김
    const gate = createScanGate({ gapMs: 150 });
    const q = createScanQueue({ send: srv.send, isInCart: (c) => srv.cart.has(c) });
    for (let i = 0; i < 40 && !srv.cart.has('C'); i++) {
      const code = i % 2 ? 'C' : 'B';
      if (gate.check(code)) q.push(code, 'camera');
      await sleep(10);
    }
    await q.idle();
    assert.ok(srv.cart.has('C'), '캔이 담겨야 한다');
    assert.ok(!srv.sent.includes('B'), '이미 담긴 병은 서버로 보내지 않는다');
  });
}
