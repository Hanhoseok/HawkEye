const test = require('node:test');
const assert = require('node:assert');
const createScanGate = require('../public/js/dedupe.js');

test('같은 바코드를 계속 비추면 한 번만 담는다', () => {
  const g = createScanGate({ gapMs: 1500 });
  assert.equal(g.check('A', 0), true);      // 첫 인식 → 담기
  assert.equal(g.check('A', 100), false);   // 계속 보임 (10fps)
  assert.equal(g.check('A', 200), false);
  assert.equal(g.check('A', 1400), false);
  assert.equal(g.check('A', 2800), false);  // 마지막으로 본 지 1.4초 → 아직 같은 스캔
});

test('화면에서 뺐다가(1.5초 이상) 다시 비추면 한 번 더 담는다', () => {
  const g = createScanGate({ gapMs: 1500 });
  assert.equal(g.check('A', 0), true);
  assert.equal(g.check('A', 300), false);
  assert.equal(g.check('A', 300 + 1600), true); // 1.6초 공백 후 재인식 → 수량 +1
  assert.equal(g.check('A', 300 + 1700), false);
});

test('다른 바코드는 즉시 담는다', () => {
  const g = createScanGate({ gapMs: 1500 });
  assert.equal(g.check('A', 0), true);
  assert.equal(g.check('B', 50), true);
  assert.equal(g.check('A', 100), true); // A → B → A 는 서로 다른 스캔
});

test('reset 후에는 같은 코드도 다시 담는다', () => {
  const g = createScanGate({ gapMs: 1500 });
  assert.equal(g.check('A', 0), true);
  g.reset();
  assert.equal(g.check('A', 10), true);
});
