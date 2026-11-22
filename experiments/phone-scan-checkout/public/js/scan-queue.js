// 스캔 대기열.
// 예전에는 서버 요청 중에 인식된 바코드를 버렸고, 중복 거름망은 그 바코드를 이미 본 것으로
// 기록해서 같은 상품을 계속 비춰도 다시 보내지 않았다 (C05 결제에서 상품 누락 원인 후보).
// 이제는:
// - 이미 장바구니에 있는 바코드는 서버에 보내지 않고 바로 "이미 담김"으로 처리한다.
//   → 담긴 상품을 함께 비춰도 요청이 생기지 않아, 새 상품 요청을 가로막지 않는다.
// - 요청 중에 들어온 새 바코드는 버리지 않고 줄을 세워 차례로 보낸다.
// - 같은 바코드가 이미 줄에 있거나 보내는 중이면 한 번만 둔다.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.createScanQueue = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  function createScanQueue(opts) {
    const send = opts.send;                                   // async (code, source) => any
    const isInCart = opts.isInCart || (() => false);          // (code) => boolean
    const onLocalDuplicate = opts.onLocalDuplicate || (() => {});
    const maxPending = opts.maxPending || 10;
    const pending = [];
    let inFlight = null;
    let idleWaiters = [];

    function settleIdle() {
      if (inFlight || pending.length) return;
      const w = idleWaiters; idleWaiters = [];
      w.forEach((r) => r());
    }

    async function pump() {
      if (inFlight || !pending.length) { settleIdle(); return; }
      const job = pending.shift();
      // 줄을 서 있는 동안 다른 경로(직접 입력, 노트북)로 담겼을 수 있다
      if (isInCart(job.code)) { onLocalDuplicate(job.code, job.source); pump(); return; }
      inFlight = job;
      try { await send(job.code, job.source); }
      catch { /* 오류 안내는 send 쪽에서 한다 */ }
      finally { inFlight = null; pump(); }
    }

    return {
      // 반환값: 'sent'(바로 보냄) | 'queued'(줄 섬) | 'in-cart'(이미 담김) | 'duplicate'(이미 줄/전송 중) | 'full'
      push(code, source) {
        if (isInCart(code)) { onLocalDuplicate(code, source); return 'in-cart'; }
        if ((inFlight && inFlight.code === code) || pending.some((j) => j.code === code)) return 'duplicate';
        if (pending.length >= maxPending) return 'full';
        pending.push({ code, source });
        const wasIdle = !inFlight;
        pump();
        return wasIdle ? 'sent' : 'queued';
      },
      get size() { return pending.length + (inFlight ? 1 : 0); },
      // 테스트용: 줄이 모두 처리될 때까지 기다린다
      idle() { return new Promise((r) => { idleWaiters.push(r); settleIdle(); }); },
    };
  }
  return createScanQueue;
});
