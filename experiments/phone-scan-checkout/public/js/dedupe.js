// 중복 스캔 방지 게이트.
// 카메라 라이브러리는 바코드가 화면에 있는 동안 같은 코드를 초당 여러 번 보고한다.
// - 같은 코드가 gapMs 안에 다시 보이면 "계속 대고 있는 것"으로 보고 담지 않는다.
// - gapMs 이상 안 보이다가 다시 보이면 새 스캔으로 보고 한 번 더 담는다.
// - 다른 코드는 즉시 담는다.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.createScanGate = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  function createScanGate(opts) {
    const gapMs = (opts && opts.gapMs) || 1500;
    let lastCode = null;
    let lastSeenAt = -Infinity;
    return {
      // 반환값 true = 장바구니에 담아야 함, false = 같은 상품을 계속 비추는 중
      check(code, now) {
        if (now === undefined) now = Date.now();
        const stillSame = code === lastCode && (now - lastSeenAt) < gapMs;
        lastCode = code;
        lastSeenAt = now;
        return !stillSame;
      },
      reset() { lastCode = null; lastSeenAt = -Infinity; },
      get gapMs() { return gapMs; },
    };
  }
  return createScanGate;
});
