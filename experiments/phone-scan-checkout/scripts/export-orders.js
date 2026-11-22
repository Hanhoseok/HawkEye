'use strict';
// 셀프 계산대 결제 기록(orders 테이블)을 CSV 한 파일로 내보낸다.
// - 품목 하나가 한 줄. 같은 주문의 줄들은 주문번호·결제시각·주문총액이 같다.
// - 엑셀에서 한글이 깨지지 않도록 UTF-8 BOM을 붙인다.
// - DB는 읽기 전용으로 열므로 서버가 켜져 있어도 안전하다.
//
// 사용: npm run export:orders            → exports/payment-log.csv
//       node scripts/export-orders.js 출력경로.csv
const { DatabaseSync } = require('node:sqlite');
const path = require('path');
const fs = require('fs');

const ROOT = path.join(__dirname, '..');
const DB_PATH = process.env.DB_PATH || path.join(ROOT, 'data', 'shop.db');
const OUT = path.resolve(process.argv[2] || path.join(ROOT, 'exports', 'payment-log.csv'));

if (!fs.existsSync(DB_PATH)) {
  console.error(`DB 파일이 없습니다: ${DB_PATH}`);
  process.exit(1);
}

const db = new DatabaseSync(DB_PATH, { readOnly: true });
const orders = db.prepare('SELECT id, session_id, total, items_json, paid_at FROM orders ORDER BY id').all();
db.close();

const HEADER = ['주문번호', '결제시각', '계산대ID', '품목순번', '바코드', '상품명', '단가', '수량', '품목금액', '주문총액', '주문수량합'];
const cell = (v) => {
  const s = String(v ?? '');
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};
// 엑셀이 바코드를 숫자(지수 표기)로 바꾸지 않도록 ="..." 형태로 감싼다
const textCell = (v) => `="${String(v ?? '').replace(/"/g, '""')}"`;

const lines = [HEADER.join(',')];
const problems = [];
let lineCount = 0;
let grandTotal = 0;

for (const o of orders) {
  let items;
  try { items = JSON.parse(o.items_json); } catch { problems.push(`주문 ${o.id}: 품목 JSON을 읽을 수 없음`); continue; }
  const qtySum = items.reduce((n, it) => n + it.qty, 0);
  const lineSum = items.reduce((n, it) => n + it.price * it.qty, 0);
  if (lineSum !== o.total) problems.push(`주문 ${o.id}: 품목 합계 ${lineSum} ≠ 주문총액 ${o.total}`);
  grandTotal += o.total;
  items.forEach((it, i) => {
    lines.push([
      o.id, o.paid_at, cell(o.session_id), i + 1, textCell(it.barcode), cell(it.name),
      it.price, it.qty, it.price * it.qty, o.total, qtySum,
    ].join(','));
    lineCount++;
  });
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, '﻿' + lines.join('\r\n') + '\r\n', 'utf8');

console.log(`주문 ${orders.length}건, 품목 줄 ${lineCount}개, 총 결제액 ${grandTotal.toLocaleString('ko-KR')}원`);
console.log(`저장: ${OUT}`);
if (problems.length) {
  console.log('확인 필요:');
  for (const p of problems) console.log('  - ' + p);
  process.exitCode = 2;
}
