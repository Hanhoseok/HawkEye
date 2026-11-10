'use strict';
// SQLite 저장소 (Node 22.13+ 내장 node:sqlite 사용, 네이티브 빌드 불필요)
const { DatabaseSync } = require('node:sqlite');
const path = require('path');
const fs = require('fs');

const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, 'data');
fs.mkdirSync(DATA_DIR, { recursive: true });
const DB_PATH = process.env.DB_PATH || path.join(DATA_DIR, 'shop.db');

const db = new DatabaseSync(DB_PATH);
db.exec(`
  PRAGMA journal_mode = WAL;
  CREATE TABLE IF NOT EXISTS products (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    barcode    TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL,
    price      INTEGER NOT NULL CHECK (price >= 0),
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
  );
  CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
  );
  CREATE TABLE IF NOT EXISTS orders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    total      INTEGER NOT NULL,
    items_json TEXT NOT NULL,
    paid_at    TEXT NOT NULL DEFAULT (datetime('now','localtime'))
  );
`);

// ---------- 바코드 유틸 ----------
function ean13CheckDigit(first12) {
  let sum = 0;
  for (let i = 0; i < 12; i++) sum += Number(first12[i]) * (i % 2 === 0 ? 1 : 3);
  return String((10 - (sum % 10)) % 10);
}

// 스캐너/입력값을 저장용 형태로 정규화: 공백 제거, 숫자 바코드는 숫자만 남김
function normalizeBarcode(raw) {
  if (raw == null) return '';
  let s = String(raw).trim();
  const digitsOnly = s.replace(/\D/g, '');
  if (digitsOnly.length >= 6 && digitsOnly.length === s.replace(/[\s-]/g, '').length) s = digitsOnly;
  return s.slice(0, 64);
}

// ---------- 상품 ----------
const stmts = {
  all: db.prepare('SELECT * FROM products ORDER BY id'),
  byId: db.prepare('SELECT * FROM products WHERE id = ?'),
  byBarcode: db.prepare('SELECT * FROM products WHERE barcode = ?'),
  insert: db.prepare('INSERT INTO products (barcode, name, price) VALUES (?, ?, ?)'),
  update: db.prepare("UPDATE products SET barcode = ?, name = ?, price = ?, updated_at = datetime('now','localtime') WHERE id = ?"),
  remove: db.prepare('DELETE FROM products WHERE id = ?'),
  count: db.prepare('SELECT COUNT(*) AS n FROM products'),
  getSetting: db.prepare('SELECT value FROM settings WHERE key = ?'),
  setSetting: db.prepare('INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value'),
  insertOrder: db.prepare('INSERT INTO orders (session_id, total, items_json) VALUES (?, ?, ?)'),
  recentOrders: db.prepare('SELECT id, session_id, total, items_json, paid_at FROM orders ORDER BY id DESC LIMIT ?'),
};

function listProducts() { return stmts.all.all(); }
function getProduct(id) { return stmts.byId.get(id) || null; }

// 스캔 결과로 상품 찾기. UPC-A(12자리)와 EAN-13(앞자리 0) 표기 차이를 흡수한다.
function findProductByBarcode(raw) {
  const code = normalizeBarcode(raw);
  if (!code) return null;
  const candidates = [code];
  if (/^\d{12}$/.test(code)) candidates.push('0' + code);
  if (/^\d{13}$/.test(code) && code.startsWith('0')) candidates.push(code.slice(1));
  for (const c of candidates) {
    const p = stmts.byBarcode.get(c);
    if (p) return p;
  }
  return null;
}

function validateProductInput(body) {
  const barcode = normalizeBarcode(body.barcode);
  const name = String(body.name ?? '').trim();
  const price = Number(body.price);
  if (!barcode) return { error: '바코드 번호를 입력하세요.' };
  if (!/^[0-9A-Za-z\-_.]{3,64}$/.test(barcode)) return { error: '바코드는 3~64자의 숫자/영문만 가능합니다.' };
  if (!name) return { error: '상품명을 입력하세요.' };
  if (name.length > 100) return { error: '상품명은 100자 이내로 입력하세요.' };
  if (!Number.isInteger(price) || price < 0) return { error: '가격은 0 이상의 정수(원)여야 합니다.' };
  return { value: { barcode, name, price } };
}

function createProduct(body) {
  const v = validateProductInput(body);
  if (v.error) return { status: 400, error: v.error };
  if (stmts.byBarcode.get(v.value.barcode)) return { status: 409, error: '이미 등록된 바코드입니다.' };
  const r = stmts.insert.run(v.value.barcode, v.value.name, v.value.price);
  return { status: 201, product: getProduct(Number(r.lastInsertRowid)) };
}

function updateProduct(id, body) {
  if (!getProduct(id)) return { status: 404, error: '상품을 찾을 수 없습니다.' };
  const v = validateProductInput(body);
  if (v.error) return { status: 400, error: v.error };
  const dup = stmts.byBarcode.get(v.value.barcode);
  if (dup && dup.id !== id) return { status: 409, error: '다른 상품이 이미 사용 중인 바코드입니다.' };
  stmts.update.run(v.value.barcode, v.value.name, v.value.price, id);
  return { status: 200, product: getProduct(id) };
}

function deleteProduct(id) {
  if (!getProduct(id)) return { status: 404, error: '상품을 찾을 수 없습니다.' };
  stmts.remove.run(id);
  return { status: 200 };
}

// ---------- 설정 ----------
function getSetting(key) { const r = stmts.getSetting.get(key); return r ? r.value : null; }
function setSetting(key, value) { stmts.setSetting.run(key, value); }

// ---------- 주문(모의 결제 기록) ----------
function recordOrder(sessionId, total, items) {
  const r = stmts.insertOrder.run(sessionId, total, JSON.stringify(items));
  return Number(r.lastInsertRowid);
}
function listOrders(limit = 20) {
  return stmts.recentOrders.all(limit).map((o) => {
    const items = JSON.parse(o.items_json);
    return { id: o.id, sessionId: o.session_id, total: o.total, count: items.reduce((n, it) => n + it.qty, 0), items, paidAt: o.paid_at };
  });
}

// ---------- 샘플 상품 (최초 실행 시 1회) ----------
const SAMPLE_PRODUCTS = [
  ['880123456001', '삼다수 생수 500ml', 950],
  ['880123456002', '새우깡 90g', 1500],
  ['880123456003', '바나나우유 240ml', 1800],
  ['880123456004', '신라면 봉지', 1100],
  ['880123456005', '초코파이 12입', 5400],
  ['880123456006', '볼펜 0.5mm 검정', 1200],
];
function seedIfEmpty() {
  if (stmts.count.get().n > 0) return false;
  const insert = db.prepare('INSERT INTO products (barcode, name, price) VALUES (?, ?, ?)');
  for (const [first12, name, price] of SAMPLE_PRODUCTS) {
    insert.run(first12 + ean13CheckDigit(first12), name, price);
  }
  return true;
}

module.exports = {
  db, DB_PATH,
  normalizeBarcode, ean13CheckDigit,
  listProducts, getProduct, findProductByBarcode, createProduct, updateProduct, deleteProduct,
  getSetting, setSetting, recordOrder, listOrders, seedIfEmpty,
};
