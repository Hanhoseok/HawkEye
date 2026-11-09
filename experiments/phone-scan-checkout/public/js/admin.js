// 상품 등록/수정/삭제 화면
(() => {
  const $ = (id) => document.getElementById(id);
  const won = (n) => n.toLocaleString('ko-KR') + '원';
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  let products = [];

  async function api(method, url, body) {
    const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  }

  function msg(text, cls) { $('msg').innerHTML = text ? `<div class="${cls}">${esc(text)}</div>` : ''; }

  async function load() {
    products = await api('GET', '/api/products');
    $('cnt').textContent = `(${products.length}개)`;
    $('rows').innerHTML = products.map((p) => `
      <tr>
        <td class="mono">${esc(p.barcode)}</td>
        <td>${esc(p.name)}</td>
        <td class="num">${won(p.price)}</td>
        <td class="num">
          <button class="sm" data-edit="${p.id}">수정</button>
          <button class="sm danger" data-del="${p.id}">삭제</button>
        </td>
      </tr>`).join('');
  }

  function resetForm() {
    $('pid').value = '';
    $('form').reset();
    $('formTitle').textContent = '상품 추가';
    $('submit').textContent = '추가';
    $('cancel').style.display = 'none';
  }

  $('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const body = { barcode: $('barcode').value, name: $('name').value, price: Number($('price').value) };
    try {
      const id = $('pid').value;
      if (id) { await api('PUT', `/api/products/${id}`, body); msg('수정되었습니다.', 'ok'); }
      else { await api('POST', '/api/products', body); msg('등록되었습니다.', 'ok'); }
      resetForm();
      await load();
    } catch (err) { msg(err.message, 'err'); }
  });

  $('cancel').addEventListener('click', () => { resetForm(); msg(''); });

  document.addEventListener('click', async (e) => {
    const ed = e.target.closest('[data-edit]');
    if (ed) {
      const p = products.find((x) => x.id === Number(ed.dataset.edit));
      $('pid').value = p.id; $('barcode').value = p.barcode; $('name').value = p.name; $('price').value = p.price;
      $('formTitle').textContent = '상품 수정'; $('submit').textContent = '저장'; $('cancel').style.display = '';
      $('barcode').focus();
      return;
    }
    const del = e.target.closest('[data-del]');
    if (del) {
      const p = products.find((x) => x.id === Number(del.dataset.del));
      if (!confirm(`"${p.name}" 상품을 삭제할까요?`)) return;
      try { await api('DELETE', `/api/products/${p.id}`); await load(); msg('삭제되었습니다.', 'ok'); }
      catch (err) { msg(err.message, 'err'); }
    }
  });

  load().catch((err) => msg('목록을 불러오지 못했습니다: ' + err.message, 'err'));
})();
