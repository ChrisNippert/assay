// Styled drop-downs over real <select>s. The select stays in the page as the
// state (its value, options and change events keep working for any script);
// a button shows the current choice and opens a small menu in its place.
//
//   <select data-dd data-dd-label="Sort">…</select>   enhanced when the script runs
//   window.assayDropdown(select, {label})              enhance one made later
//
// Changing the select from script (value, selectedIndex, adding options)
// repaints the button; no event is needed.
(() => {
  if (window.assayDropdown) return;
  let open = null; // {menu, btn}

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function chevron() {
    const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    s.setAttribute('viewBox', '0 0 24 24');
    s.setAttribute('width', '12');
    s.setAttribute('height', '12');
    s.setAttribute('aria-hidden', 'true');
    s.setAttribute('class', 'dd-chev');
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', 'M6 9l6 6 6-6');
    s.append(p);
    return s;
  }
  function closeMenu(focus) {
    if (!open) return;
    open.menu.hidden = true;
    open.btn.setAttribute('aria-expanded', 'false');
    if (focus) open.btn.focus();
    open = null;
  }
  document.addEventListener('pointerdown', (e) => { if (open && !open.wrap.contains(e.target)) closeMenu(); });
  window.addEventListener('resize', () => closeMenu());
  document.addEventListener('scroll', (e) => { if (open && !open.menu.contains(e.target)) closeMenu(); }, true);

  const valueProp = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value');
  const indexProp = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'selectedIndex');

  window.assayDropdown = (select, opts = {}) => {
    if (select.dataset.ddReady) return;
    select.dataset.ddReady = '1';
    const wrap = el('div', 'dd' + (opts.className ? ' ' + opts.className : ''));
    const btn = el('button', 'dd-btn');
    btn.type = 'button';
    btn.setAttribute('aria-haspopup', 'listbox');
    btn.setAttribute('aria-expanded', 'false');
    const name = select.getAttribute('aria-label') || select.dataset.ddName;
    if (name) btn.setAttribute('aria-label', name);
    const labelText = opts.label ?? select.dataset.ddLabel;
    const cur = el('span', 'dd-value');
    btn.append(...(labelText ? [el('span', 'dd-label', labelText)] : []), cur, chevron());
    const menu = el('div', 'dd-menu');
    menu.setAttribute('role', 'listbox');
    menu.hidden = true;
    wrap.append(btn, menu);
    select.hidden = true;
    select.after(wrap);

    const paint = () => {
      const o = select.selectedOptions[0];
      cur.textContent = o ? o.textContent : '';
      btn.disabled = select.disabled;
    };
    // Script changes to value / selectedIndex repaint without an event.
    Object.defineProperty(select, 'value', {
      configurable: true,
      get() { return valueProp.get.call(this); },
      set(v) { valueProp.set.call(this, v); paint(); },
    });
    Object.defineProperty(select, 'selectedIndex', {
      configurable: true,
      get() { return indexProp.get.call(this); },
      set(v) { indexProp.set.call(this, v); paint(); },
    });
    new MutationObserver(paint).observe(select, { childList: true, subtree: true, attributes: true, characterData: true });
    select.addEventListener('change', paint);

    const items = () => [...menu.querySelectorAll('.dd-item')];
    function fill() {
      menu.replaceChildren();
      const add = (o, into) => {
        const it = el('button', 'dd-item', o.textContent);
        it.type = 'button';
        it.setAttribute('role', 'option');
        it.disabled = o.disabled;
        const on = o.value === select.value && o === select.selectedOptions[0];
        it.setAttribute('aria-selected', String(on));
        it.classList.toggle('on', on);
        it.addEventListener('click', () => {
          const changedTo = !on;
          valueProp.set.call(select, o.value);
          if (o.index >= 0) indexProp.set.call(select, o.index);
          paint();
          closeMenu(true);
          if (changedTo) select.dispatchEvent(new Event('change', { bubbles: true }));
        });
        into.append(it);
      };
      for (const child of select.children) {
        if (child.tagName === 'OPTGROUP') {
          menu.append(el('div', 'dd-group', child.label));
          for (const o of child.children) add(o, menu);
        } else add(child, menu);
      }
    }
    function position() {
      const r = btn.getBoundingClientRect();
      const below = window.innerHeight - r.bottom - 12;
      const above = r.top - 12;
      const up = below < 200 && above > below;
      menu.style.minWidth = r.width + 'px';
      menu.style.maxHeight = Math.max(140, Math.min(360, up ? above : below)) + 'px';
      menu.style.left = Math.max(8, Math.min(r.left, window.innerWidth - menu.offsetWidth - 8)) + 'px';
      menu.style.top = up ? '' : r.bottom + 4 + 'px';
      menu.style.bottom = up ? window.innerHeight - r.top + 4 + 'px' : '';
    }
    function openMenu() {
      closeMenu();
      fill();
      menu.hidden = false;
      btn.setAttribute('aria-expanded', 'true');
      open = { menu, btn, wrap };
      position();
      (menu.querySelector('.dd-item.on') || menu.querySelector('.dd-item'))?.focus();
    }
    btn.addEventListener('click', () => { if (open?.btn === btn) closeMenu(); else openMenu(); });
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); openMenu(); }
    });
    menu.addEventListener('keydown', (e) => {
      const list = items().filter((x) => !x.disabled);
      const i = list.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); list[Math.min(i + 1, list.length - 1)]?.focus(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); list[Math.max(i - 1, 0)]?.focus(); }
      else if (e.key === 'Home') { e.preventDefault(); list[0]?.focus(); }
      else if (e.key === 'End') { e.preventDefault(); list[list.length - 1]?.focus(); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeMenu(true); }
      else if (e.key === 'Tab') closeMenu();
    });
    paint();
    return wrap;
  };

  const run = () => { for (const s of document.querySelectorAll('select[data-dd]')) window.assayDropdown(s); };
  run(); // for selects above the script tag
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run);
})();
