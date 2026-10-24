// Drag handles on the inner edges of the sidebar and the docked Tune panel.
// Widths live in CSS variables (--sidebar-w, --tune-w) and in localStorage.
// Pointer drag, arrow keys when focused, double-click to reset.
(() => {
  const root = document.documentElement;
  const GROUPS = [
    { id: 'sidebar', sel: '#sidebar', v: '--sidebar-w', def: 220, min: 180, max: 420, edge: 'right', label: 'Resize sidebar' },
    { id: 'tune', sel: '#tune', v: '--tune-w', def: 380, min: 300, max: 720, edge: 'left', label: 'Resize Tune panel' },
  ];
  const key = (g) => 'width:' + g.id;
  const clamp = (g, n) => Math.round(Math.min(g.max, Math.max(g.min, n)));
  const set = (g, n, save) => {
    n = clamp(g, n);
    root.style.setProperty(g.v, n + 'px');
    if (save) try { localStorage.setItem(key(g), String(n)); } catch {}
    return n;
  };
  const cur = (g) => parseFloat(getComputedStyle(root).getPropertyValue(g.v)) || g.def;
  for (const g of GROUPS) {
    try {
      const n = parseInt(localStorage.getItem(key(g)), 10);
      if (n) set(g, n, false);
    } catch {}
  }
  addEventListener('DOMContentLoaded', () => {
    for (const g of GROUPS) {
      const host = document.querySelector(g.sel);
      if (!host) continue;
      const h = document.createElement('div');
      h.className = 'resize-handle ' + g.edge;
      h.tabIndex = 0;
      h.setAttribute('role', 'separator');
      h.setAttribute('aria-orientation', 'vertical');
      h.setAttribute('aria-label', g.label);
      h.title = g.label + ' (drag, arrow keys, double-click to reset)';
      h.innerHTML = '<span class="grip"></span>';
      host.append(h);
      const sync = (n) => {
        h.setAttribute('aria-valuemin', g.min);
        h.setAttribute('aria-valuemax', g.max);
        h.setAttribute('aria-valuenow', n);
      };
      sync(cur(g));
      let startX = 0, startW = 0;
      h.addEventListener('pointerdown', (e) => {
        if (e.button) return;
        e.preventDefault();
        h.setPointerCapture(e.pointerId);
        startX = e.clientX;
        startW = cur(g);
        root.classList.add('resizing');
        h.classList.add('active');
      });
      h.addEventListener('pointermove', (e) => {
        if (!h.hasPointerCapture(e.pointerId)) return;
        const dx = e.clientX - startX;
        sync(set(g, startW + (g.edge === 'right' ? dx : -dx), false));
      });
      const end = (e) => {
        if (!h.hasPointerCapture(e.pointerId)) return;
        h.releasePointerCapture(e.pointerId);
        root.classList.remove('resizing');
        h.classList.remove('active');
        set(g, cur(g), true);
      };
      h.addEventListener('pointerup', end);
      h.addEventListener('pointercancel', end);
      h.addEventListener('dblclick', () => {
        root.style.removeProperty(g.v);
        try { localStorage.removeItem(key(g)); } catch {}
        sync(g.def);
      });
      h.addEventListener('keydown', (e) => {
        const step = e.shiftKey ? 40 : 10;
        // The sidebar grows rightward, the panel leftward.
        const dir = { ArrowRight: g.edge === 'right' ? 1 : -1, ArrowLeft: g.edge === 'right' ? -1 : 1 }[e.key];
        if (dir) { e.preventDefault(); sync(set(g, cur(g) + dir * step, true)); }
        else if (e.key === 'Home') { e.preventDefault(); sync(set(g, g.min, true)); }
        else if (e.key === 'End') { e.preventDefault(); sync(set(g, g.max, true)); }
      });
    }
  });
})();
