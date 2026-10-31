// Applies the saved theme and accent before first paint so neither flashes.
// Theme: 'light' | 'dark' (nothing saved = follow the system).
// Accent: 'copper' (default) | 'indigo'.
try {
  const t = localStorage.getItem('theme');
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
} catch {}
try {
  document.documentElement.dataset.accent = localStorage.getItem('accent') === 'indigo' ? 'indigo' : 'copper';
} catch {
  document.documentElement.dataset.accent = 'copper';
}

// Runs fn once the page is actually showing. A page that the browser
// prerendered after a hover must not poll, reattach to streams or load models
// until the person really opens it; for a normal load fn runs at once.
window.assayWhenVisible = (fn) => {
  if (document.prerendering) document.addEventListener('prerenderingchange', () => fn(), { once: true });
  else fn();
};
