// Applies the saved theme before first paint so dark mode doesn't flash.
try {
  const t = localStorage.getItem('theme');
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
} catch {}
