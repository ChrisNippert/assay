// Minimal service worker: makes the app installable. It does not cache
// anything; every request goes to the network, so pages, API calls and
// streamed chat responses are never served stale or intercepted.
self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});
