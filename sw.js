// =========================================================================
// BANGLASHIELD™ PWA DISASTER RESILIENCE SERVICE WORKER (ROOT)
// =========================================================================

const CACHE_NAME = 'banglashield-root-v1';
const PRECACHE_URLS = [
  './index.html',
  './web/index.html',
  './web/common.css',
  './web/banglashield_extended_intel.js',
  './manifest.json'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS).catch(() => {})).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((names) => Promise.all(names.map((n) => n !== CACHE_NAME ? caches.delete(n) : null))).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then((res) => res || fetch(e.request).catch(() => res))
  );
});
