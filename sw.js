// =========================================================================
// BANGLASHIELD™ PWA DISASTER RESILIENCE SERVICE WORKER (V2 FULL SUITE)
// =========================================================================

const CACHE_NAME = 'banglashield-v2-enterprise';
const PRECACHE_URLS = [
  './index.html',
  './manifest.json',
  './web/index.html',
  './web/common.css',
  './web/map_gps.html',
  './web/disaster_alerts.html',
  './web/transboundary_rivers.html',
  './web/market_income_forecast.html',
  './web/district_heritage.html',
  './web/weather_ai.html',
  './web/agroclaim.html',
  './web/evacroute.html',
  './web/climadeval.html',
  './web/deltasim.html',
  './web/health_epidemic.html',
  './web/grid_resilience.html',
  './web/riverbank_erosion.html',
  './web/agri_supply.html',
  './web/ussd_phone.html',
  './web/deltabot.html',
  './web/dataset_explorer.html',
  './web/executive_sitrep.html',
  './web/user_manual_video.html'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS).catch((err) => console.warn('Precache partial fallback:', err)))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((names) => 
      Promise.all(names.map((n) => n !== CACHE_NAME ? caches.delete(n) : null))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  // Network first with cache fallback for HTML pages, cache first for static assets
  e.respondWith(
    caches.match(e.request).then((cached) => {
      const networked = fetch(e.request)
        .then((response) => {
          if (response && response.status === 200 && response.type === 'basic') {
            const cacheCopy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(e.request, cacheCopy)).catch(() => {});
          }
          return response;
        })
        .catch(() => cached);
      return cached || networked;
    })
  );
});
