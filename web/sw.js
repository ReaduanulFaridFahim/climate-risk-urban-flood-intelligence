// =========================================================================
// BANGLASHIELD™ PWA DISASTER RESILIENCE SERVICE WORKER
// National Climate Risk & Urban Flood Intelligence Platform
// =========================================================================

const CACHE_NAME = 'banglashield-v2-cache';
const PRECACHE_URLS = [
  './index.html',
  './common.css',
  './deltabot_ai_engine.js',
  './deltabot.html',
  './banglashield_extended_intel.js',
  './districts_heritage_data.js',
  './disaster_alerts.html',
  './district_heritage.html',
  './executive_sitrep.html',
  './weather_ai.html',
  './agroclaim.html',
  './map_gps.html',
  './manifest.json'
];

// Install: Pre-cache essential operational assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[BanglaShield SW] Pre-caching critical disaster intelligence assets');
      return cache.addAll(PRECACHE_URLS).catch((err) => {
        console.warn('[BanglaShield SW] Non-fatal precache warning:', err);
      });
    }).then(() => self.skipWaiting())
  );
});

// Activate: Clean up older cache versions
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[BanglaShield SW] Purging deprecated cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch: Stale-While-Revalidate Strategy for field resilience
self.addEventListener('fetch', (event) => {
  // Only handle GET requests
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Network-first for dynamic live Open-Meteo & RainViewer tiles, cache fallback for everything else
  if (url.hostname.includes('open-meteo.com') || url.hostname.includes('rainviewer.com')) {
    event.respondWith(
      fetch(event.request).catch(() => caches.match(event.request))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      const fetchPromise = fetch(event.request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      }).catch((err) => {
        // Offline: Return cached response or fallback
        return cachedResponse;
      });

      return cachedResponse || fetchPromise;
    })
  );
});
