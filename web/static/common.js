/**
 * BANGLADESH CLIMATE RISK & MULTI-HAZARD INTELLIGENCE PLATFORM (BCIP)
 * Shared Production Utilities & Real-Time Data Pipeline
 */

let globalPlatformData = null;

// Initialize Live BST Clock (UTC+6)
function initLiveClock() {
  const clockEl = document.getElementById("liveBstClock");
  if (!clockEl) return;

  function tick() {
    const now = new Date();
    const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
    const bst = new Date(utc + (3600000 * 6));
    const timeStr = bst.toLocaleTimeString("en-US", { hour12: false });
    const dateStr = bst.toLocaleDateString("en-GB", { day: '2-digit', month: 'short', year: 'numeric' });
    clockEl.textContent = `BST ${timeStr} | ${dateStr}`;
  }
  tick();
  setInterval(tick, 1000);
}

// Global Cached Data Loader
async function getPlatformData() {
  if (globalPlatformData) return globalPlatformData;
  try {
    const res = await fetch("data/platform_data.json");
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    globalPlatformData = await res.json();
    return globalPlatformData;
  } catch (err) {
    console.error("[BCIP] Failed loading platform_data.json:", err);
    return null;
  }
}

// Utility: Format numbers with commas
function formatNumber(num, decimals = 0) {
  if (num === null || num === undefined || isNaN(num)) return "N/A";
  return Number(num).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

// Utility: Get alert tier badge HTML
function getAlertBadge(tier) {
  const t = String(tier).toUpperCase();
  if (t.includes("CRITICAL") || t.includes("RED") || t.includes("TIER 1")) {
    return `<span class="tier-badge tier-critical">🚨 ${tier}</span>`;
  }
  if (t.includes("HIGH") || t.includes("ORANGE") || t.includes("TIER 2")) {
    return `<span class="tier-badge tier-high">⚠️ ${tier}</span>`;
  }
  if (t.includes("MODERATE") || t.includes("BLUE")) {
    return `<span class="tier-badge tier-moderate">ℹ️ ${tier}</span>`;
  }
  return `<span class="tier-badge tier-low">✅ ${tier}</span>`;
}

document.addEventListener("DOMContentLoaded", () => {
  initLiveClock();
});
