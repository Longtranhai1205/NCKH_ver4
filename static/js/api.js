async function req(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { const j = await r.json(); msg = j.detail || msg; } catch {}
    const err = new Error(msg); err.status = r.status; throw err;
  }
  if (r.status === 204) return null;
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

export const API = {
  status: () => req("/api/status"),
  snapshotUrl: () => `/api/snapshot.jpg?t=${Date.now()}`,
  systemInfo: () => req("/api/system/info"),
  liveTracks: () => req("/api/live/tracks"),

  workers: (includeInactive = false) =>
    req(`/api/workers${includeInactive ? "?include_inactive=true" : ""}`),
  deactivateWorker: (id, reason = "inactive") =>
    req(`/api/workers/${encodeURIComponent(id)}/deactivate`, {
      method: "POST", body: JSON.stringify({ reason }),
    }),

  violations: (params = {}) => {
    // Mặc định gộp vi phạm trong 5 phút
    const p = { group_minutes: 5, ...params };
    const qs = new URLSearchParams();
    Object.entries(p).forEach(([k, v]) => {
      if (v !== "" && v != null) qs.set(k, v);
    });
    const q = qs.toString();
    return req(`/api/violations${q ? "?" + q : ""}`);
  },
  assignViolation: (id, worker_id) =>
    req(`/api/violations/${id}/assign`, {
      method: "POST", body: JSON.stringify({ worker_id }),
    }),

  statsOverview: () => req("/api/stats/overview"),
  statsByDay: (days = 30) => req(`/api/stats/by-day?days=${days}`),
  statsByType: () => req("/api/stats/by-type"),
  statsTopWorkers: (limit = 5) => req(`/api/stats/top-workers?limit=${limit}`),

  enrollStart: (id, name, replace = false) =>
    req("/api/enroll/start", {
      method: "POST", body: JSON.stringify({ id, name, replace }),
    }),
  enrollStatus: () => req("/api/enroll/status"),
  enrollCancel: () => req("/api/enroll/cancel", { method: "POST" }),
};