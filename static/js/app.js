import { t, apply, setLang, lang } from "./i18n.js";
import { API } from "./api.js";
import * as Pages from "./pages.js";

const routes = {
  dashboard:  { render: Pages.renderDashboard,  title: "nav.dashboard" },
  monitoring: { render: Pages.renderMonitoring, title: "nav.monitoring", cleanup: Pages.cleanupMonitoring },
  violations: { render: Pages.renderViolations, title: "nav.violations" },
  employees:  { render: Pages.renderEmployees,  title: "nav.employees" },
  enrollment: { render: Pages.renderEnrollment, title: "nav.enrollment", cleanup: Pages.cleanupEnrollment },
  history:    { render: Pages.renderHistory,    title: "nav.history" },
  analytics:  { render: Pages.renderAnalytics,  title: "nav.analytics" },
  settings:   { render: Pages.renderSettings,   title: "nav.settings" },
};

let currentCleanup = null;

function go(route) {
  if (!routes[route]) route = "dashboard";
  if (currentCleanup) {
    try { currentCleanup(); } catch {}
    currentCleanup = null;
  }
  document.querySelectorAll(".nav-item").forEach(a => {
    a.classList.toggle("active", a.dataset.route === route);
  });
  const titleEl = document.getElementById("page-title");
  titleEl.dataset.i18n = routes[route].title;
  titleEl.textContent = t(routes[route].title);
  apply();
  Promise.resolve(routes[route].render()).catch(e => {
    console.error(e);
  });
  currentCleanup = routes[route].cleanup || null;
  if (window.lucide) window.lucide.createIcons();
}

function router() {
  let route = location.hash.replace(/^#\/?/, "") || "dashboard";
  if (!routes[route]) route = "dashboard";
  go(route);
}

function initSidebar() {
  const sb = document.getElementById("sidebar");
  if (localStorage.getItem("sidebar") === "collapsed") sb.classList.add("collapsed");
  document.getElementById("sidebar-toggle").onclick = () => {
    sb.classList.toggle("collapsed");
    localStorage.setItem("sidebar", sb.classList.contains("collapsed") ? "collapsed" : "expanded");
  };
}

async function pollStatus() {
  try {
    const s = await API.status();
    const el = document.getElementById("header-status");
    if (!el) return;
    el.classList.toggle("online", s.online);
    const span = el.querySelector("span:last-child");
    if (span) span.textContent = s.online ? t("status.online") : t("status.offline");
  } catch {}
}

window.addEventListener("hashchange", router);

window.addEventListener("lang-changed", () => {
  // Re-render trang hiện tại để cập nhật các text động (t() trong template literals)
  router();
});

document.addEventListener("click", (e) => {
  const btn = e.target.closest(".lang-opt");
  if (btn && btn.dataset.lang) setLang(btn.dataset.lang);
});

document.addEventListener("DOMContentLoaded", () => {
  apply();
  initSidebar();
  router();
  if (window.lucide) window.lucide.createIcons();
  pollStatus();
  setInterval(pollStatus, 3000);
});