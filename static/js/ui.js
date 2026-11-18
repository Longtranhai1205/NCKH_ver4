import { t } from "./i18n.js";

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

export function toast(msg, type = "success", ms = 4000) {
  const c = document.getElementById("toast-container");
  if (!c) return;
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

export function modal({ title, body, footer, size = "", onMount }) {
  const root = document.getElementById("modal-root");
  const back = document.createElement("div");
  back.className = "modal-backdrop";
  back.innerHTML = `
    <div class="modal ${size}">
      <div class="modal-header">
        <h3>${esc(title)}</h3>
        <button class="close-x" aria-label="close">✕</button>
      </div>
      <div class="modal-body">${body}</div>
      ${footer ? `<div class="modal-footer">${footer}</div>` : ""}
    </div>`;
  root.appendChild(back);
  const close = () => back.remove();
  back.querySelector(".close-x").onclick = close;
  back.addEventListener("click", e => { if (e.target === back) close(); });
  onMount?.(back, close);
  return { close, root: back };
}

export function confirmModal({ title, message, confirmText, danger = false }) {
  return new Promise(resolve => {
    const m = modal({
      title,
      body: `<p style="margin:0;line-height:1.6">${esc(message)}</p>`,
      footer: `
        <button class="btn ghost" data-act="no">${esc(t("common.cancel"))}</button>
        <button class="btn ${danger ? "danger" : ""}" data-act="yes">${esc(confirmText || t("common.confirm"))}</button>
      `,
    });
    m.root.querySelector("[data-act=no]").onclick = () => { m.close(); resolve(false); };
    m.root.querySelector("[data-act=yes]").onclick = () => { m.close(); resolve(true); };
  });
}

export function emptyState(text) {
  return `<div class="empty">
    <div class="empty-icon">🗂️</div>
    <div>${esc(text || t("common.no_data"))}</div>
  </div>`;
}

export function skeletonRows(n = 4, cols = 5) {
  return Array.from({ length: n }).map(() =>
    `<tr>${Array.from({ length: cols }).map(() =>
      `<td><div class="skeleton"></div></td>`).join("")}</tr>`
  ).join("");
}

export function badgeForPpe(value) {
  if (value === true) return `<span class="badge pass">${esc(t("ppe.pass"))}</span>`;
  if (value === false) return `<span class="badge violation">${esc(t("ppe.violation"))}</span>`;
  return `<span class="badge unknown">${esc(t("ppe.unknown"))}</span>`;
}

export function formatUptime(sec) {
  if (sec == null) return "—";
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return `${h}h ${m}m`;
}

export function translateTypeCodes(codes) {
  return (codes || []).map(c => t(`ppe.type.raw.${c}`) || c).join(", ");
}

export function evidenceModal(imgSrc, extraInfo = "") {
  modal({
    title: t("common.evidence"),
    size: "lg",
    body: `
      <img src="${esc(imgSrc)}" style="width:100%;border-radius:8px;background:#000">
      ${extraInfo ? `<div style="margin-top:12px;font-size:13.5px;line-height:1.6">${extraInfo}</div>` : ""}
    `,
  });
}