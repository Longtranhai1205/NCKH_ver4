import { API } from "./api.js";
import { t, lang } from "./i18n.js";
import {
  esc, toast, modal, confirmModal, emptyState, skeletonRows,
  badgeForPpe, formatUptime, translateTypeCodes, evidenceModal,
} from "./ui.js";

const view = () => document.getElementById("view");
const actions = () => document.getElementById("page-actions");

/* ============================================================
   DASHBOARD
   ============================================================ */
let dashChartDay = null, dashChartType = null;

export async function renderDashboard() {
  actions().innerHTML = "";
  view().innerHTML = `
    <div class="stat-grid" id="dstat">
      ${[1,2,3,4].map(() => `<div class="stat-card">
        <div class="skeleton" style="width:60%"></div>
        <div class="skeleton" style="height:28px;margin-top:10px"></div>
      </div>`).join("")}
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">${esc(t("dash.by_day"))}</div>
        <div style="height:260px"><canvas id="ch-day"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">${esc(t("dash.by_type"))}</div>
        <div style="height:260px"><canvas id="ch-type"></canvas></div>
      </div>
    </div>
    <div class="grid-2" style="margin-top:18px">
      <div class="card">
        <div class="card-title">${esc(t("dash.top_workers"))}</div>
        <div id="top-workers">${emptyState()}</div>
      </div>
      <div class="card">
        <div class="card-title">${esc(t("dash.recent"))}</div>
        <div id="recent-vio">${emptyState()}</div>
      </div>
    </div>`;

  try {
    const [ov, days, types, top, recent] = await Promise.all([
      API.statsOverview(), API.statsByDay(14), API.statsByType(),
      API.statsTopWorkers(5), API.violations({ limit: 5, page: 1 }),
    ]);

    document.getElementById("dstat").innerHTML = `
      <div class="stat-card">
        <div class="stat-label">${esc(t("dash.total_workers"))}</div>
        <div class="stat-value">${ov.total_workers}</div>
      </div>
      <div class="stat-card danger">
        <div class="stat-label">${esc(t("dash.total_violations"))}</div>
        <div class="stat-value">${ov.total_violations}</div>
      </div>
      <div class="stat-card danger">
        <div class="stat-label">${esc(t("dash.today_violations"))}</div>
        <div class="stat-value">${ov.today_violations}</div>
      </div>
      <div class="stat-card warning">
        <div class="stat-label">${esc(t("dash.unassigned"))}</div>
        <div class="stat-value">${ov.unassigned_violations}</div>
      </div>`;

    dashChartDay?.destroy();
    dashChartType?.destroy();
    dashChartDay = new Chart(document.getElementById("ch-day"), {
      type: "line",
      data: { labels: days.map(d => d.date.slice(5)), datasets: [{
        data: days.map(d => d.count),
        borderColor: "#1763AE", backgroundColor: "rgba(23,99,174,.12)",
        fill: true, tension: .35, pointRadius: 3,
      }] },
      options: {
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
      },
    });
    dashChartType = new Chart(document.getElementById("ch-type"), {
      type: "doughnut",
      data: {
        labels: types.map(x => t(`ppe.type.raw.${x.type}`) || x.type),
        datasets: [{
          data: types.map(x => x.count),
          backgroundColor: ["#1763AE", "#F59E0B", "#DC2626"],
        }],
      },
      options: { plugins: { legend: { position: "bottom" } }, cutout: "60%" },
    });

    const tw = document.getElementById("top-workers");
    tw.innerHTML = top.length
      ? `<table class="data"><tbody>${top.map(w =>
          `<tr>
             <td><b>${esc(w.id)}</b><br>
               <span style="color:var(--muted);font-size:12px">${esc(w.name)}</span></td>
             <td style="text-align:right;font-weight:700;color:var(--danger)">${w.count}</td>
           </tr>`).join("")}</tbody></table>`
      : emptyState();

    const items = recent.items || [];
    const rv = document.getElementById("recent-vio");
    rv.innerHTML = items.length
      ? items.map(v => `
        <div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)">
          <img class="img-thumb" src="${esc(v.img)}" data-evidence="${esc(v.img)}">
          <div style="flex:1;min-width:0">
            <div style="font-weight:600;color:var(--danger)">${esc(translateTypeCodes(v.type_raw))}</div>
            <div style="font-size:12px;color:var(--muted)">
              ${esc(v.wid || t("vio.unknown"))} · ${v.date} ${v.time}
            </div>
          </div>
        </div>`).join("")
      : emptyState();
    rv.querySelectorAll("[data-evidence]").forEach(img => {
      img.onclick = () => evidenceModal(img.dataset.evidence);
    });
  } catch (e) {
    toast(t("common.backend_down") + ": " + e.message, "error");
  }
}

/* ============================================================
   MONITORING
   ============================================================ */
let monTimer = null, monToastTimer = null, monLastMaxVio = null;

export function cleanupMonitoring() {
  clearInterval(monTimer);  monTimer = null;
  clearInterval(monToastTimer); monToastTimer = null;
}

export async function renderMonitoring() {
  actions().innerHTML = "";
  view().innerHTML = `
    <div class="grid-2" style="grid-template-columns: 2fr 1fr">
      <div class="card" style="padding:0;overflow:hidden">
        <div class="cam-box" id="cam-box">
          <img id="cam-img" alt="camera">
          <div class="cam-overlay-msg" id="cam-msg">${esc(t("common.loading"))}</div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">${esc(t("mon.detail"))}</div>
        <div style="display:flex;flex-direction:column;gap:12px;font-size:13.5px">
          <div>
            <span class="status-pill" id="mon-status">
              <span class="dot"></span><span>—</span>
            </span>
          </div>
          <div><b>${esc(t("mon.update"))}:</b> <span id="mon-ts">—</span></div>
          <div><b>${esc(t("mon.people"))}:</b> <span id="mon-people">0</span></div>
          <div><b>${esc(t("mon.recent"))}:</b> <span id="mon-recent">—</span></div>
          <button class="btn ghost sm" id="btn-expand">${esc(t("mon.expand"))}</button>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:18px">
      <div class="card-title">${esc(t("mon.tracks"))}</div>
      <div id="tracks-wrap">${emptyState(t("mon.not_available"))}</div>
    </div>`;

  document.getElementById("btn-expand").onclick = () => {
    const m = modal({
      title: t("mon.title"), size: "lg",
      body: `<div class="cam-box"><img src="${API.snapshotUrl()}" style="width:100%"></div>`,
    });
    const int = setInterval(() => {
      if (!document.body.contains(m.root)) { clearInterval(int); return; }
      const img = m.root.querySelector("img");
      if (img) img.src = API.snapshotUrl();
    }, 1000);
  };

  const tick = async () => {
    try {
      const img = document.getElementById("cam-img");
      const overlay = document.getElementById("cam-msg");
      if (img) {
        img.src = API.snapshotUrl();
        img.onload = () => overlay?.classList.add("hidden");
        img.onerror = () => {
          overlay?.classList.remove("hidden");
          if (overlay) overlay.textContent = t("common.backend_down");
        };
      }
      const [st, tracks, vio] = await Promise.all([
        API.status(), API.liveTracks(), API.violations({ limit: 1 }),
      ]);

      const sp = document.getElementById("mon-status");
      if (sp) {
        sp.classList.toggle("online", st.online);
        sp.querySelector("span:last-child").textContent =
          st.online ? t("status.online") : t("status.offline");
      }
      const ts = document.getElementById("mon-ts");
      if (ts) ts.textContent = tracks.updated_at
        ? new Date(tracks.updated_at * 1000).toLocaleTimeString() : "—";
      const ppl = document.getElementById("mon-people");
      if (ppl) ppl.textContent = tracks.tracks?.length || 0;

      const rec = document.getElementById("mon-recent");
      const first = (vio.items || [])[0];
      if (rec) rec.textContent = first ? `${first.date} ${first.time}` : "—";

      // toast khi có vi phạm mới
      if (first) {
        if (monLastMaxVio === null) {
          monLastMaxVio = first.id;
        } else if (first.id > monLastMaxVio) {
          monLastMaxVio = first.id;
          toast(
            `${t("ppe.violation")}: ${translateTypeCodes(first.type_raw)} — ${first.wid || "UNKNOWN"}`,
            "warning", 5000
          );
        }
      }

      const tw = document.getElementById("tracks-wrap");
      if (tracks.tracks && tracks.tracks.length) {
        tw.innerHTML = `<div class="table-wrap"><table class="data">
          <thead><tr>
            <th>ID</th>
            <th>${esc(t("emp.name"))}</th>
            <th>${esc(t("ppe.hat"))}</th>
            <th>${esc(t("ppe.mask"))}</th>
            <th>${esc(t("ppe.vest"))}</th>
            <th>${esc(t("common.status"))}</th>
          </tr></thead>
          <tbody>${tracks.tracks.map(x => {
            const anyVio = x.hat === false || x.mask === false || x.vest === false;
            const allPass = x.hat === true && x.mask === true && x.vest === true;
            const status = anyVio
              ? `<span class="badge violation">${esc(t("ppe.violation"))}</span>`
              : allPass
                ? `<span class="badge pass">${esc(t("ppe.pass"))}</span>`
                : `<span class="badge unknown">${esc(t("ppe.unknown"))}</span>`;
            return `<tr>
              <td class="mono">${x.track_id}</td>
              <td>
                <b>${esc(x.employee_id || "UNKNOWN")}</b><br>
                <span style="color:var(--muted);font-size:12px">${esc(x.full_name || "")}</span>
              </td>
              <td>${badgeForPpe(x.hat)}</td>
              <td>${badgeForPpe(x.mask)}</td>
              <td>${badgeForPpe(x.vest)}</td>
              <td>${status}</td>
            </tr>`;
          }).join("")}</tbody></table></div>`;
      } else {
        tw.innerHTML = emptyState(t("mon.not_available"));
      }
    } catch (e) {
      const overlay = document.getElementById("cam-msg");
      overlay?.classList.remove("hidden");
      if (overlay) overlay.textContent = t("common.backend_down");
    }
  };

  await tick();
  monTimer = setInterval(tick, 1500);
}

/* ============================================================
   VIOLATIONS
   ============================================================ */
export async function renderViolations() {
  actions().innerHTML = "";
  view().innerHTML = `
    <div class="card">
      <div class="toolbar">
        <div class="field"><label>${esc(t("vio.filter.date"))}</label>
          <input type="date" id="f-date"></div>
        <div class="field"><label>${esc(t("vio.filter.type"))}</label>
          <select id="f-type">
            <option value="">${esc(t("vio.all"))}</option>
            <option value="Mu">${esc(t("ppe.hat"))}</option>
            <option value="Khau trang">${esc(t("ppe.mask"))}</option>
            <option value="Ao bao ho">${esc(t("ppe.vest"))}</option>
          </select></div>
        <div class="field"><label>${esc(t("vio.filter.status"))}</label>
          <select id="f-status">
            <option value="">${esc(t("vio.all"))}</option>
            <option value="assigned">${esc(t("vio.status.assigned"))}</option>
            <option value="unassigned">${esc(t("vio.status.unassigned"))}</option>
          </select></div>
        <button class="btn ghost" id="btn-filter">${esc(t("common.search"))}</button>
      </div>
      <div id="vio-list">${emptyState()}</div>
    </div>`;

  const openAssign = async (vid) => {
    try {
      const ws = await API.workers();
      if (!ws.length) {
        toast(t("common.no_data"), "error"); return;
      }
      const m = modal({
        title: t("vio.assign.title"),
        body: `<div class="field"><label>${esc(t("emp.id"))}</label>
          <select id="assign-w">${ws.map(w =>
            `<option value="${esc(w.id)}">${esc(w.id)} — ${esc(w.name)}</option>`).join("")}
          </select></div>`,
        footer: `<button class="btn ghost" data-act="no">${esc(t("common.cancel"))}</button>
                 <button class="btn" data-act="yes">${esc(t("common.confirm"))}</button>`,
      });
      m.root.querySelector("[data-act=no]").onclick = m.close;
      m.root.querySelector("[data-act=yes]").onclick = async () => {
        const wid = m.root.querySelector("#assign-w").value;
        if (!wid) return;
        try {
          await API.assignViolation(vid, wid);
          m.close();
          toast("OK", "success");
          load();
        } catch (e) { toast(e.message, "error"); }
      };
    } catch (e) { toast(e.message, "error"); }
  };

  const load = async () => {
    const p = {
      date: document.getElementById("f-date").value,
      type: document.getElementById("f-type").value,
      status: document.getElementById("f-status").value,
      page: 1, limit: 60,
    };
    try {
      const r = await API.violations(p);
      const items = r.items || [];
      const wrap = document.getElementById("vio-list");
      if (!items.length) { wrap.innerHTML = emptyState(); return; }
      wrap.innerHTML = `<div class="grid-3">${items.map(v => `
        <div class="card" style="padding:0;overflow:hidden;display:flex;flex-direction:column">
          <img src="${esc(v.img)}" data-evidence="${esc(v.img)}"
               style="width:100%;aspect-ratio:4/3;object-fit:cover;cursor:pointer">
          <div style="padding:12px 14px;display:flex;flex-direction:column;gap:6px">
            <div style="font-weight:700;color:var(--danger);font-size:13px">
              ${esc(translateTypeCodes(v.type_raw))}
            </div>
            <div style="font-size:12px;color:var(--muted)">
              ${esc(v.wid || t("vio.unknown"))}${v.w_name ? " · " + esc(v.w_name) : ""}
            </div>
            <div style="font-size:12px;color:var(--muted)">${v.date} ${v.time}</div>
            <div style="display:flex;gap:6px;align-items:center;margin-top:4px">
              ${v.wid
                ? `<span class="badge pass">${esc(t("vio.status.assigned"))}</span>`
                : `<span class="badge unknown">${esc(t("vio.status.unassigned"))}</span>`}
              ${!v.wid
                ? `<button class="btn sm" data-assign="${v.id}">${esc(t("common.assign"))}</button>`
                : ""}
            </div>
          </div>
        </div>`).join("")}</div>`;

      wrap.querySelectorAll("[data-evidence]").forEach(img => {
        img.onclick = () => evidenceModal(img.dataset.evidence);
      });
      wrap.querySelectorAll("[data-assign]").forEach(b => {
        b.onclick = () => openAssign(+b.dataset.assign);
      });
    } catch (e) { toast(e.message, "error"); }
  };

  document.getElementById("btn-filter").onclick = load;
  load();
}

/* ============================================================
   EMPLOYEES
   ============================================================ */
export async function renderEmployees() {
  actions().innerHTML = `<a class="btn" href="#/enrollment">+ ${esc(t("emp.new"))}</a>`;
  view().innerHTML = `
    <div class="card">
      <div class="toolbar">
        <div class="field" style="flex:1">
          <input id="emp-q" placeholder="${esc(t("common.search"))}..."
                 style="min-width:280px">
        </div>
      </div>
      <div class="table-wrap"><table class="data">
        <thead><tr>
          <th>${esc(t("emp.id"))}</th>
          <th>${esc(t("emp.name"))}</th>
          <th>${esc(t("emp.samples"))}</th>
          <th>${esc(t("common.actions"))}</th>
        </tr></thead>
        <tbody id="emp-body">${skeletonRows(4, 4)}</tbody>
      </table></div>
    </div>`;

  let workers = [];

  const openDelete = (empId) => {
    const w = workers.find(x => x.id === empId);
    if (!w) return;
    const m = modal({
      title: t("emp.delete.title"),
      body: `
        <p style="margin:0 0 12px">${esc(t("emp.delete.confirm"))}</p>
        <div style="background:var(--light-blue);padding:10px 12px;border-radius:6px;margin-bottom:12px">
          <b>${esc(w.id)}</b> — ${esc(w.name)}<br>
          <span style="font-size:12px;color:var(--muted)">${w.n} ${esc(t("emp.samples"))}</span>
        </div>
        <div class="field">
          <label>${esc(t("emp.delete.reason"))}</label>
          <select id="del-reason">
            <option value="duplicate">${esc(t("emp.delete.duplicate"))}</option>
            <option value="inactive">${esc(t("emp.delete.inactive"))}</option>
          </select>
        </div>
        <p style="font-size:12.5px;color:var(--muted);margin:0">${esc(t("emp.delete.note"))}</p>`,
      footer: `<button class="btn ghost" data-act="no">${esc(t("common.cancel"))}</button>
               <button class="btn danger" data-act="yes">${esc(t("common.delete"))}</button>`,
    });
    m.root.querySelector("[data-act=no]").onclick = m.close;
    m.root.querySelector("[data-act=yes]").onclick = async () => {
      const reason = m.root.querySelector("#del-reason").value;
      try {
        await API.deactivateWorker(w.id, reason);
        m.close();
        toast(t("emp.deleted_ok"), "success");
        workers = workers.filter(x => x.id !== w.id);
        paint();
      } catch (e) { toast(e.message, "error"); }
    };
  };

  const paint = () => {
    const q = (document.getElementById("emp-q")?.value || "").trim().toLowerCase();
    const rows = workers.filter(w =>
      !q || w.id.toLowerCase().includes(q) || w.name.toLowerCase().includes(q));
    document.getElementById("emp-body").innerHTML = rows.length
      ? rows.map(w => `
        <tr>
          <td class="mono"><b>${esc(w.id)}</b></td>
          <td>${esc(w.name)}</td>
          <td class="mono">${w.n}</td>
          <td><button class="btn danger sm" data-del="${esc(w.id)}">${esc(t("common.delete"))}</button></td>
        </tr>`).join("")
      : `<tr><td colspan="4">${emptyState()}</td></tr>`;
    document.querySelectorAll("[data-del]").forEach(b => {
      b.onclick = () => openDelete(b.dataset.del);
    });
  };

  document.getElementById("emp-q").oninput = paint;

  try {
    workers = await API.workers();
    paint();
  } catch (e) {
    toast(e.message, "error");
    document.getElementById("emp-body").innerHTML =
      `<tr><td colspan="4">${emptyState(t("common.backend_down"))}</td></tr>`;
  }
}

/* ============================================================
   ENROLLMENT
   ============================================================ */
let enrTimer = null, enrCamTimer = null;

export function cleanupEnrollment() {
  clearInterval(enrTimer); enrTimer = null;
  clearInterval(enrCamTimer); enrCamTimer = null;
}

export function renderEnrollment() {
  actions().innerHTML = "";
  view().innerHTML = `
    <div class="grid-2">
      <div class="card">
        <div class="card-title">${esc(t("enr.title"))}</div>
        <div class="field"><label>${esc(t("enr.emp_id"))}</label>
          <input id="enr-id" maxlength="64"></div>
        <div class="field"><label>${esc(t("enr.emp_name"))}</label>
          <input id="enr-name" maxlength="150"></div>
        <div style="display:flex;gap:8px">
          <button class="btn" id="btn-enr-start">${esc(t("enr.start"))}</button>
          <button class="btn ghost" id="btn-enr-cancel" disabled>${esc(t("enr.cancel"))}</button>
        </div>
        <div style="margin-top:20px">
          <div style="font-size:12.5px;font-weight:600;color:var(--muted);margin-bottom:6px">
            ${esc(t("enr.progress"))}</div>
          <div class="progress"><div id="enr-bar" style="width:0%"></div></div>
          <div style="margin-top:6px"><span id="enr-count">0</span>/10</div>
        </div>
        <div style="margin-top:16px">
          <div style="font-size:12.5px;font-weight:600;color:var(--muted)">
            ${esc(t("enr.pose"))}</div>
          <div id="enr-pose" style="margin-top:4px">—</div>
        </div>
        <div id="enr-msg" style="margin-top:14px;padding:10px 12px;border-radius:6px;
             background:var(--light-blue);font-size:13px">
          ${esc(t("enr.state.idle"))}</div>
      </div>
      <div class="card" style="padding:0;overflow:hidden">
        <div class="cam-box">
          <img id="enr-cam" alt="camera">
          <div class="cam-overlay-msg" id="enr-cam-msg">${esc(t("common.loading"))}</div>
        </div>
      </div>
    </div>`;

  const camTick = () => {
    const img = document.getElementById("enr-cam");
    if (!img) return;
    img.src = API.snapshotUrl();
    img.onload = () => document.getElementById("enr-cam-msg")?.classList.add("hidden");
    img.onerror = () => document.getElementById("enr-cam-msg")?.classList.remove("hidden");
  };
  camTick();
  enrCamTimer = setInterval(camTick, 1000);

  const beginPoll = () => {
    clearInterval(enrTimer);
    enrTimer = setInterval(async () => {
      try {
        const s = await API.enrollStatus();
        document.getElementById("enr-bar").style.width = `${(s.count / s.total) * 100}%`;
        document.getElementById("enr-count").textContent = s.count;
        document.getElementById("enr-pose").textContent = s.pose || "—";
        document.getElementById("enr-msg").textContent =
          s.message || t("enr.state." + s.state);

        if (s.state === "done") {
          clearInterval(enrTimer);
          document.getElementById("btn-enr-start").disabled = false;
          document.getElementById("btn-enr-cancel").disabled = true;
          toast(t("enr.success"), "success");
        } else if (s.state === "cancelled" || s.state === "error") {
          clearInterval(enrTimer);
          document.getElementById("btn-enr-start").disabled = false;
          document.getElementById("btn-enr-cancel").disabled = true;
        }
      } catch {}
    }, 500);
  };

  const start = async (replace = false) => {
    const id = document.getElementById("enr-id").value.trim();
    const name = document.getElementById("enr-name").value.trim();
    if (!id || !name) { toast(t("enr.fill_required"), "error"); return; }
    try {
      await API.enrollStart(id, name, replace);
      document.getElementById("btn-enr-start").disabled = true;
      document.getElementById("btn-enr-cancel").disabled = false;
      beginPoll();
    } catch (e) {
      if (e.status === 409) {
        const ok = await confirmModal({
          title: t("enr.exists"),
          message: t("enr.replace"),
          confirmText: t("enr.replace"),
          danger: true,
        });
        if (ok) start(true);
      } else toast(e.message, "error");
    }
  };

  document.getElementById("btn-enr-start").onclick = () => start(false);
  document.getElementById("btn-enr-cancel").onclick = async () => {
    try { await API.enrollCancel(); } catch {}
  };
}

/* ============================================================
   HISTORY
   ============================================================ */
export async function renderHistory() {
  actions().innerHTML = "";
  view().innerHTML = `
    <div class="card">
      <div class="toolbar">
        <div class="field"><label>${esc(t("vio.filter.date"))}</label>
          <input type="date" id="h-date"></div>
        <div class="field"><label>${esc(t("vio.filter.worker"))}</label>
          <input id="h-worker" placeholder="ID / Name"></div>
        <div class="field"><label>${esc(t("vio.filter.type"))}</label>
          <select id="h-type">
            <option value="">${esc(t("vio.all"))}</option>
            <option value="Mu">${esc(t("ppe.hat"))}</option>
            <option value="Khau trang">${esc(t("ppe.mask"))}</option>
            <option value="Ao bao ho">${esc(t("ppe.vest"))}</option>
          </select></div>
        <button class="btn ghost" id="h-search">${esc(t("common.search"))}</button>
        <button class="btn ghost" id="h-export">${esc(t("his.export"))}</button>
      </div>
      <div class="table-wrap"><table class="data">
        <thead><tr>
          <th>${esc(t("his.stt"))}</th>
          <th>${esc(t("vio.filter.date"))}</th>
          <th>${esc(t("emp.id"))}</th>
          <th>${esc(t("emp.name"))}</th>
          <th>${esc(t("vio.filter.type"))}</th>
          <th>${esc(t("common.evidence"))}</th>
          <th>${esc(t("common.status"))}</th>
        </tr></thead>
        <tbody id="h-body">${skeletonRows(6, 7)}</tbody>
      </table></div>
      <div id="h-page" style="margin-top:14px;display:flex;gap:8px;justify-content:flex-end"></div>
    </div>`;

  const state = { page: 1, limit: 20, data: null };

  const load = async () => {
    const p = {
      date: document.getElementById("h-date").value,
      worker_id: document.getElementById("h-worker").value.trim(),
      type: document.getElementById("h-type").value,
      page: state.page, limit: state.limit,
    };
    try {
      const r = await API.violations(p);
      state.data = r;
      const items = r.items || [];
      document.getElementById("h-body").innerHTML = items.length
        ? items.map((v, i) => `
          <tr>
            <td>${(r.page - 1) * r.limit + i + 1}</td>
            <td class="mono">${v.date} ${v.time}</td>
            <td><b>${esc(v.wid || "UNKNOWN")}</b></td>
            <td>${esc(v.w_name || "—")}</td>
            <td>${esc(translateTypeCodes(v.type_raw))}</td>
            <td><img class="img-thumb" src="${esc(v.img)}" data-evidence="${esc(v.img)}"></td>
            <td>${v.wid
              ? `<span class="badge pass">${esc(t("vio.status.assigned"))}</span>`
              : `<span class="badge unknown">${esc(t("vio.status.unassigned"))}</span>`}</td>
          </tr>`).join("")
        : `<tr><td colspan="7">${emptyState()}</td></tr>`;
      document.querySelectorAll("#h-body [data-evidence]").forEach(img => {
        img.onclick = () => evidenceModal(img.dataset.evidence);
      });

      const pg = document.getElementById("h-page");
      pg.innerHTML = `
        <button class="btn ghost sm" ${r.page <= 1 ? "disabled" : ""} id="pg-prev">‹</button>
        <span style="padding:6px 10px;font-size:13px">${r.page} / ${r.total_pages}</span>
        <button class="btn ghost sm" ${r.page >= r.total_pages ? "disabled" : ""} id="pg-next">›</button>`;
      document.getElementById("pg-prev")?.addEventListener("click", () => {
        if (state.page > 1) { state.page--; load(); }
      });
      document.getElementById("pg-next")?.addEventListener("click", () => {
        if (state.page < r.total_pages) { state.page++; load(); }
      });
    } catch (e) { toast(e.message, "error"); }
  };

  document.getElementById("h-search").onclick = () => { state.page = 1; load(); };
  document.getElementById("h-export").onclick = () => {
    if (!state.data?.items?.length) { toast(t("common.no_data"), "error"); return; }
    const rows = [["date","time","worker_id","worker_name","type","status"]];
    state.data.items.forEach(v => rows.push([
      v.date, v.time, v.wid || "UNKNOWN", v.w_name || "",
      translateTypeCodes(v.type_raw), v.status,
    ]));
    const csv = rows.map(r => r.map(x =>
      `"${String(x).replace(/"/g, '""')}"`).join(",")).join("\n");
    const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `violations_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  };

  load();
}

/* ============================================================
   ANALYTICS
   ============================================================ */
let anaChartDay = null, anaChartType = null;

export async function renderAnalytics() {
  actions().innerHTML = "";
  view().innerHTML = `
    <div class="card" style="margin-bottom:18px">
      <div class="toolbar">
        <div class="field"><label>${esc(t("ana.range"))}</label>
          <select id="a-days">
            <option value="7">7</option>
            <option value="30" selected>30</option>
            <option value="90">90</option>
            <option value="365">365</option>
          </select></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">${esc(t("dash.by_day"))}</div>
        <div style="height:280px"><canvas id="a-day"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">${esc(t("dash.by_type"))}</div>
        <div style="height:280px"><canvas id="a-type"></canvas></div>
      </div>
    </div>
    <div class="card" style="margin-top:18px">
      <div class="card-title">${esc(t("dash.top_workers"))}</div>
      <div id="a-top">${emptyState()}</div>
    </div>`;

  const draw = async () => {
    const days = +document.getElementById("a-days").value;
    try {
      const [d, ty, top] = await Promise.all([
        API.statsByDay(days), API.statsByType(), API.statsTopWorkers(10),
      ]);
      anaChartDay?.destroy();
      anaChartType?.destroy();
      anaChartDay = new Chart(document.getElementById("a-day"), {
        type: "bar",
        data: { labels: d.map(x => x.date.slice(5)),
                datasets: [{ data: d.map(x => x.count), backgroundColor: "#1763AE" }] },
        options: {
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
        },
      });
      anaChartType = new Chart(document.getElementById("a-type"), {
        type: "bar",
        data: {
          labels: ty.map(x => t(`ppe.type.raw.${x.type}`) || x.type),
          datasets: [{
            data: ty.map(x => x.count),
            backgroundColor: ["#1763AE", "#F59E0B", "#DC2626"],
          }],
        },
        options: {
          indexAxis: "y",
          plugins: { legend: { display: false } },
          scales: { x: { beginAtZero: true, ticks: { precision: 0 } } },
        },
      });
      document.getElementById("a-top").innerHTML = top.length
        ? `<table class="data"><tbody>${top.map(w =>
            `<tr>
               <td><b>${esc(w.id)}</b> — ${esc(w.name)}</td>
               <td style="text-align:right;font-weight:700;color:var(--danger)">${w.count}</td>
             </tr>`).join("")}</tbody></table>`
        : emptyState();
    } catch (e) { toast(e.message, "error"); }
  };

  document.getElementById("a-days").onchange = draw;
  draw();
}

/* ============================================================
   SETTINGS
   ============================================================ */
export async function renderSettings() {
  actions().innerHTML = "";
  view().innerHTML = `
    <div class="card" style="margin-bottom:18px">
      <div class="card-title">${esc(t("set.system"))}</div>
      <div class="grid-3" id="sys-grid">
        ${[1,2,3,4,5].map(() => `<div class="card">
          <div class="skeleton" style="width:40%"></div>
          <div class="skeleton" style="margin-top:10px"></div>
        </div>`).join("")}
      </div>
    </div>
    <div class="card">
      <div class="card-title">${esc(t("set.language"))}</div>
      <div class="segmented">
        <button class="lang-opt" data-lang="vi">Tiếng Việt</button>
        <button class="lang-opt" data-lang="en">English</button>
      </div>
      <p style="font-size:12.5px;color:var(--muted);margin-top:10px">
        ${esc(t("set.lang_hint"))}</p>
    </div>`;

  try {
    const s = await API.systemInfo();
    const dot = (ok) => `<span class="status-pill ${ok ? "online" : ""}"
      style="font-size:11px;padding:3px 8px">
      <span class="dot"></span>${ok ? "OK" : "—"}</span>`;
    document.getElementById("sys-grid").innerHTML = `
      <div class="card">
        <div class="stat-label">${esc(t("set.camera"))}</div>
        <div style="margin-top:8px">${dot(s.camera.online)}</div>
        <div style="margin-top:6px;font-size:12px;color:var(--muted)">
          ${s.camera.age != null ? `${s.camera.age.toFixed(1)}s` : "—"}</div>
      </div>
      <div class="card">
        <div class="stat-label">${esc(t("set.backend"))}</div>
        <div style="margin-top:8px">${dot(s.backend.running)}</div>
        <div style="margin-top:6px;font-size:12px;color:var(--muted)">
          ${esc(t("set.uptime"))}: ${formatUptime(s.backend.uptime)}</div>
      </div>
      <div class="card">
        <div class="stat-label">${esc(t("set.yolo"))}</div>
        <div style="margin-top:8px">${dot(s.yolo.loaded)}</div>
        <div style="margin-top:6px;font-size:12px;color:var(--muted)">
          ${esc(s.yolo.model)} · ${s.yolo.imgsz}</div>
      </div>
      <div class="card">
        <div class="stat-label">${esc(t("set.face"))}</div>
        <div style="margin-top:8px">${dot(s.face.loaded)}</div>
        <div style="margin-top:6px;font-size:12px;color:var(--muted)">
          ${esc(t("set.workers_count"))}: ${s.face.employees}</div>
      </div>
      <div class="card">
        <div class="stat-label">${esc(t("set.database"))}</div>
        <div style="margin-top:8px">${dot(true)}</div>
        <div style="margin-top:6px;font-size:12px;color:var(--muted)">
          ${s.database.size_kb} KB · ${s.database.workers} workers</div>
      </div>`;
  } catch (e) { toast(e.message, "error"); }
}