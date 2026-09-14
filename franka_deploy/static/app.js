// franka-deploy dashboard -- vanilla JS, no build step (single control panel
// doesn't earn a framework). All state truth lives server-side
// (control_loop.py's State machine); this just reflects it.

const $ = (id) => document.getElementById(id);

// --------------------------------------------------------------------- tabs
function setTab(name) {
  document.querySelectorAll(".tab-page").forEach((el) => {
    el.classList.toggle("hidden", el.dataset.tab !== name);
  });
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  try { localStorage.setItem("franka-deploy-tab", name); } catch (e) { /* ignore */ }
}
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});
(() => {
  let initial = "settings";
  try { initial = localStorage.getItem("franka-deploy-tab") || "settings"; } catch (e) { /* ignore */ }
  const fromUrl = new URLSearchParams(location.search).get("tab");
  if (fromUrl === "settings" || fromUrl === "deploy") initial = fromUrl;
  setTab(initial);
})();

const DEFAULT_SCHEMA_FIELDS = [
  { name: "instruction", source: "static:pick up the cup", dtype: null, shape: null,
    resize: null, layout: "HWC", normalize: null, encoding: "none", transform_fn: null },
];

$("schema-editor").value = JSON.stringify(DEFAULT_SCHEMA_FIELDS, null, 2);

// ------------------------------------------------------------- schema table
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function readSchemaFields() {
  return JSON.parse($("schema-editor").value);
}

function writeSchemaFields(fields) {
  $("schema-editor").value = JSON.stringify(fields, null, 2);
  renderSchemaSummary();
}

function renderSchemaSummary() {
  const tbody = $("schema-summary-body");
  let fields;
  try {
    fields = JSON.parse($("schema-editor").value);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="6" class="err">JSON 파싱 오류: ${escapeHtml(e.message)}</td></tr>`;
    return;
  }
  if (!Array.isArray(fields) || fields.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="hint">필드 없음</td></tr>`;
    return;
  }
  tbody.innerHTML = fields.map((f, i) => `
    <tr>
      <td class="mono">${escapeHtml(f.name ?? "")}</td>
      <td class="mono">${escapeHtml(f.source ?? "")}</td>
      <td>${escapeHtml(f.dtype ?? "-")}</td>
      <td>${f.shape ? escapeHtml(JSON.stringify(f.shape)) : "-"}</td>
      <td>${escapeHtml(f.encoding ?? "none")}</td>
      <td><button class="field-delete-btn" data-idx="${i}" title="필드 삭제">✕</button></td>
    </tr>`).join("");
}

$("schema-summary-body").addEventListener("click", (ev) => {
  const btn = ev.target.closest(".field-delete-btn");
  if (!btn) return;
  const idx = parseInt(btn.dataset.idx, 10);
  const fields = readSchemaFields();
  fields.splice(idx, 1);
  writeSchemaFields(fields);
});

$("schema-editor").addEventListener("input", renderSchemaSummary);
renderSchemaSummary();

// ---------------------------------------------------------------- field builder
function parseIntList(s) {
  if (!s || !s.trim()) return null;
  return s.split(",").map((v) => parseInt(v.trim(), 10));
}

$("btn-add-field").addEventListener("click", () => {
  const name = $("fb-name").value.trim();
  const source = $("fb-source").value.trim();
  if (!name || !source) { alert("필드명과 출처는 필수입니다."); return; }

  const scaleStr = $("fb-normalize-scale").value.trim();
  const field = {
    name,
    source,
    dtype: $("fb-dtype").value || null,
    shape: parseIntList($("fb-shape").value),
    resize: parseIntList($("fb-resize").value),
    layout: $("fb-layout").value,
    normalize: scaleStr ? { scale: parseFloat(scaleStr) } : null,
    encoding: $("fb-encoding").value,
    transform_fn: $("fb-transform-fn").value.trim() || null,
  };

  let fields;
  try {
    fields = readSchemaFields();
    if (!Array.isArray(fields)) fields = [];
  } catch (e) {
    fields = [];
  }
  fields.push(field);
  writeSchemaFields(fields);

  for (const id of ["fb-name", "fb-source", "fb-shape", "fb-resize", "fb-normalize-scale", "fb-transform-fn"]) {
    $(id).value = "";
  }
  $("fb-dtype").value = "";
  $("fb-layout").value = "HWC";
  $("fb-encoding").value = "none";
});

// ------------------------------------------------------------ state badge
function setBadge(state) {
  const el = $("state-badge");
  el.textContent = state;
  el.className = "badge " + state;
}

// ------------------------------------------------------------------ config
function gatherConfig() {
  let fields;
  try {
    fields = JSON.parse($("schema-editor").value);
  } catch (e) {
    $("schema-status").textContent = "JSON 파싱 오류: " + e.message;
    throw e;
  }
  const cameras = [];
  if ($("cam-agent-serial").value.trim()) {
    cameras.push({ role: "agentview", serial: $("cam-agent-serial").value.trim() });
  } else {
    cameras.push({ role: "agentview", serial: null });
  }
  if ($("cam-wrist-serial").value.trim()) {
    cameras.push({ role: "eye_in_hand", serial: $("cam-wrist-serial").value.trim() });
  } else {
    cameras.push({ role: "eye_in_hand", serial: null });
  }
  return {
    robot_node_address: $("robot-node-address").value.trim(),
    cameras,
    request_spec: {
      connection: {
        server_ip: $("server-ip").value.trim(),
        server_port: parseInt($("server-port").value, 10),
        scheme: "http",
        predict_endpoint: $("predict-endpoint").value.trim(),
        reset_endpoint: $("reset-endpoint").value.trim(),
      },
      fields,
      actions_key: $("actions-key").value.trim(),
      instruction_field: $("instruction-field").value.trim() || null,
    },
    loop: {
      fps: parseFloat($("loop-fps").value),
      exec_horizon: parseInt($("loop-exec-horizon").value, 10),
      lead_ticks: parseInt($("loop-lead-ticks").value, 10),
      max_step_rad: parseFloat($("loop-max-step").value),
      ema_alpha: parseFloat($("loop-ema-alpha").value),
      watchdog_enabled: $("loop-watchdog-enabled").checked,
      watchdog_window: parseInt($("loop-watchdog-window").value, 10),
      watchdog_threshold: parseFloat($("loop-watchdog-threshold").value),
      watchdog_trip_count: parseInt($("loop-watchdog-trip").value, 10),
      watchdog_deadzone: parseFloat($("loop-watchdog-deadzone").value),
    },
    v_max: parseFloat($("v-max").value),
    a_max: parseFloat($("a-max").value),
    j_max: parseFloat($("j-max").value),
    filter_wn: parseFloat($("filter-wn").value),
  };
}

async function saveConfig() {
  let body;
  try {
    body = gatherConfig();
  } catch (e) {
    return;
  }
  const r = await fetch("/api/config", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    $("schema-status").textContent = "저장 실패: " + (err.detail || r.status);
    return;
  }
  $("schema-status").textContent = "저장됨 (" + new Date().toLocaleTimeString() + ")";
}

$("btn-save-config").addEventListener("click", saveConfig);
$("btn-apply-schema").addEventListener("click", saveConfig);

// -------------------------------------------------------------- examples
async function loadExampleList() {
  const r = await fetch("/api/config/examples");
  const names = await r.json();
  const sel = $("example-select");
  for (const n of names) {
    const opt = document.createElement("option");
    opt.value = n; opt.textContent = n;
    sel.appendChild(opt);
  }
}
loadExampleList();

$("btn-load-example").addEventListener("click", async () => {
  const name = $("example-select").value;
  if (!name) return;
  const r = await fetch(`/api/config/examples/${name}`);
  if (!r.ok) return;
  const cfg = await r.json();
  if (cfg.request_spec) {
    $("schema-editor").value = JSON.stringify(cfg.request_spec.fields, null, 2);
    const c = cfg.request_spec.connection || {};
    if (c.predict_endpoint) $("predict-endpoint").value = c.predict_endpoint;
    if (c.reset_endpoint) $("reset-endpoint").value = c.reset_endpoint;
    if (cfg.request_spec.actions_key) $("actions-key").value = cfg.request_spec.actions_key;
    if (cfg.request_spec.instruction_field) $("instruction-field").value = cfg.request_spec.instruction_field;
  }
  $("schema-status").textContent = `예시 '${name}' 불러옴 (참고용, 저장하려면 '스키마 적용' 클릭)`;
});

// ----------------------------------------------------------------- workspace
function renderWorkspace(data) {
  const tbody = $("workspace-points-body");
  if (!data.points || data.points.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="hint">기록된 경계점 없음 -- 안전장치 꺼짐</td></tr>`;
  } else {
    tbody.innerHTML = data.points.map((p, i) => `
      <tr>
        <td>${i + 1}</td>
        <td class="mono">${p[0].toFixed(3)}</td>
        <td class="mono">${p[1].toFixed(3)}</td>
        <td class="mono">${p[2].toFixed(3)}</td>
        <td><button class="field-delete-btn" data-idx="${i}" title="이 점 삭제">✕</button></td>
      </tr>`).join("");
  }
  const bboxEl = $("workspace-bbox");
  if (data.enabled) {
    const f = (arr) => arr.map((v) => v.toFixed(3)).join(", ");
    bboxEl.textContent = `활성화됨 -- 허용 범위: x/y/z ∈ [${f(data.lo)}] ~ [${f(data.hi)}] (margin ${data.margin} m)`;
  } else {
    bboxEl.textContent = "2개 이상 점을 찍어야 활성화됩니다.";
  }
  $("workspace-margin").value = data.margin;
}

async function refreshWorkspace() {
  const r = await fetch("/api/control/workspace");
  if (r.ok) renderWorkspace(await r.json());
}
refreshWorkspace();

$("btn-workspace-add").addEventListener("click", async () => {
  const r = await fetch("/api/control/workspace/add_point", { method: "POST" });
  const data = await r.json();
  if (!r.ok) { alert("경계점 추가 실패: " + (data.detail || r.status)); return; }
  renderWorkspace(data);
});

$("btn-workspace-clear").addEventListener("click", async () => {
  if (!confirm("기록된 경계점을 전부 지울까요?")) return;
  const r = await fetch("/api/control/workspace/clear", { method: "POST" });
  renderWorkspace(await r.json());
});

$("btn-workspace-margin").addEventListener("click", async () => {
  const margin = parseFloat($("workspace-margin").value);
  const r = await fetch("/api/control/workspace/margin", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ margin }),
  });
  renderWorkspace(await r.json());
});

$("workspace-points-body").addEventListener("click", async (ev) => {
  const btn = ev.target.closest(".field-delete-btn");
  if (!btn) return;
  const index = parseInt(btn.dataset.idx, 10);
  const r = await fetch("/api/control/workspace/remove_point", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index }),
  });
  renderWorkspace(await r.json());
});

// ------------------------------------------------------------------- cameras
async function scanCameras() {
  const r = await fetch("/api/cameras/devices");
  if (!r.ok) return;
  const devices = await r.json();  // [{name, serial}, ...]
  for (const selectId of ["cam-agent-serial", "cam-wrist-serial"]) {
    const sel = $(selectId);
    const prev = sel.value;
    sel.innerHTML = '<option value="">(mock 이미지)</option>';
    for (const d of devices) {
      const opt = document.createElement("option");
      opt.value = d.serial;
      opt.textContent = `${d.name} — ${d.serial}`;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  }
}

$("btn-scan-cameras").addEventListener("click", scanCameras);
scanCameras();

let cameraPreviewTimer = null;

function refreshCameraFrames() {
  const t = Date.now();
  for (const role of ["agentview", "eye_in_hand"]) {
    $(`cam-preview-${role}`).src = `/api/cameras/${role}/frame.jpg?t=${t}`;
  }
}

$("btn-camera-start").addEventListener("click", async () => {
  const r = await fetch("/api/cameras/start", { method: "POST" });
  const data = await r.json();
  if (!r.ok) { alert("카메라 시작 실패: " + (data.detail || r.status)); return; }
  if (cameraPreviewTimer) clearInterval(cameraPreviewTimer);
  cameraPreviewTimer = setInterval(refreshCameraFrames, 500);
  refreshCameraFrames();
});

$("btn-camera-stop").addEventListener("click", async () => {
  if (cameraPreviewTimer) { clearInterval(cameraPreviewTimer); cameraPreviewTimer = null; }
  const r = await fetch("/api/cameras/stop", { method: "POST" });
  const data = await r.json();
  if (!r.ok) alert("카메라 중지 실패: " + (data.detail || r.status));
});

// ------------------------------------------------------------------ connect
$("btn-connect").addEventListener("click", async () => {
  const readOnly = $("read-only-toggle").checked;
  const r = await fetch("/api/control/connect", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ read_only: readOnly }),
  });
  const data = await r.json();
  if (!r.ok) { alert("연결 실패: " + (data.detail || r.status)); return; }
  setBadge(data.state);
});

$("btn-disconnect").addEventListener("click", async () => {
  const r = await fetch("/api/control/disconnect", { method: "POST" });
  const data = await r.json();
  setBadge(data.state);
  $("detect-result").classList.add("hidden");
});

// ------------------------------------------------------------------- detect
function fillOverrideFromSpec(spec) {
  $("override-kind").value = spec.kind;
  $("override-has-gripper").checked = spec.has_gripper;
  $("override-gripper-conv").value = spec.gripper_convention;
  $("override-rot-repr").value = spec.rotation_repr || "";
  $("override-pos-scale").value = spec.delta_pos_scale;
  $("override-rot-scale").value = spec.delta_rot_scale;
}

let lastDetectDim = null;

$("btn-detect").addEventListener("click", async () => {
  const instruction = $("instruction").value.trim() || null;
  const r = await fetch("/api/control/detect", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction }),
  });
  const data = await r.json();
  if (!r.ok) { alert("감지 실패: " + (data.detail || r.status)); return; }
  setBadge(data.state);
  lastDetectDim = data.spec.dim;

  const tbody = $("detect-table");
  tbody.innerHTML = "";
  const rows = [
    ["감지된 종류", data.spec.kind],
    ["차원 (D)", data.spec.dim],
    ["그리퍼 포함", data.spec.has_gripper],
    ["그리퍼 컨벤션", data.spec.gripper_convention],
    ["회전 표현", data.spec.rotation_repr || "-"],
    ["확신도", data.spec.confidence.toFixed(2)],
    ["청크 shape", data.chunk_shape.join(" x ")],
    ["샘플 행", data.sample_row.map((v) => v.toFixed(3)).join(", ")],
    ["비고", data.spec.notes],
  ];
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="k">${k}</td><td>${v}</td>`;
    tbody.appendChild(tr);
  }
  $("detect-result").classList.remove("hidden");
  fillOverrideFromSpec(data.spec);
});

$("btn-confirm").addEventListener("click", async () => {
  const spec = {
    kind: $("override-kind").value,
    dim: lastDetectDim,
    has_gripper: $("override-has-gripper").checked,
    gripper_convention: $("override-gripper-conv").value,
    rotation_repr: $("override-rot-repr").value || null,
    confidence: 1.0,
    notes: "user-confirmed",
    delta_pos_scale: parseFloat($("override-pos-scale").value),
    delta_rot_scale: parseFloat($("override-rot-scale").value),
    delta_joint_scale: 1.0,
  };
  const r = await fetch("/api/control/confirm", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ spec }),
  });
  const data = await r.json();
  if (!r.ok) { alert("확정 실패: " + (data.detail || r.status)); return; }
  setBadge(data.state);
});

// --------------------------------------------------------------- run controls
$("btn-start").addEventListener("click", async () => {
  const btn = $("btn-start");
  if (btn.disabled) return;  // guard against double-click firing two /start calls
  btn.disabled = true;
  try {
    const instruction = $("instruction").value.trim() || null;
    const r = await fetch("/api/control/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction }),
    });
    const data = await r.json();
    if (!r.ok) { alert("시작 실패: " + (data.detail || r.status)); return; }
    setBadge(data.state);
  } finally {
    setTimeout(() => { btn.disabled = false; }, 1000);
  }
});

$("btn-stop").addEventListener("click", async () => {
  const r = await fetch("/api/control/stop", { method: "POST" });
  const data = await r.json();
  setBadge(data.state);
});

$("btn-estop").addEventListener("click", async () => {
  const r = await fetch("/api/control/estop", { method: "POST" });
  const data = await r.json();
  setBadge(data.state);
});

// ------------------------------------------------------------------ telemetry
function statTile(label, value, cls) {
  return `<div class="stat${cls ? " " + cls : ""}"><div class="stat-label">${escapeHtml(label)}</div>` +
         `<div class="stat-value">${escapeHtml(value)}</div></div>`;
}

function renderTelemetryGrid(t) {
  const grid = $("telemetry-grid");
  const tiles = [];

  tiles.push(statTile("상태", t.state ?? "-"));

  if (t.control_command_success_rate !== undefined && t.control_command_success_rate !== null) {
    const rate = t.control_command_success_rate;
    const cls = rate >= 0.99 ? "stat-ok" : rate >= 0.9 ? "stat-warn" : "stat-error";
    tiles.push(statTile("제어 성공률", (rate * 100).toFixed(1) + "%", cls));
  }
  if (t.last_predict_ms !== undefined && t.last_predict_ms !== null) {
    tiles.push(statTile("정책 지연", t.last_predict_ms.toFixed(1) + " ms"));
  }
  if (t.n_replans !== undefined) tiles.push(statTile("재계획 횟수", t.n_replans));
  if (t.n_late !== undefined) {
    tiles.push(statTile("예산 초과", t.n_late, t.n_late > 0 ? "stat-warn" : "stat-ok"));
  }
  if (Array.isArray(t.dq)) {
    const maxDq = Math.max(...t.dq.map(Math.abs));
    tiles.push(statTile("최대 |dq|", maxDq.toFixed(3) + " rad/s", maxDq > 0.8 ? "stat-warn" : ""));
  }
  if (Array.isArray(t.ee_pos)) {
    tiles.push(statTile("EE 위치 (m)", t.ee_pos.map((v) => v.toFixed(3)).join(", ")));
  }
  if (t.error) {
    tiles.push(`<div class="stat stat-error" style="grid-column:1/-1"><div class="stat-label">에러</div>` +
                `<div class="stat-value">${escapeHtml(t.error)}</div></div>`);
  }

  grid.innerHTML = tiles.join("");
}

function connectTelemetry() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  ws.onmessage = (ev) => {
    const t = JSON.parse(ev.data);
    setBadge(t.state);
    renderTelemetryGrid(t);
    $("telemetry-view").textContent = JSON.stringify(t, null, 2);
  };
  ws.onclose = () => setTimeout(connectTelemetry, 1000);
  ws.onerror = () => ws.close();
}
connectTelemetry();

// initial state
fetch("/api/control/state").then((r) => r.json()).then((d) => setBadge(d.state)).catch(() => {});
