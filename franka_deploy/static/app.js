// franka-deploy dashboard -- vanilla JS, no build step (single control panel
// doesn't earn a framework). All state truth lives server-side
// (control_loop.py's State machine); this just reflects it.

const $ = (id) => document.getElementById(id);

const DEFAULT_SCHEMA_FIELDS = [
  { name: "instruction", source: "static:pick up the cup", dtype: null, shape: null,
    resize: null, layout: "HWC", normalize: null, encoding: "none", transform_fn: null },
];

$("schema-editor").value = JSON.stringify(DEFAULT_SCHEMA_FIELDS, null, 2);

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
    robot_ip: $("robot-ip").value.trim(),
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
      ema_alpha: 0.3,
      watchdog_enabled: true,
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
  const instruction = $("instruction").value.trim() || null;
  const r = await fetch("/api/control/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction }),
  });
  const data = await r.json();
  if (!r.ok) { alert("시작 실패: " + (data.detail || r.status)); return; }
  setBadge(data.state);
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
function connectTelemetry() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  ws.onmessage = (ev) => {
    const t = JSON.parse(ev.data);
    setBadge(t.state);
    $("telemetry-view").textContent = JSON.stringify(t, null, 2);
  };
  ws.onclose = () => setTimeout(connectTelemetry, 1000);
  ws.onerror = () => ws.close();
}
connectTelemetry();

// initial state
fetch("/api/control/state").then((r) => r.json()).then((d) => setBadge(d.state)).catch(() => {});
