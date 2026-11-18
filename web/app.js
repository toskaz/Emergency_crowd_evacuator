const CELL = { EMPTY: 0, WALL: 1, EXIT: 2, EVACUEE: 3, DRONE: 4, STAIRS: 5 };
const COLORS = {
  exterior: "#090D16",
  floor: "#111827",
  grid: "rgba(100,116,139,.22)",
  wall: "#CBD5E1",
  exit: "#10B981",
  stairs: "#818CF8",
  evacuee: "#F59E0B",
  drone: "#38BDF8",
  hazard: "#EF4444"
};
const canvas = document.querySelector("#floorplan");
const ctx = canvas.getContext("2d");
const sidePanel = document.querySelector("#sidePanel");
const floorSelector = document.querySelector("#floorSelector");
const connectionLabel = document.querySelector("#connectionLabel");
const uiState = {
  showGrid: false,
  showClusters: false,
  showDroneRoutes: false,
  showHeatmap: false
};
const overlayStateKeys = {
  grid: "showGrid",
  clusters: "showClusters",
  routes: "showDroneRoutes",
  heatmap: "showHeatmap"
};
const motion = new Map();
let state = null;
let socket;
let selectedTool = "WALL";
let selectedStair = null;
let linkMode = false;
let spaceHeld = false;
let panning = false;
let panPoint = null;
let stroke = null;
const pendingStrokeCells = new Map();
let pendingStrokeFloorId = null;
let toastTimer;
let view = { scale: 8, offsetX: 0, offsetY: 0, floorId: null };
let dpr = 1;

function toast(message, error = false) {
  const element = document.querySelector("#toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("visible"), 2600);
}

function send(type, payload = {}) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    toast("Reconnecting to simulation server…", true);
    return false;
  }
  socket.send(JSON.stringify({ type, ...payload }));
  return true;
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${scheme}//${location.host}/ws`);
  socket.addEventListener("open", () => {
    connectionLabel.textContent = "Live connection";
    document.querySelector(".live-dot").classList.add("connected");
  });
  socket.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    if (message.type === "state") {
      updateState(message.state);
    } else if (message.type === "error") {
      pendingStrokeCells.clear();
      pendingStrokeFloorId = null;
      toast(message.message, true);
      if (message.state) updateState(message.state);
    }
  });
  socket.addEventListener("close", () => {
    connectionLabel.textContent = "Reconnecting";
    document.querySelector(".live-dot").classList.remove("connected");
    setTimeout(connect, 1400);
  });
}

function updateState(next) {
  if (state) {
    const oldById = new Map(state.evacuees.map(person => [`e${person.id}`, person]));
    for (const person of next.evacuees) {
      const old = oldById.get(`e${person.id}`);
      if (old && old.floor_id === person.floor_id && (old.x !== person.x || old.y !== person.y)) {
        motion.set(`e${person.id}`, { from: [old.x, old.y], to: [person.x, person.y], start: performance.now() });
      }
    }
    const oldDrones = new Map(state.drones.map(drone => [`d${drone.id}`, drone]));
    for (const drone of next.drones) {
      const old = oldDrones.get(`d${drone.id}`);
      if (old && old.floor_id === drone.floor_id && (old.x !== drone.x || old.y !== drone.y)) {
        motion.set(`d${drone.id}`, { from: [old.x, old.y], to: [drone.x, drone.y], start: performance.now() });
      }
    }
  }
  if (pendingStrokeFloorId !== null) {
    const floor = next.floors.find(item => item.id === pendingStrokeFloorId);
    if (floor) {
      for (const [key, previewType] of pendingStrokeCells) {
        const [x, y] = key.split(",").map(Number);
        if (
          x < 0 || y < 0 || x >= floor.width || y >= floor.height ||
          floor.cells[y][x] === previewType
        ) {
          pendingStrokeCells.delete(key);
        }
      }
      if (!pendingStrokeCells.size) pendingStrokeFloorId = null;
    }
  }
  if (selectedStair) {
    const floor = next.floors.find(item => item.id === selectedStair.floorId);
    if (!floor?.clusters.some(cluster =>
      cluster.type === "STAIRS" && cluster.id === selectedStair.clusterId
    )) {
      selectedStair = null;
    }
  }
  const floorChanged = !state || state.active_floor_id !== next.active_floor_id;
  state = next;
  if (floorChanged) fitFloor();
  render();
}

function activeFloor() {
  return state?.floors.find(floor => floor.id === state.active_floor_id);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  renderCanvas();
}

function fitFloor() {
  const floor = activeFloor();
  if (!floor) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const availableWidth = Math.max(260, width - (window.innerWidth > 760 ? 390 : 32));
  const availableHeight = Math.max(260, height - 190);
  view.scale = Math.max(3, Math.min(availableWidth / floor.width, availableHeight / floor.height));
  view.offsetX = (availableWidth - floor.width * view.scale) / 2 + 20;
  view.offsetY = (height - floor.height * view.scale) / 2 + 14;
  view.floorId = floor.id;
  renderCanvas();
}

function render() {
  if (!state) return;
  renderFloorSelector();
  renderWorkflow();
  renderSidebar();
  renderOverlayButtons();
  if (window.lucide) lucide.createIcons();
  renderCanvas();
}

function renderFloorSelector() {
  floorSelector.innerHTML = state.floors.map(floor => `
    <button class="floor-pill ${floor.id === state.active_floor_id ? "active" : ""}" data-floor="${floor.id}">${escapeHtml(floor.name)}</button>
  `).join("") + `<button class="add-floor" data-action="upload"><i data-lucide="plus"></i> Add floor</button>`;
}

function renderWorkflow() {
  document.querySelectorAll(".workflow [data-mode]").forEach(button => {
    const active = button.dataset.mode === state.mode || (linkMode && button.dataset.mode === "overview");
    button.classList.toggle("active", active);
  });
}

function renderOverlayButtons() {
  document.querySelectorAll("[data-overlay]").forEach(button => {
    const enabled = uiState[overlayStateKeys[button.dataset.overlay]];
    button.classList.toggle("active", enabled);
    button.setAttribute("aria-pressed", String(enabled));
  });
}

function header(title, subtitle, tag = "WORKSPACE") {
  return `<div class="panel-heading"><div><div class="eyebrow">${tag}</div><h1>${title}</h1><p class="panel-subtitle">${subtitle}</p></div></div>`;
}

function renderSidebar() {
  if (state.mode === "editor") {
    sidePanel.innerHTML = editorPanel();
  } else if (state.mode === "overview") {
    sidePanel.innerHTML = stairsPanel();
  } else if (state.mode === "results") {
    sidePanel.innerHTML = resultsPanel();
    requestAnimationFrame(drawChart);
  } else {
    sidePanel.innerHTML = simulationPanel();
  }
}

function editorPanel() {
  const tools = [
    ["WALL", "square", "Wall", "W"],
    ["EXIT", "door-open", "Exit", "E"],
    ["STAIRS", "arrow-up-down", "Stairs", "S"],
    ["EVACUEE", "accessibility", "Evacuee", "A"],
    ["DRONE", "radio", "Drone", "D"],
    ["ERASER", "eraser", "Eraser", "X"]
  ];
  const undoButton = document.querySelector("#undo-button-template").innerHTML
    .replace('data-action="undo"', `data-action="undo"${state.can_undo ? "" : " disabled"}`);
  return `${header("Edit floorplan", "Paint architectural elements and place simulation entities.")}
    <div class="section-label"><span>Map tools</span><span>${activeFloor()?.width ?? 0} × ${activeFloor()?.height ?? 0}</span></div>
    <div class="tool-grid">${tools.map(([tool, icon, label, key]) => `
      <button class="tool-card ${selectedTool === tool ? "active" : ""}" data-tool="${tool}">
        <span class="tool-icon"><i data-lucide="${icon}"></i></span>${label}<span class="tool-shortcut">${key}</span>
      </button>`).join("")}
    </div>
    <div class="divider"></div>
    <div class="section-label"><span>Floor map</span></div>
    <div class="button-row">${undoButton}<button class="button-secondary" data-action="reset"><i data-lucide="rotate-ccw"></i>Reset to loaded map</button></div>
    <p class="panel-subtitle" style="margin-top:11px">Drag to paint. Right-click or choose Eraser to clear cells. Stair links and floor edits are preserved in project files.</p>`;
}

function stairsPanel() {
  const pairs = [];
  for (const floor of state.floors) {
    for (const cluster of floor.clusters) {
      if (cluster.type !== "STAIRS" || !cluster.linked_to) continue;
      const endpoint = `${floor.name} · ${cluster.cells.length} cells`;
      const target = state.floors.find(candidate => candidate.id === cluster.linked_to[0]);
      if (target && floor.id < target.id) {
        pairs.push(`<div class="pair-item"><span>${escapeHtml(endpoint)} <small>↔ ${escapeHtml(target.name)}</small></span><button class="unlink-button" data-action="unlink" data-floor="${floor.id}" data-cluster="${cluster.id}">Unlink</button></div>`);
      }
    }
  }
  return `${header("Link stairwells", "Connect stair clusters across floors to create vertical routes.", "OVERVIEW · CONNECTIVITY")}
    <div class="link-instruction">${selectedStair ? `Selected Floor ${selectedStair.floorId + 1} Stairs — switch to target floor and click destination stairs` : "Select a stair tile on one floor, then select its matching landing on another floor."}</div>
    <div class="section-label"><span>Connected pairs</span><span>${pairs.length}</span></div>
    <div class="pair-card">${pairs.length ? pairs.join("") : `<div class="empty-state">No stair links yet.<br />Choose a stair cluster on the canvas to begin.</div>`}</div>
    <div class="divider"></div>
    <div class="section-label"><span>How to link</span></div>
    <p class="panel-subtitle">Stair tiles are highlighted in indigo. Links pair entire connected stair clusters and are saved with the project.</p>`;
}

function simulationPanel() {
  const telemetry = state.telemetry;
  const playing = !telemetry.paused;
  const total = telemetry.total_evacuees;
  const speedLabels = [0.5, 0.75, 1, 2, 4];
  return `${header("Live simulation", "Real-time movement, evacuation progress, and hazard control.", "SIMULATION · ACTIVE")}
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-label">TICK</div><div class="metric-value">${telemetry.step_count}</div></div>
      <div class="metric-card"><div class="metric-label">EVACUATED</div><div class="metric-value">${telemetry.evacuated_count}<small>/ ${total}</small></div></div>
    </div>
    <div class="metric-card" style="margin-top:8px"><div style="display:flex;justify-content:space-between"><span class="metric-label">EVACUATION PROGRESS</span><span class="metric-label">${telemetry.evacuation_pct}%</span></div><div class="progress-track"><div class="progress-fill" style="width:${Math.min(100, telemetry.evacuation_pct)}%"></div></div></div>
    <div style="margin-top:7px">
      <div class="telemetry-line"><span>Remaining</span><strong>${telemetry.remaining_count}</strong></div>
      <div class="telemetry-line"><span>Active hazards</span><strong>${telemetry.active_hazards}</strong></div>
      <div class="telemetry-line"><span>Current speed</span><strong>${Number(telemetry.speed_multiplier).toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}×</strong></div>
    </div>
    <div class="section-label"><span>Simulation tools</span></div>
    <div class="button-row">
      <button class="button-secondary ${selectedTool === "HAZARD" ? "active" : ""}" data-tool="HAZARD"><i data-lucide="triangle-alert"></i>Place hazard</button>
      <button class="button-secondary ${selectedTool === "SPAWN" ? "active" : ""}" data-tool="SPAWN"><i data-lucide="user-round-plus"></i>Spawn evacuee</button>
    </div>
    <div class="section-label"><span>Playback</span><span>STEP ${telemetry.step_count}</span></div>
    <button class="button-primary playback-primary full-width" data-action="${playing ? "pause" : "start"}"><i data-lucide="${playing ? "pause" : "play"}"></i>${playing ? "Pause simulation" : telemetry.step_count ? "Resume simulation" : "Start simulation"}</button>
    <div class="button-row"><button class="button-secondary" data-action="step"><i data-lucide="step-forward"></i>+1 Step</button><button class="button-secondary" data-action="reset"><i data-lucide="rotate-ccw"></i>Reset</button></div>
    <div class="section-label"><span>Playback speed</span></div>
    <div class="speed-row">${speedLabels.map(speed => `<button class="speed-button ${telemetry.speed_multiplier === speed ? "active" : ""}" data-speed="${speed}">${speed}×</button>`).join("")}</div>
    <div class="divider"></div>
    <button class="button-secondary full-width" data-action="results"><i data-lucide="chart-no-axes-combined"></i>View evacuation results</button>`;
}

function resultsPanel() {
  const telemetry = state.telemetry;
  const average = telemetry.history.length ? Math.round(telemetry.history.at(-1).step) : 0;
  const meanEvacuation = telemetry.average_evacuation_time ?? "—";
  return `${header("Evacuation results", "Performance summary for the current run.", "ANALYTICS · RUN SUMMARY")}
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-label">TOTAL TICKS</div><div class="metric-value">${telemetry.step_count}</div></div>
      <div class="metric-card"><div class="metric-label">EVACUATION RATE</div><div class="metric-value">${telemetry.evacuation_pct}<small>%</small></div></div>
      <div class="metric-card"><div class="metric-label">EVACUATED</div><div class="metric-value">${telemetry.evacuated_count}<small>/ ${telemetry.total_evacuees}</small></div></div>
      <div class="metric-card"><div class="metric-label">AVG. EVACUATION TIME</div><div class="metric-value">${meanEvacuation}<small>ticks</small></div></div>
    </div>
    <div class="section-label"><span>Evacuation curve</span><span>PEOPLE / TICK</span></div>
    <div class="chart-wrap"><canvas id="progressChart"></canvas></div>
    <div class="telemetry-line" style="margin-top:14px"><span>Current step</span><strong>${average}</strong></div>
    <button class="button-secondary full-width" style="margin-top:15px" data-mode="simulate"><i data-lucide="arrow-left"></i>Return to simulation</button>`;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}

function renderCanvas() {
  if (!state || !canvas.clientWidth) return;
  if (view.floorId !== state.active_floor_id) fitFloor();
  const floor = activeFloor();
  if (!floor) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = COLORS.exterior;
  ctx.fillRect(0, 0, width, height);
  ctx.save();
  ctx.translate(view.offsetX, view.offsetY);
  ctx.scale(view.scale, view.scale);
  ctx.fillStyle = COLORS.floor;
  ctx.fillRect(0, 0, floor.width, floor.height);

  for (const cluster of floor.clusters) {
    const isExit = cluster.type === "EXIT";
    const isStairs = cluster.type === "STAIRS";
    if (!uiState.showClusters && !(state.mode === "overview" && (isExit || isStairs))) {
      continue;
    }
    if (isExit) {
      ctx.fillStyle = "rgba(16,185,129,.18)";
    } else if (isStairs) {
      ctx.fillStyle = "rgba(129,140,248,.18)";
    } else if (cluster.is_doorway || cluster.type === "DOOR") {
      ctx.fillStyle = "rgba(251,191,36,.22)";
    } else {
      ctx.fillStyle = Number(cluster.id) % 2 === 0
        ? "rgba(56,189,248,.06)"
        : "rgba(148,163,184,.06)";
    }
    for (const [x, y] of cluster.cells) ctx.fillRect(x, y, 1, 1);
  }

  for (let y = 0; y < floor.height; y++) {
    for (let x = 0; x < floor.width; x++) {
      const cell = canvasCellType(floor, x, y);
      if (cell === CELL.WALL) {
        ctx.fillStyle = COLORS.wall;
        ctx.fillRect(x, y, 1, 1);
      } else if (cell === CELL.EXIT) {
        ctx.fillStyle = COLORS.exit;
        drawRoundedCell(floor, x, y, CELL.EXIT);
      } else if (cell === CELL.STAIRS) {
        ctx.fillStyle = COLORS.stairs;
        drawRoundedCell(floor, x, y, CELL.STAIRS);
      } else if (
        pendingStrokeFloorId === floor.id &&
        pendingStrokeCells.has(`${x},${y}`) &&
        cell === CELL.EMPTY
      ) {
        ctx.fillStyle = COLORS.floor;
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }

  drawWallEdges(floor);
  if (uiState.showGrid && view.scale >= 5) {
    ctx.beginPath();
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1 / view.scale;
    for (let x = 0; x <= floor.width; x++) { ctx.moveTo(x, 0); ctx.lineTo(x, floor.height); }
    for (let y = 0; y <= floor.height; y++) { ctx.moveTo(0, y); ctx.lineTo(floor.width, y); }
    ctx.stroke();
  }

  drawExitBadges(floor);
  drawStairBadges(floor);
  drawSelectedStair(floor);
  drawHazards(floor);
  drawEntities(floor);
  if (uiState.showHeatmap) drawHeatmap(floor);
  ctx.restore();
  requestAnimationFrame(renderCanvas);
}

function canvasCellType(floor, x, y) {
  const key = `${x},${y}`;
  if (pendingStrokeFloorId === floor.id && pendingStrokeCells.has(key)) {
    return pendingStrokeCells.get(key);
  }
  return floor.cells[y][x];
}

function drawRoundedCell(floor, x, y, cellType) {
  const same = (cellX, cellY) =>
    cellX >= 0 && cellY >= 0 && cellX < floor.width && cellY < floor.height &&
    canvasCellType(floor, cellX, cellY) === cellType;
  const radius = .11;
  const corners = [
    !same(x - 1, y) && !same(x, y - 1) ? radius : 0,
    !same(x + 1, y) && !same(x, y - 1) ? radius : 0,
    !same(x + 1, y) && !same(x, y + 1) ? radius : 0,
    !same(x - 1, y) && !same(x, y + 1) ? radius : 0
  ];
  ctx.beginPath();
  ctx.roundRect(x, y, 1, 1, corners);
  ctx.fill();
}

function drawWallEdges(floor) {
  const isWall = (x, y) =>
    x >= 0 && y >= 0 && x < floor.width && y < floor.height &&
    canvasCellType(floor, x, y) === CELL.WALL;
  ctx.beginPath();
  ctx.strokeStyle = "#94A3B8";
  ctx.lineWidth = 1 / view.scale;
  ctx.shadowColor = "rgba(2,6,23,.35)";
  ctx.shadowBlur = 1.5 / view.scale;
  for (let y = 0; y < floor.height; y++) {
    for (let x = 0; x < floor.width; x++) {
      if (!isWall(x, y)) continue;
      if (!isWall(x, y - 1)) { ctx.moveTo(x, y); ctx.lineTo(x + 1, y); }
      if (!isWall(x + 1, y)) { ctx.moveTo(x + 1, y); ctx.lineTo(x + 1, y + 1); }
      if (!isWall(x, y + 1)) { ctx.moveTo(x, y + 1); ctx.lineTo(x + 1, y + 1); }
      if (!isWall(x - 1, y)) { ctx.moveTo(x, y); ctx.lineTo(x, y + 1); }
    }
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawExitBadges(floor) {
  if (view.scale < 13) return;
  ctx.font = `600 ${Math.max(7, 8 / view.scale)}px Inter`;
  ctx.textAlign = "center";
  for (const cluster of floor.clusters.filter(item =>
    item.type === "EXIT" && item.cells.some(([x, y]) => canvasCellType(floor, x, y) === CELL.EXIT)
  )) {
    const [x, y] = cluster.cells[0] || [];
    if (x === undefined) continue;
    ctx.fillStyle = "rgba(6,78,59,.92)";
    ctx.fillRect(x - .25, y - .45, 1.5, .34);
    ctx.fillStyle = "#a7f3d0";
    ctx.fillText("EXIT", x + .5, y - .21);
  }
}

function drawStairBadges(floor) {
  if (view.scale < 12) return;
  ctx.font = `500 ${Math.max(6, 7 / view.scale)}px "JetBrains Mono"`;
  for (const cluster of floor.clusters.filter(item =>
    item.type === "STAIRS" && item.cells.some(([x, y]) => canvasCellType(floor, x, y) === CELL.STAIRS)
  )) {
    const [x, y] = cluster.cells[0] || [];
    if (x === undefined) continue;
    ctx.fillStyle = "rgba(49,46,129,.93)";
    ctx.fillRect(x - .22, y - .43, 1.7, .31);
    ctx.fillStyle = "#e0e7ff";
    const label = cluster.linked_to ? `→ F${cluster.linked_to[0] + 1}` : "STAIRS";
    ctx.fillText(label, x + .62, y - .22);
  }
}

function drawSelectedStair(floor) {
  if (!selectedStair || selectedStair.floorId !== floor.id) return;
  const cells = selectedStair.cells;
  if (!cells.length) return;
  const xs = cells.map(([x]) => x);
  const ys = cells.map(([, y]) => y);
  const centerX = (Math.min(...xs) + Math.max(...xs) + 1) / 2;
  const centerY = (Math.min(...ys) + Math.max(...ys) + 1) / 2;
  const baseRadius = Math.max(
    Math.max(...xs) - Math.min(...xs) + 1,
    Math.max(...ys) - Math.min(...ys) + 1
  ) / 2 + .12;
  const pulse = .07 * (1 + Math.sin(performance.now() / 170));
  ctx.beginPath();
  ctx.ellipse(centerX, centerY, baseRadius + pulse, baseRadius + pulse, 0, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(199,210,254,${.72 + .2 * Math.sin(performance.now() / 170)})`;
  ctx.lineWidth = .09;
  ctx.shadowColor = COLORS.stairs;
  ctx.shadowBlur = .35;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawHazards(floor) {
  const time = performance.now() / 1000;
  for (const [x, y] of floor.temporary_walls) {
    ctx.fillStyle = `rgba(239,68,68,${.35 + .14 * Math.sin(time * 4)})`;
    ctx.fillRect(x - .1, y - .1, 1.2, 1.2);
    ctx.fillStyle = COLORS.hazard;
    ctx.fillRect(x + .12, y + .12, .76, .76);
  }
}

function interpolatedPosition(kind, entity) {
  const item = motion.get(`${kind}${entity.id}`);
  if (!item) return [entity.x, entity.y];
  const t = Math.min(1, (performance.now() - item.start) / 180);
  if (t >= 1) motion.delete(`${kind}${entity.id}`);
  return [item.from[0] + (item.to[0] - item.from[0]) * t, item.from[1] + (item.to[1] - item.from[1]) * t];
}

function drawEntities(floor) {
  const isPendingCell = (x, y) =>
    pendingStrokeFloorId === floor.id && pendingStrokeCells.has(`${x},${y}`);
  if (uiState.showDroneRoutes) {
    for (const drone of state.drones.filter(item => item.floor_id === floor.id)) {
      if (drone.route.length < 2) continue;
      ctx.beginPath();
      ctx.setLineDash([.22, .18]);
      ctx.strokeStyle = "rgba(56,189,248,.45)";
      ctx.lineWidth = .055;
      drone.route.forEach(([x, y], index) => index ? ctx.lineTo(x + .5, y + .5) : ctx.moveTo(x + .5, y + .5));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  for (const evacuee of state.evacuees.filter(item => item.floor_id === floor.id && !isPendingCell(item.x, item.y))) {
    const [x, y] = interpolatedPosition("e", evacuee);
    ctx.beginPath();
    ctx.fillStyle = "rgba(245,158,11,.16)";
    ctx.arc(x + .5, y + .5, .43, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = COLORS.evacuee;
    ctx.strokeStyle = "rgba(69,26,3,.9)";
    ctx.lineWidth = .045;
    ctx.arc(x + .5, y + .5, .23, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  for (const drone of state.drones.filter(item => item.floor_id === floor.id && !isPendingCell(item.x, item.y))) {
    const [x, y] = interpolatedPosition("d", drone);
    ctx.beginPath();
    ctx.strokeStyle = "rgba(56,189,248,.16)";
    ctx.lineWidth = .045;
    ctx.arc(x + .5, y + .5, 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.save();
    ctx.translate(x + .5, y + .5);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = COLORS.drone;
    ctx.shadowColor = COLORS.drone;
    ctx.shadowBlur = .25;
    ctx.fillRect(-.2, -.2, .4, .4);
    ctx.restore();
  }
  if (pendingStrokeFloorId === floor.id) {
    for (const [key, cellType] of pendingStrokeCells) {
      if (cellType !== CELL.EVACUEE && cellType !== CELL.DRONE) continue;
      const [x, y] = key.split(",").map(Number);
      if (cellType === CELL.EVACUEE) {
        ctx.beginPath();
        ctx.fillStyle = COLORS.evacuee;
        ctx.strokeStyle = "rgba(69,26,3,.9)";
        ctx.lineWidth = .045;
        ctx.arc(x + .5, y + .5, .23, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.save();
        ctx.translate(x + .5, y + .5);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = COLORS.drone;
        ctx.fillRect(-.2, -.2, .4, .4);
        ctx.restore();
      }
    }
  }
}

function drawHeatmap(floor) {
  const people = state.evacuees.filter(person => person.floor_id === floor.id);
  for (const person of people) {
    const gradient = ctx.createRadialGradient(person.x + .5, person.y + .5, .1, person.x + .5, person.y + .5, 3.2);
    gradient.addColorStop(0, "rgba(239,68,68,.22)");
    gradient.addColorStop(1, "rgba(239,68,68,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(person.x - 3, person.y - 3, 7, 7);
  }
}

function drawChart() {
  const chart = document.querySelector("#progressChart");
  if (!chart || !state) return;
  const rect = chart.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  chart.width = Math.max(1, Math.round(rect.width * ratio));
  chart.height = Math.max(1, Math.round(rect.height * ratio));
  const c = chart.getContext("2d");
  c.scale(ratio, ratio);
  const width = rect.width, height = rect.height, pad = { l: 28, r: 8, t: 10, b: 20 };
  c.clearRect(0, 0, width, height);
  c.font = '9px "JetBrains Mono"';
  c.fillStyle = "#718198";
  c.strokeStyle = "rgba(148,163,184,.12)";
  const points = state.telemetry.history;
  const maxStep = Math.max(1, ...points.map(point => point.step));
  const maxEvacuated = Math.max(1, state.telemetry.total_evacuees, ...points.map(point => point.evacuated));
  for (let i = 0; i <= 2; i++) {
    const y = pad.t + (height - pad.t - pad.b) * i / 2;
    c.beginPath(); c.moveTo(pad.l, y); c.lineTo(width - pad.r, y); c.stroke();
    c.fillText(String(Math.round(maxEvacuated * (1 - i / 2))), 2, y + 3);
  }
  if (points.length) {
    const coords = points.map(point => [
      pad.l + (width - pad.l - pad.r) * point.step / maxStep,
      height - pad.b - (height - pad.t - pad.b) * point.evacuated / maxEvacuated
    ]);
    const fill = c.createLinearGradient(0, pad.t, 0, height - pad.b);
    fill.addColorStop(0, "rgba(16,185,129,.25)");
    fill.addColorStop(1, "rgba(16,185,129,0)");
    c.beginPath();
    c.moveTo(coords[0][0], height - pad.b);
    coords.forEach(([x, y]) => c.lineTo(x, y));
    c.lineTo(coords.at(-1)[0], height - pad.b);
    c.closePath(); c.fillStyle = fill; c.fill();
    c.beginPath();
    coords.forEach(([x, y], index) => index ? c.lineTo(x, y) : c.moveTo(x, y));
    c.strokeStyle = "#34d399"; c.lineWidth = 2; c.stroke();
  }
  c.fillStyle = "#718198";
  c.fillText("TICKS", width - 35, height - 4);
}

function gridLine(start, end) {
  let [x, y] = start;
  const [endX, endY] = end;
  const dx = Math.abs(endX - x), dy = Math.abs(endY - y);
  const stepX = x < endX ? 1 : -1, stepY = y < endY ? 1 : -1;
  let error = dx - dy;
  const cells = [];
  while (true) {
    cells.push([x, y]);
    if (x === endX && y === endY) break;
    const doubled = 2 * error;
    if (doubled > -dy) { error -= dy; x += stepX; }
    if (doubled < dx) { error += dx; y += stepY; }
  }
  return cells;
}

function canvasCell(event) {
  const rect = canvas.getBoundingClientRect();
  return [
    Math.floor((event.clientX - rect.left - view.offsetX) / view.scale),
    Math.floor((event.clientY - rect.top - view.offsetY) / view.scale)
  ];
}

canvas.addEventListener("pointerdown", event => {
  if (!state) return;
  if (event.button === 1 || (event.button === 0 && spaceHeld)) {
    panning = true;
    panPoint = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
    return;
  }
  if (event.button !== 0 && event.button !== 2) return;
  const cell = canvasCell(event);
  canvas.setPointerCapture(event.pointerId);
  if (state.mode === "editor") {
    const button = event.button === 2 || selectedTool === "ERASER" ? 3 : 1;
    const tool = button === 3 ? (selectedTool === "ERASER" ? "WALL" : selectedTool) : selectedTool;
    stroke = { button, tool, last: cell, cells: new Map() };
    pendingStrokeCells.clear();
    pendingStrokeFloorId = state.active_floor_id;
    addStrokeCells(stroke, cell, cell);
    renderCanvas();
  } else if (state.mode === "overview") {
    const floor = activeFloor();
    const cluster = floor?.clusters.find(item => item.type === "STAIRS" && item.cells.some(([x, y]) => x === cell[0] && y === cell[1]));
    if (!cluster) return toast("Select an indigo stair tile.", true);
    if (!selectedStair) {
      selectedStair = {
        floorId: state.active_floor_id,
        clusterId: cluster.id,
        cells: cluster.cells.map(([x, y]) => [x, y])
      };
      render();
      toast(`Selected Floor ${selectedStair.floorId + 1} Stairs — switch to target floor and click destination stairs`);
    } else {
      if (selectedStair.floorId === state.active_floor_id) return toast("Choose a landing on a different floor.", true);
      send("link_stairs", {
        floor_a: selectedStair.floorId,
        cluster_a: selectedStair.clusterId,
        floor_b: state.active_floor_id,
        cluster_b: cluster.id
      });
      selectedStair = null;
      render();
    }
  } else if (state.mode === "simulate") {
    if (selectedTool === "HAZARD") send("toggle_hazard", { x: cell[0], y: cell[1] });
    else if (selectedTool === "SPAWN") send("spawn_evacuee", { x: cell[0], y: cell[1] });
  }
});

function addStrokeCells(activeStroke, from, to) {
  const previewType = activeStroke.button === 3
    ? CELL.EMPTY
    : CELL[activeStroke.tool];
  for (const [x, y] of gridLine(from, to)) {
    const key = `${x},${y}`;
    activeStroke.cells.set(key, [x, y]);
    pendingStrokeCells.set(key, previewType);
  }
}

canvas.addEventListener("pointermove", event => {
  if (panning && panPoint) {
    view.offsetX += event.clientX - panPoint[0];
    view.offsetY += event.clientY - panPoint[1];
    panPoint = [event.clientX, event.clientY];
    renderCanvas();
    return;
  }
  if (!stroke) return;
  const next = canvasCell(event);
  addStrokeCells(stroke, stroke.last, next);
  stroke.last = next;
  renderCanvas();
});

canvas.addEventListener("pointerup", event => {
  panning = false;
  panPoint = null;
  if (stroke) {
    const sent = send("edit_cells", { cells: [...stroke.cells.values()], button: stroke.button, tool: stroke.tool });
    if (!sent) {
      pendingStrokeCells.clear();
      pendingStrokeFloorId = null;
    }
    stroke = null;
  }
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("pointercancel", () => {
  panning = false;
  panPoint = null;
  stroke = null;
  pendingStrokeCells.clear();
  pendingStrokeFloorId = null;
});
canvas.addEventListener("contextmenu", event => event.preventDefault());
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left, my = event.clientY - rect.top;
  const gridX = (mx - view.offsetX) / view.scale, gridY = (my - view.offsetY) / view.scale;
  const nextScale = Math.max(2, Math.min(48, view.scale * Math.exp(-event.deltaY * .001)));
  view.offsetX = mx - gridX * nextScale;
  view.offsetY = my - gridY * nextScale;
  view.scale = nextScale;
  renderCanvas();
}, { passive: false });

document.addEventListener("keydown", event => {
  if (event.code === "Space" && !event.repeat) { spaceHeld = true; event.preventDefault(); }
  const target = event.target;
  if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && state?.mode === "editor") {
    event.preventDefault();
    send("undo");
    return;
  }
  const panDirections = {
    ArrowUp: [0, 1],
    ArrowDown: [0, -1],
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0]
  };
  if (panDirections[event.key]) {
    const [xDirection, yDirection] = panDirections[event.key];
    const distance = event.shiftKey ? 120 : 48;
    view.offsetX += xDirection * distance;
    view.offsetY += yDirection * distance;
    event.preventDefault();
    renderCanvas();
    return;
  }
  const key = event.key.toLowerCase();
  const tools = { w: "WALL", e: "EXIT", s: "STAIRS", a: "EVACUEE", d: "DRONE", x: "ERASER" };
  if (tools[key] && state?.mode === "editor") { selectedTool = tools[key]; renderSidebar(); }
});
document.addEventListener("keyup", event => { if (event.code === "Space") spaceHeld = false; });

document.addEventListener("click", async event => {
  const modeButton = event.target.closest("[data-mode]");
  if (modeButton) {
    const mode = modeButton.dataset.mode;
    linkMode = mode === "overview";
    selectedStair = null;
    send("set_mode", { mode });
    return;
  }
  const floorButton = event.target.closest("[data-floor]");
  if (floorButton) {
    try { await fetch(`/api/floors/${floorButton.dataset.floor}/select`, { method: "POST" }); }
    catch { toast("Could not switch floors.", true); }
    return;
  }
  const overlay = event.target.closest("[data-overlay]");
  if (overlay) {
    const key = overlayStateKeys[overlay.dataset.overlay];
    uiState[key] = !uiState[key];
    renderOverlayButtons();
    renderCanvas();
    return;
  }
  const toolButton = event.target.closest("[data-tool]");
  if (toolButton) {
    selectedTool = toolButton.dataset.tool;
    renderSidebar();
    return;
  }
  const speedButton = event.target.closest("[data-speed]");
  if (speedButton) {
    const action = state.telemetry.paused ? "pause" : "start";
    send("playback", { action, speed: Number(speedButton.dataset.speed) });
    return;
  }
  const button = event.target.closest("[data-action]");
  if (!button) return;
  switch (button.dataset.action) {
    case "upload": document.querySelector("#uploadInput").click(); break;
    case "save": await saveProject(); break;
    case "load": document.querySelector("#loadInput").click(); break;
    case "start": send("playback", { action: "start" }); break;
    case "pause": send("playback", { action: "pause" }); break;
    case "step": send("playback", { action: "step" }); break;
    case "reset": send("playback", { action: "reset" }); break;
    case "undo": send("undo"); break;
    case "results": send("set_mode", { mode: "results" }); break;
    case "unlink": send("unlink_stairs", { floor_id: Number(button.dataset.floor), cluster_id: Number(button.dataset.cluster) }); break;
  }
});

async function saveProject() {
  try {
    const response = await fetch("/api/project/save", { method: "POST" });
    if (!response.ok) throw new Error(await response.text());
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "evacuation-project.json"; anchor.click();
    URL.revokeObjectURL(url);
    toast("Project saved.");
  } catch (error) { toast(error.message || "Could not save project.", true); }
}

document.querySelector("#uploadInput").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  const form = new FormData(); form.append("file", file);
  try {
    const response = await fetch("/api/floors/upload", { method: "POST", body: form });
    if (!response.ok) throw new Error((await response.json()).detail || "Upload failed.");
    const result = await response.json(); updateState(result.state); toast(`${file.name} added as Floor ${result.floor_id + 1}.`);
  } catch (error) { toast(error.message, true); }
  event.target.value = "";
});

document.querySelector("#loadInput").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const response = await fetch("/api/project/load", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: await file.text()
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Load failed.");
    updateState(result); toast("Project loaded.");
  } catch (error) { toast(error.message, true); }
  event.target.value = "";
});

window.addEventListener("resize", resizeCanvas);
connect();
resizeCanvas();
