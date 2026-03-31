const state = {
  formations: {},
  videos: [],
  commentaryMode: "live",
  currentCommentaryMode: "live",
  commentaryCursor: -1,
  commentaryQueue: [],
  commentaryAudio: null,
  teams: [
    {
      team_name: "Equipo 1",
      team_color: "",
      formation: "4-3-3",
      players_by_slot: {},
    },
    {
      team_name: "Equipo 2",
      team_color: "",
      formation: "4-3-3",
      players_by_slot: {},
    },
  ],
  currentRunId: null,
  pollTimer: null,
};

const teamsGrid = document.getElementById("teams-grid");
const videoSelect = document.getElementById("video-source");
const commentaryModeSelect = document.getElementById("commentary-mode");
const runBadge = document.getElementById("run-badge");
const runSummary = document.getElementById("run-summary");
const runLog = document.getElementById("run-log");
const launchButton = document.getElementById("launch-run");
const teamTemplate = document.getElementById("team-card-template");

function ensureSlots(teamState) {
  const formation = state.formations[teamState.formation];
  if (!formation) {
    return;
  }
  const nextPlayers = {};
  for (const slot of formation.ui_slots) {
    nextPlayers[slot] = teamState.players_by_slot?.[slot] || "";
  }
  teamState.players_by_slot = nextPlayers;
}

function renderVideoOptions() {
  videoSelect.innerHTML = "";
  for (const video of state.videos) {
    const option = document.createElement("option");
    option.value = video.key;
    option.textContent = `${video.key} · ${video.path}`;
    videoSelect.appendChild(option);
  }
}

function createSlotCard(teamIndex, slot, coords, value) {
  const slotCard = document.createElement("div");
  slotCard.className = "slot-card";
  slotCard.style.left = `${coords.x}%`;
  slotCard.style.top = `${coords.y}%`;

  const slotLabel = document.createElement("span");
  slotLabel.className = "slot-label";
  slotLabel.textContent = slot;

  const slotInput = document.createElement("input");
  slotInput.className = "slot-input";
  slotInput.type = "text";
  slotInput.placeholder = "Nombre del jugador";
  slotInput.value = value || "";

  slotCard.addEventListener("click", () => {
    slotInput.focus();
    slotInput.select();
  });
  slotInput.addEventListener("focus", () => {
    slotCard.classList.add("is-active");
  });
  slotInput.addEventListener("blur", () => {
    slotCard.classList.remove("is-active");
  });
  slotInput.addEventListener("click", (event) => {
    event.stopPropagation();
  });
  slotInput.addEventListener("input", (event) => {
    state.teams[teamIndex].players_by_slot[slot] = event.target.value;
  });

  slotCard.appendChild(slotLabel);
  slotCard.appendChild(slotInput);
  return slotCard;
}

function appendPitchMarkings(pitch) {
  const markings = document.createElement("div");
  markings.className = "pitch-markings";

  const outline = document.createElement("div");
  outline.className = "pitch-outline";
  markings.appendChild(outline);

  const halfLine = document.createElement("div");
  halfLine.className = "pitch-half-line";
  markings.appendChild(halfLine);

  const centerCircle = document.createElement("div");
  centerCircle.className = "pitch-center-circle";
  markings.appendChild(centerCircle);

  const centerSpot = document.createElement("div");
  centerSpot.className = "pitch-center-spot";
  markings.appendChild(centerSpot);

  const topPenalty = document.createElement("div");
  topPenalty.className = "penalty-box penalty-box-top";
  markings.appendChild(topPenalty);

  const bottomPenalty = document.createElement("div");
  bottomPenalty.className = "penalty-box penalty-box-bottom";
  markings.appendChild(bottomPenalty);

  const topSixYard = document.createElement("div");
  topSixYard.className = "six-yard-box six-yard-box-top";
  markings.appendChild(topSixYard);

  const bottomSixYard = document.createElement("div");
  bottomSixYard.className = "six-yard-box six-yard-box-bottom";
  markings.appendChild(bottomSixYard);

  const topSpot = document.createElement("div");
  topSpot.className = "penalty-spot penalty-spot-top";
  markings.appendChild(topSpot);

  const bottomSpot = document.createElement("div");
  bottomSpot.className = "penalty-spot penalty-spot-bottom";
  markings.appendChild(bottomSpot);

  const topArc = document.createElement("div");
  topArc.className = "penalty-arc penalty-arc-top";
  markings.appendChild(topArc);

  const bottomArc = document.createElement("div");
  bottomArc.className = "penalty-arc penalty-arc-bottom";
  markings.appendChild(bottomArc);

  pitch.appendChild(markings);
}

function renderTeams() {
  teamsGrid.innerHTML = "";

  state.teams.forEach((teamState, teamIndex) => {
    ensureSlots(teamState);
    const fragment = teamTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".team-card");
    const title = fragment.querySelector(".team-title");
    const nameInput = fragment.querySelector(".team-name-input");
    const colorInput = fragment.querySelector(".team-color-input");
    const formationSelect = fragment.querySelector(".formation-select");
    const pitch = fragment.querySelector('[data-role="pitch"]');

    title.textContent = `Equipo ${teamIndex + 1}`;
    nameInput.value = teamState.team_name;
    colorInput.value = teamState.team_color;

    for (const formationName of Object.keys(state.formations)) {
      const option = document.createElement("option");
      option.value = formationName;
      option.textContent = formationName;
      if (formationName === teamState.formation) {
        option.selected = true;
      }
      formationSelect.appendChild(option);
    }

    nameInput.addEventListener("input", (event) => {
      state.teams[teamIndex].team_name = event.target.value;
    });
    colorInput.addEventListener("input", (event) => {
      state.teams[teamIndex].team_color = event.target.value;
    });
    formationSelect.addEventListener("change", (event) => {
      state.teams[teamIndex].formation = event.target.value;
      ensureSlots(state.teams[teamIndex]);
      renderTeams();
    });

    const formation = state.formations[teamState.formation];
    appendPitchMarkings(pitch);
    for (const slot of formation.ui_slots) {
      const coords = formation.pitch_layout[slot];
      pitch.appendChild(
        createSlotCard(
          teamIndex,
          slot,
          coords,
          teamState.players_by_slot?.[slot] || "",
        ),
      );
    }

    teamsGrid.appendChild(card);
  });
}

function setRunStatus(status, text) {
  runBadge.className = `status-badge ${status}`;
  runBadge.textContent = text;
}

function renderRunStatus(statusPayload) {
  if (!statusPayload) {
    setRunStatus("idle", "Sin lanzar");
    runSummary.textContent = "Todavía no hay ninguna ejecución activa.";
    runLog.textContent = "Esperando ejecución...";
    return;
  }

  const status = statusPayload.status || "idle";
  const runId = statusPayload.run_id || "-";
  const videoSource = statusPayload.video_source || "-";
  const outputVideoPath = statusPayload.output_video_path || "-";
  const teams = Array.isArray(statusPayload.teams) ? statusPayload.teams : [];
  const commentary = statusPayload.commentary || {};
  setRunStatus(status, status.toUpperCase());
  runSummary.innerHTML = `
    <strong>Run:</strong> ${runId}<br />
    <strong>Vídeo:</strong> ${videoSource}<br />
    <strong>Salida:</strong> ${outputVideoPath}<br />
    <strong>Equipos:</strong> ${teams.map((team) => team.team_name).join(" vs ")}<br />
    <strong>Comentarios:</strong> ${commentary.mode || "-"} · intro ${commentary.intro_status || "-"}<br />
    <strong>Mix diferido:</strong> ${commentary.deferred_mux_status || "-"}<br />
    <strong>Spec:</strong> ${statusPayload.spec_path || "-"}<br />
    <strong>Log:</strong> ${statusPayload.log_path || "-"}<br />
    <strong>Manifest:</strong> ${commentary.manifest_path || "-"}
  `;
  runLog.textContent = (statusPayload.log_tail || []).join("\n") || "Sin salida todavía.";
}

function resetCommentaryPlayback() {
  state.commentaryCursor = -1;
  state.commentaryQueue = [];
  if (state.commentaryAudio) {
    state.commentaryAudio.pause();
    state.commentaryAudio = null;
  }
}

function playNextCommentary() {
  if (state.commentaryAudio || state.commentaryQueue.length === 0) {
    return;
  }
  const nextItem = state.commentaryQueue.shift();
  if (!nextItem?.audio_url) {
    playNextCommentary();
    return;
  }
  const audio = new Audio(nextItem.audio_url);
  state.commentaryAudio = audio;
  const releaseAudio = () => {
    state.commentaryAudio = null;
    playNextCommentary();
  };
  audio.addEventListener("ended", releaseAudio, { once: true });
  audio.addEventListener("error", releaseAudio, { once: true });
  audio.play().catch(() => {
    releaseAudio();
  });
}

function enqueueCommentaryEvents(events) {
  for (const eventItem of events) {
    if (eventItem?.audio_url) {
      state.commentaryQueue.push(eventItem);
    }
  }
  playNextCommentary();
}

async function pollCommentaryEvents() {
  if (!state.currentRunId || state.currentCommentaryMode !== "live") {
    return;
  }
  const response = await fetch(
    `/api/runs/${state.currentRunId}/commentary-events?after=${state.commentaryCursor}`,
  );
  const payload = await response.json();
  const events = Array.isArray(payload.events) ? payload.events : [];
  if (events.length > 0) {
    state.commentaryCursor = events[events.length - 1].index;
    enqueueCommentaryEvents(events);
  }
}

async function pollRunStatus() {
  if (!state.currentRunId) {
    return;
  }
  const response = await fetch(`/api/runs/${state.currentRunId}`);
  const payload = await response.json();
  renderRunStatus(payload);
  if (state.currentCommentaryMode === "live") {
    await pollCommentaryEvents();
  }
  if (!["queued", "running"].includes(payload.status)) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function submitRun() {
  const payload = {
    video_source: videoSelect.value,
    commentary_mode: state.commentaryMode,
    teams: state.teams.map((teamState) => ({
      team_name: teamState.team_name,
      team_color: teamState.team_color,
      formation: teamState.formation,
      players_by_slot: teamState.players_by_slot,
    })),
  };

  launchButton.disabled = true;
  launchButton.textContent = "Lanzando tracking...";

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "No se pudo lanzar la ejecución.");
    }
    resetCommentaryPlayback();
    state.currentRunId = result.run_id;
    state.currentCommentaryMode = result.commentary?.mode || state.commentaryMode;
    const introEventIndex = Number(result.commentary?.intro_event_index);
    if (Number.isFinite(introEventIndex)) {
      state.commentaryCursor = introEventIndex;
    }
    renderRunStatus(result);
    if (
      state.currentCommentaryMode === "live" &&
      result.commentary?.intro_audio_url
    ) {
      enqueueCommentaryEvents([
        {
          audio_url: result.commentary.intro_audio_url,
        },
      ]);
    }
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
    }
    state.pollTimer = setInterval(() => {
      pollRunStatus().catch((error) => {
        runLog.textContent = error.message;
      });
    }, 2500);
  } catch (error) {
    setRunStatus("failed", "ERROR");
    runSummary.textContent = error.message;
    runLog.textContent = String(error.stack || error.message || error);
  } finally {
    launchButton.disabled = false;
    launchButton.textContent = "Guardar alineaciones y ejecutar";
  }
}

async function loadBootstrapData() {
  const [formationsResponse, videosResponse] = await Promise.all([
    fetch("/api/formations"),
    fetch("/api/videos"),
  ]);
  const formationsPayload = await formationsResponse.json();
  const videosPayload = await videosResponse.json();

  state.formations = formationsPayload.formations || {};
  state.videos = videosPayload.videos || [];
  commentaryModeSelect.value = state.commentaryMode;

  for (const teamState of state.teams) {
    ensureSlots(teamState);
  }

  renderVideoOptions();
  renderTeams();
  renderRunStatus(null);
}

launchButton.addEventListener("click", () => {
  submitRun().catch((error) => {
    runSummary.textContent = error.message;
  });
});

commentaryModeSelect.addEventListener("change", (event) => {
  state.commentaryMode = event.target.value;
});

loadBootstrapData().catch((error) => {
  setRunStatus("failed", "ERROR");
  runSummary.textContent = error.message;
  runLog.textContent = String(error.stack || error.message || error);
});
