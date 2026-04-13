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
      team_name: "",
      team_color: "",
      formation: "4-3-3",
      players_by_slot: {},
    },
    {
      team_name: "",
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
const nameGateBanner = document.getElementById("name-gate-banner");
const resultVideo = document.getElementById("result-video");
const resultVideoEmpty = document.getElementById("result-video-empty");
const teamTemplate = document.getElementById("team-card-template");
const DEFAULT_RESULT_VIDEO_EMPTY_TEXT =
  "El vídeo final aparecerá aquí cuando termine la ejecución y la interfaz tenga el MP4 listo.";

function normalizedTeamName(teamState) {
  return String(teamState?.team_name || "").trim();
}

function teamNamesReady() {
  return state.teams.every((teamState) => normalizedTeamName(teamState));
}

function syncFormAvailability() {
  const namesReady = teamNamesReady();
  const hasVideo = Boolean(videoSelect.value);

  videoSelect.disabled = !namesReady;
  commentaryModeSelect.disabled = !namesReady;
  launchButton.disabled = !namesReady || !hasVideo;

  if (!namesReady) {
    launchButton.textContent = "Primero escribe los dos equipos";
    nameGateBanner.classList.remove("is-hidden");
    return;
  }

  launchButton.textContent = "Guardar alineaciones y ejecutar";
  nameGateBanner.classList.add("is-hidden");
}

function setResultVideoPlaceholder(message = DEFAULT_RESULT_VIDEO_EMPTY_TEXT) {
  resultVideoEmpty.textContent = message;
}

function describeResultVideoError() {
  const mediaError = resultVideo.error;
  if (!mediaError) {
    return "El navegador no ha podido reproducir el vídeo final.";
  }
  switch (mediaError.code) {
    case MediaError.MEDIA_ERR_ABORTED:
      return "La carga del vídeo final se ha cancelado antes de terminar.";
    case MediaError.MEDIA_ERR_NETWORK:
      return "Ha fallado la descarga del vídeo final desde la interfaz.";
    case MediaError.MEDIA_ERR_DECODE:
      return "El navegador ha rechazado el MP4 final al intentar decodificarlo.";
    case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
      return "El navegador no soporta el formato del MP4 final servido por la interfaz.";
    default:
      return "El navegador no ha podido reproducir el vídeo final.";
  }
}

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

function createSlotCard(teamIndex, slot, coords, value, disabled) {
  const slotCard = document.createElement("div");
  slotCard.className = "slot-card";
  slotCard.style.left = `${coords.x}%`;
  slotCard.style.top = `${coords.y}%`;
  if (disabled) {
    slotCard.classList.add("is-disabled");
  }

  const slotLabel = document.createElement("span");
  slotLabel.className = "slot-label";
  slotLabel.textContent = slot;

  const slotInput = document.createElement("input");
  slotInput.className = "slot-input";
  slotInput.type = "text";
  slotInput.placeholder = disabled
    ? "Desbloquea con los equipos"
    : "Nombre del jugador";
  slotInput.value = value || "";
  slotInput.disabled = Boolean(disabled);

  slotCard.addEventListener("click", () => {
    if (disabled) {
      return;
    }
    slotInput.focus();
    slotInput.select();
  });
  slotInput.addEventListener("focus", () => {
    if (disabled) {
      return;
    }
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
  const namesReady = teamNamesReady();

  state.teams.forEach((teamState, teamIndex) => {
    ensureSlots(teamState);
    const fragment = teamTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".team-card");
    const title = fragment.querySelector(".team-title");
    const nameInput = fragment.querySelector(".team-name-input");
    const colorInput = fragment.querySelector(".team-color-input");
    const formationSelect = fragment.querySelector(".formation-select");
    const pitchShell = fragment.querySelector(".pitch-shell");
    const pitch = fragment.querySelector('[data-role="pitch"]');

    title.textContent = `Equipo ${teamIndex + 1}`;
    nameInput.value = teamState.team_name;
    nameInput.placeholder = `Escribe el equipo ${teamIndex + 1}`;
    colorInput.value = teamState.team_color;
    colorInput.disabled = !namesReady;
    formationSelect.disabled = !namesReady;
    if (!namesReady) {
      card.classList.add("is-locked");
      const lockNote = document.createElement("p");
      lockNote.className = "team-lock-note";
      lockNote.textContent =
        "Completa el nombre de los dos equipos para desbloquear colores, formación y jugadores.";
      pitchShell.before(lockNote);
    }

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
    const rerenderIfNameGateChanged = () => {
      syncFormAvailability();
      if (namesReady !== teamNamesReady()) {
        renderTeams();
      }
    };
    nameInput.addEventListener("change", rerenderIfNameGateChanged);
    nameInput.addEventListener("blur", rerenderIfNameGateChanged);
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
          !namesReady,
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

function renderResultVideo(statusPayload) {
  const resultVideoUrl = statusPayload?.result_video_url || "";
  if (!resultVideoUrl) {
    resultVideo.pause();
    resultVideo.removeAttribute("src");
    resultVideo.dataset.src = "";
    resultVideo.load();
    resultVideo.hidden = true;
    setResultVideoPlaceholder();
    resultVideoEmpty.hidden = false;
    return;
  }

  const cacheBuster = encodeURIComponent(statusPayload?.updated_at_utc || Date.now());
  const videoUrl = `${resultVideoUrl}?v=${cacheBuster}`;
  if (resultVideo.dataset.src !== videoUrl) {
    resultVideo.pause();
    setResultVideoPlaceholder("Cargando el vídeo final...");
    resultVideo.src = videoUrl;
    resultVideo.dataset.src = videoUrl;
    resultVideo.load();
  }
  resultVideo.hidden = false;
  resultVideoEmpty.hidden = true;
}

function renderRunStatus(statusPayload) {
  if (!statusPayload) {
    setRunStatus("idle", "Sin lanzar");
    runSummary.textContent = "Todavía no hay ninguna ejecución activa.";
    runLog.textContent = "Esperando ejecución...";
    renderResultVideo(null);
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
  renderResultVideo(statusPayload);
}

function resetCommentaryPlayback() {
  state.commentaryCursor = -1;
  state.commentaryQueue = [];
  if (state.commentaryAudio) {
    state.commentaryAudio.pause();
    state.commentaryAudio = null;
  }
}

function isInterruptingCommentary(eventItem) {
  const metadata = eventItem?.metadata || {};
  return Boolean(eventItem?.interrupt_audio || metadata.interrupt_audio);
}

function stopCurrentCommentary() {
  if (!state.commentaryAudio) {
    return;
  }
  state.commentaryAudio.pause();
  state.commentaryAudio = null;
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
    if (state.commentaryAudio === audio) {
      state.commentaryAudio = null;
    }
    playNextCommentary();
  };
  audio.addEventListener("ended", releaseAudio, { once: true });
  audio.addEventListener("error", releaseAudio, { once: true });
  audio.play().catch(() => {
    releaseAudio();
  });
}

function enqueueCommentaryEvents(events) {
  const playableEvents = [...events].filter(
    (eventItem) => eventItem?.audio_url && !eventItem?.text_only,
  );
  if (playableEvents.length === 0) {
    return;
  }

  if (state.commentaryAudio) {
    const interruptItem = [...playableEvents]
      .reverse()
      .find((eventItem) => isInterruptingCommentary(eventItem));
    if (interruptItem) {
      stopCurrentCommentary();
      state.commentaryQueue = [interruptItem];
      playNextCommentary();
    }
    return;
  }

  const nextItem = [...playableEvents].reverse()[0];
  if (!nextItem) {
    return;
  }
  state.commentaryQueue = [nextItem];
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
  if (!teamNamesReady()) {
    throw new Error("Primero completa el nombre de los dos equipos.");
  }
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
  syncFormAvailability();
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

videoSelect.addEventListener("change", () => {
  syncFormAvailability();
});

resultVideo.addEventListener("loadeddata", () => {
  resultVideo.hidden = false;
  resultVideoEmpty.hidden = true;
});

resultVideo.addEventListener("error", () => {
  resultVideo.pause();
  resultVideo.hidden = true;
  setResultVideoPlaceholder(describeResultVideoError());
  resultVideoEmpty.hidden = false;
});

loadBootstrapData().catch((error) => {
  setRunStatus("failed", "ERROR");
  runSummary.textContent = error.message;
  runLog.textContent = String(error.stack || error.message || error);
});
