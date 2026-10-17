const views = ["chat", "code", "models", "images", "library", "tools", "automations", "settings", "iris"];
const conversation = [];
let toastTimer;
let activeSessionId = null;
let sessionImages = [];
let sessionState = {};
let libraryFiles = [];
let showArchivedSessions = false;

const elements = {
  connectionPill: document.querySelector("#connection-pill"),
  connectionLabel: document.querySelector("#connection-label"),
  activeModelName: document.querySelector("#active-model-name"),
  activeModelProvider: document.querySelector("#active-model-provider"),
  modelSelect: document.querySelector("#model-select"),
  modelGrid: document.querySelector("#model-grid"),
  importGrid: document.querySelector("#import-grid"),
  imageEngine: document.querySelector("#image-engine"),
  imageCheckpoint: document.querySelector("#image-checkpoint"),
  imageLoraList: document.querySelector("#image-lora-list"),
  imageResult: document.querySelector("#image-result"),
  imageHistory: document.querySelector("#image-history"),
  modelCount: document.querySelector("#model-count"),
  hfRepo: document.querySelector("#hf-repo"),
  hfFile: document.querySelector("#hf-file"),
  hfListFiles: document.querySelector("#hf-list-files"),
  hfDownload: document.querySelector("#hf-download"),
  downloadStatus: document.querySelector("#model-download-status"),
  ggufFile: document.querySelector("#gguf-file"),
  ggufChoose: document.querySelector("#gguf-choose"),
  ggufFileName: document.querySelector("#gguf-file-name"),
  ggufAdd: document.querySelector("#gguf-add"),
  runtimeTitle: document.querySelector("#runtime-title"),
  runtimeCopy: document.querySelector("#runtime-copy"),
  runtimeState: document.querySelector("#runtime-state"),
  runtimeBanner: document.querySelector("#runtime-banner"),
  settingsRuntime: document.querySelector("#settings-runtime"),
  welcome: document.querySelector("#welcome-block"),
  messages: document.querySelector("#message-list"),
  input: document.querySelector("#chat-input"),
};

function switchView(name) {
  if (!views.includes(name)) return;
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.querySelector(`#view-${name}`).classList.add("active");
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  const title = name === "automations" ? "Automations" : name.charAt(0).toUpperCase() + name.slice(1);
  document.querySelector("#page-name").textContent = title;
  document.title = `${title} · AI Command Center`;
  if (name === "models") refreshModels();
  if (name === "images") refreshImageCatalog();
  if (name === "library") refreshLibrary();
  if (name === "tools") refreshLibrary();
  if (name === "iris") irisShowHome();
}

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});

function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 3200);
}

function formatSize(bytes) {
  if (!bytes) return "Size unavailable";
  const gib = bytes / (1024 ** 3);
  return `${gib >= 1 ? gib.toFixed(1) : (bytes / (1024 ** 2)).toFixed(0)} ${gib >= 1 ? "GB" : "MB"}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[char]);
}

function setRuntimeStatus(available, installed = true) {
  const ready = available || installed;
  elements.connectionPill.classList.toggle("connected", ready);
  elements.connectionPill.classList.toggle("disconnected", !ready);
  elements.connectionLabel.textContent = ready ? "In-app engine ready" : "Model engine setting up";
  elements.runtimeBanner.classList.toggle("is-connected", ready);
  elements.runtimeTitle.textContent = available ? "Local model is running" : installed ? "In-app model engine ready" : "Model engine setting up";
  elements.runtimeCopy.textContent = available
    ? "The selected local model is running inside this app."
    : installed ? "Choose a model and send a message to load it into the local engine."
      : "The in-app model engine is not available yet. Restart the app after setup finishes.";
  elements.runtimeState.textContent = ready ? "READY" : "SETTING UP";
  elements.settingsRuntime.textContent = ready ? "Ready" : "Setting up";
}

async function refreshModels() {
  elements.modelGrid.innerHTML = '<div class="empty-card">Checking for installed models…</div>';
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    const data = await response.json();
    setRuntimeStatus(Boolean(data.available), Boolean(data.runtime_installed));
    renderModels(data.models || []);
    renderImportable(data.importable || [], data.models || []);
  } catch {
    setRuntimeStatus(false, false);
    renderModels([]);
    renderImportable([], []);
  }
}

async function refreshImageCatalog() {
  const banner = document.querySelector("#image-runtime-banner");
  try {
    const response = await fetch("/api/image-catalog", { cache: "no-store" });
    const data = await response.json();
    const checkpoints = data.checkpoints || [];
    const loras = data.loras || [];
    const ready = Boolean(data.engine_ready);
    const dockerCpu = Boolean(data.docker_cpu);
    document.querySelector("#image-runtime-title").textContent = checkpoints.length ? "Local image models found" : "No compatible image models found";
    const pytorch = data.pytorch || {};
    const engines = data.engines || {};
    const mpsReady = !!(engines.pytorch && engines.pytorch.ready && pytorch.mps_available);
    const pytorchReady = !!(engines.pytorch && engines.pytorch.ready);
    const sdReady = !!(engines.sd_cpp && engines.sd_cpp.ready) || !!data.sd_cpp_ready;
    let engineNote = "";
    if (mpsReady) engineNote = " PyTorch MPS is available (preferred for reliability).";
    else if (pytorchReady) engineNote = " PyTorch CPU is available; generation may be slow.";
    else if (pytorch.message) engineNote = ` ${pytorch.message}`;
    if (pytorch.hint) engineNote += ` ${pytorch.hint}`;
    if (sdReady) engineNote += " stable-diffusion.cpp is available.";
    document.querySelector("#image-runtime-copy").textContent = ready
      ? `${checkpoints.length} checkpoints and ${loras.length} LoRAs are available.${engineNote} ${dockerCpu ? "Docker uses CPU image generation; start with a smaller image and fewer steps. " : ""}Image runs unload the chat model to free memory; the next chat message reloads it.`
      : "No image engine is ready yet. On native macOS install PyTorch (see README) or finish sd.cpp setup with run.command.";
    document.querySelector("#image-runtime-state").textContent = ready ? (mpsReady ? "MPS READY" : pytorchReady ? "CPU READY" : "READY") : "SETTING UP";
    if (elements.imageEngine) {
      const autoOpt = elements.imageEngine.querySelector('option[value="auto"]');
      const ptOpt = elements.imageEngine.querySelector('option[value="pytorch"]');
      const sdOpt = elements.imageEngine.querySelector('option[value="sd_cpp"]');
      if (ptOpt) {
        ptOpt.disabled = !pytorchReady;
        ptOpt.textContent = mpsReady ? "PyTorch MPS" : "PyTorch CPU (slow)";
      }
      if (sdOpt) sdOpt.disabled = !sdReady;
      if (autoOpt) autoOpt.textContent = mpsReady
        ? "Auto (prefer PyTorch MPS when available)"
        : "Auto (sd.cpp — install PyTorch for MPS)";
    }
    banner.classList.toggle("is-connected", ready);
    elements.imageCheckpoint.innerHTML = checkpoints.length
      ? checkpoints.map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.name)} · ${escapeHtml(item.location || "Local models")} · ${escapeHtml(formatSize(item.size))}</option>`).join("")
      : '<option value="">No image models found</option>';
    document.querySelector("#lora-count").textContent = `${loras.length} found`;
    elements.imageLoraList.innerHTML = loras.length
      ? loras.map((item, index) => `<label class="lora-option"><input type="checkbox" value="${escapeHtml(item.path)}" id="lora-${index}"><span><strong>${escapeHtml(item.name)} · ${escapeHtml(item.location || "Local")}</strong><small>${escapeHtml(formatSize(item.size))}</small></span></label>`).join("")
      : '<div class="empty-card lora-empty">No LoRAs found in the local image folders.</div>';
    applySessionState();
  } catch {
    document.querySelector("#image-runtime-title").textContent = "Could not read local image models";
    document.querySelector("#image-runtime-copy").textContent = "Check local model folders and refresh.";
    document.querySelector("#image-runtime-state").textContent = "OFFLINE";
    elements.imageCheckpoint.innerHTML = '<option value="">No image models found</option>';
    elements.imageLoraList.innerHTML = '<div class="empty-card lora-empty">Could not read the local model folders.</div>';
  }
}

document.querySelector("#refresh-image-catalog").addEventListener("click", refreshImageCatalog);
document.querySelector("#image-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.querySelector("#generate-image");
  const prompt = document.querySelector("#image-prompt").value.trim();
  const checkpoint = elements.imageCheckpoint.value;
  const seedMode = document.querySelector("#seed-mode").value;
  if (!checkpoint || !prompt) return;
  button.disabled = true;
  button.innerHTML = "Starting local image engine…";
  elements.imageResult.innerHTML = '<div class="image-result-placeholder generating"><span class="image-spinner"></span><strong>Loading your model in the app</strong><p>The first model load can take a few minutes. Generation speed depends on the model and image size.</p></div>';
  try {
    button.textContent = "Generating image…";
    const response = await fetch("/api/images/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        checkpoint,
        engine: elements.imageEngine ? elements.imageEngine.value : "auto",
        loras: [...elements.imageLoraList.querySelectorAll("input:checked")].map((input) => input.value),
        lora_strength: Number(document.querySelector("#lora-strength").value),
        session_id: activeSessionId,
        prompt,
        negative: document.querySelector("#image-negative").value,
        seed_mode: seedMode,
        seed: Number(document.querySelector("#image-seed").value),
        width: Number(document.querySelector("#image-width").value),
        height: Number(document.querySelector("#image-height").value),
        steps: Number(document.querySelector("#image-steps").value),
        cfg: Number(document.querySelector("#image-cfg").value),
        sampler: document.querySelector("#image-sampler").value,
        scheduler: document.querySelector("#image-scheduler").value,
        count: Number(document.querySelector("#image-count").value),
        pdf_tools: selectedChatTools(),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Image generation failed.");
    elements.imageResult.innerHTML = `<div class="generated-image-grid">${data.images.map((image) => `<article><a href="${escapeHtml(image.url)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(image.url)}" alt="Generated image"></a><div class="image-result-caption">${escapeHtml(image.filename)} · seed ${escapeHtml(image.seed)}</div></article>`).join("")}</div><div class="image-result-caption">${escapeHtml(data.checkpoint)} · ${data.width} × ${data.height} · ${data.steps} steps · ${escapeHtml(data.sampler)} · ${escapeHtml(data.scheduler === "model" ? "model scheduler" : data.scheduler)}</div>`;
    if (seedMode === "increment") document.querySelector("#image-seed").value = String((data.images.at(-1).seed + 1) % 4294967296);
    document.querySelector("#image-run-summary").textContent = `${data.width} × ${data.height} · ${data.steps} steps · ${data.sampler} · ${data.scheduler === "model" ? "model scheduler" : data.scheduler} · seed ${seedMode}`;
    await saveActiveSession();
    await refreshActiveSession();
    showToast("Image generated and saved on this Mac");
  } catch (error) {
    elements.imageResult.innerHTML = `<div class="image-result-placeholder"><span>!</span><strong>Image generation didn’t finish</strong><p>${escapeHtml(error.message || "Check the app's local image engine log and try again.")}</p></div>`;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Generate image <span>↗</span>';
  }
});

document.querySelector("#lora-strength").addEventListener("input", (event) => {
  document.querySelector("#lora-strength-value").value = Number(event.target.value).toFixed(2);
});

function renderImportable(importable, installed) {
  const installedPaths = new Set(installed.map((model) => model.path));
  const available = importable.filter((item) => !installedPaths.has(item.path));
  if (!available.length) {
    elements.importGrid.innerHTML = '<div class="empty-card empty-models"><strong>No GGUF models waiting to be added</strong><p>New files from LM Studio and this app’s model folder will appear here.</p></div>';
    return;
  }
  elements.importGrid.innerHTML = available.map((model) => `<article class="model-card import-card">
    <div class="model-card-top"><span class="model-emblem">⇩</span><span class="model-ready import-source">${escapeHtml((model.source || "ON THIS MAC").toUpperCase())}</span></div>
    <h3>${escapeHtml(model.name)}</h3><p class="model-family">${escapeHtml(model.file)}</p>
    <div class="model-meta"><span>${escapeHtml(formatSize(model.size))}</span><span>GGUF</span></div>
    <button class="model-use-button" data-import-path="${escapeHtml(model.path)}">Add to app <span>↗</span></button>
  </article>`).join("");
  elements.importGrid.querySelectorAll("[data-import-path]").forEach((button) => {
    button.addEventListener("click", async () => {
      const original = button.innerHTML;
      button.disabled = true;
      button.textContent = "Adding model…";
      try {
        const response = await fetch("/api/models/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: button.dataset.importPath }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Model import failed.");
        showToast(`${result.model} added to the app`);
        await refreshModels();
      } catch (error) {
        showToast(error.message || "Model import failed.");
        button.disabled = false;
        button.innerHTML = original;
      }
    });
  });
}

function renderModels(models) {
  const coderButton = document.querySelector("#use-coder-model");
  if (coderButton) coderButton.disabled = !models.some((model) => model.name === "Qwen2.5-Coder-14B-Instruct-MLX-4bit");
  elements.modelCount.textContent = `${models.length} ${models.length === 1 ? "model" : "models"}`;
  const previous = elements.modelSelect.value;
  elements.modelSelect.innerHTML = models.length
    ? models.map((model) => `<option value="${escapeHtml(model.name)}" data-backend="${escapeHtml(model.backend || "llama.cpp")}">${escapeHtml(model.name)}</option>`).join("")
    : '<option value="">No local model</option>';
  if (models.some((model) => model.name === previous)) elements.modelSelect.value = previous;
  updateActiveModel();

  if (!models.length) {
    elements.modelGrid.innerHTML = '<div class="empty-card empty-models"><span class="empty-model-mark">◈</span><strong>No models added yet</strong><p>Find a GGUF model on Hugging Face or add one already downloaded on this Mac.</p><span class="empty-card-foot">LOCAL MODELS FOR THIS DEVICE</span></div>';
    return;
  }

  elements.modelGrid.innerHTML = models.map((model, index) => {
    const details = model.details || {};
    const family = model.backend === "mlx" ? "Apple MLX · preferred" : details.family || details.families?.[0] || "Local model";
    const quant = model.backend === "mlx" ? "MLX 4-bit" : details.quantization_level || "Installed";
    return `<article class="model-card ${index === 0 ? "featured-model" : ""}">
      <div class="model-card-top"><span class="model-emblem">◈</span><span class="model-ready"><i></i> READY</span></div>
      <h3>${escapeHtml(model.name)}</h3><p class="model-family">${escapeHtml(family)}</p>
      <div class="model-meta"><span>${escapeHtml(formatSize(model.size))}</span><span>${escapeHtml(quant)}</span></div>
      <button class="model-use-button" data-use-model="${escapeHtml(model.name)}">Use in chat <span>↗</span></button>
    </article>`;
  }).join("");
  elements.modelGrid.querySelectorAll("[data-use-model]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.modelSelect.value = button.dataset.useModel;
      updateActiveModel();
      switchView("chat");
      showToast(`${button.dataset.useModel} selected for chat`);
    });
  });
}

function updateActiveModel() {
  const model = elements.modelSelect.value;
  const backend = elements.modelSelect.selectedOptions[0]?.dataset.backend || "llama.cpp";
  elements.activeModelName.textContent = model || "Choose a model";
  elements.activeModelProvider.textContent = model ? `${backend === "mlx" ? "Apple MLX" : "llama.cpp"} · Local` : "Local runtime";
}

elements.modelSelect.addEventListener("change", updateActiveModel);
document.querySelector("#use-coder-model")?.addEventListener("click", async () => {
  const coder = [...elements.modelSelect.options].find((option) => option.value === "Qwen2.5-Coder-14B-Instruct-MLX-4bit");
  if (!coder) return showToast("Qwen Coder is not available in the local model folder.");
  elements.modelSelect.value = coder.value;
  updateActiveModel();
  await saveActiveSession();
  switchView("chat");
  showToast("Qwen Coder selected. Send it a small app idea to try.");
});
document.querySelector("#refresh-models").addEventListener("click", refreshModels);

elements.hfListFiles.addEventListener("click", async () => {
  const repo = elements.hfRepo.value.trim();
  elements.hfListFiles.disabled = true;
  elements.hfListFiles.textContent = "Looking for GGUF files…";
  elements.downloadStatus.textContent = "Checking the public Hugging Face repository…";
  try {
    const response = await fetch(`/api/huggingface/files?repo=${encodeURIComponent(repo)}`, { cache: "no-store" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not read this repository.");
    elements.hfFile.innerHTML = result.files.map((file) => `<option value="${escapeHtml(file)}">${escapeHtml(file)}</option>`).join("");
    elements.hfFile.disabled = !result.files.length;
    const recommended = result.files.find((file) => /Q4_K_M\.gguf$/i.test(file)) || result.files.find((file) => /Q4/i.test(file)) || result.files[0];
    if (recommended) elements.hfFile.value = recommended;
    elements.hfDownload.disabled = !recommended;
    elements.downloadStatus.textContent = `${result.files.length} single-file GGUF option${result.files.length === 1 ? "" : "s"} found. Split model files are hidden.`;
  } catch (error) {
    elements.hfFile.innerHTML = '<option value="">No files found</option>';
    elements.hfFile.disabled = true;
    elements.hfDownload.disabled = true;
    elements.downloadStatus.textContent = error.message || "Could not read this repository.";
  } finally {
    elements.hfListFiles.disabled = false;
    elements.hfListFiles.textContent = "Find GGUF files";
  }
});

elements.hfDownload.addEventListener("click", async () => {
  const repo = elements.hfRepo.value.trim();
  const file = elements.hfFile.value;
  if (!repo || !file) return;
  elements.hfDownload.disabled = true;
  elements.hfListFiles.disabled = true;
  elements.hfDownload.innerHTML = 'Starting download… <span>↓</span>';
  elements.downloadStatus.textContent = "Starting a direct download from Hugging Face…";
  try {
    const response = await fetch("/api/models/huggingface/download", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ repo, file }),
    });
    const started = await response.json();
    if (!response.ok) throw new Error(started.error || "Could not start the download.");
    let state;
    do {
      await new Promise((resolve) => window.setTimeout(resolve, 1200));
      const statusResponse = await fetch(`/api/model-downloads/${encodeURIComponent(started.job_id)}`, { cache: "no-store" });
      state = await statusResponse.json();
      if (!statusResponse.ok) throw new Error(state.error || "Download status was lost.");
      if (state.status === "downloading") {
        const progress = state.total ? `${formatSize(state.downloaded)} of ${formatSize(state.total)}` : `${formatSize(state.downloaded)} downloaded`;
        elements.downloadStatus.textContent = `Downloading ${state.file} · ${progress}`;
        elements.hfDownload.innerHTML = 'Downloading… <span>↓</span>';
      }
    } while (state.status === "downloading");
    if (state.status !== "complete") throw new Error(state.error || "The download did not finish.");
    elements.downloadStatus.textContent = `${state.model} downloaded and added to this app.`;
    showToast(`${state.model} is ready to use in chat`);
    await refreshModels();
  } catch (error) {
    elements.downloadStatus.textContent = error.message || "Model download failed.";
  } finally {
    elements.hfDownload.disabled = !elements.hfFile.value;
    elements.hfDownload.innerHTML = 'Download to this app <span>↓</span>';
    elements.hfListFiles.disabled = false;
  }
});

elements.ggufChoose.addEventListener("click", () => elements.ggufFile.click());
elements.ggufFile.addEventListener("change", () => {
  const file = elements.ggufFile.files[0];
  elements.ggufFileName.textContent = file ? `${file.name} · ${formatSize(file.size)} · the app will make a copy` : "No file selected";
  elements.ggufAdd.disabled = !file;
});
elements.ggufAdd.addEventListener("click", async () => {
  const file = elements.ggufFile.files[0];
  if (!file) return;
  elements.ggufAdd.disabled = true;
  elements.ggufChoose.disabled = true;
  elements.ggufAdd.innerHTML = 'Copying model… <span>↗</span>';
  elements.ggufFileName.textContent = `Copying ${file.name} into this app… Keep this page open.`;
  try {
    const response = await fetch("/api/models/upload", {
      method: "POST", headers: { "Content-Type": "application/octet-stream", "X-File-Name": encodeURIComponent(file.name) }, body: file,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not add this model file.");
    showToast(`${result.model} added to the app`);
    elements.ggufFileName.textContent = `${file.name} copied into the app's local model folder.`;
    elements.ggufFile.value = "";
    await refreshModels();
  } catch (error) {
    elements.ggufFileName.textContent = error.message || "Could not add this model file.";
    elements.ggufAdd.disabled = false;
  } finally {
    elements.ggufChoose.disabled = false;
    elements.ggufAdd.innerHTML = 'Add model to app <span>↗</span>';
  }
});

function addMessage(role, content) {
  const item = document.createElement("article");
  item.className = `message ${role}`;
  const label = role === "user" ? "YOU" : "ASSISTANT";
  item.innerHTML = `<div class="message-avatar ${role}">${role === "user" ? "A" : "✦"}</div><div class="message-body"><div class="message-label">${label}</div><div class="message-content">${escapeHtml(content).replace(/\n/g, "<br>")}</div></div>`;
  elements.messages.append(item);
  item.scrollIntoView({ behavior: "smooth", block: "end" });
  return item;
}

function clearConversationView() {
  conversation.length = 0;
  elements.messages.innerHTML = "";
  elements.welcome.hidden = false;
  elements.input.value = "";
  elements.input.focus();
}

function renderSessionList(sessions) {
  const list = document.querySelector("#recent-list");
  if (!sessions.length) {
    list.innerHTML = '<div class="recent-item"><span class="recent-dot"></span><span>No saved sessions</span></div>';
    return;
  }
  list.innerHTML = sessions.map((session) => `
    <div class="recent-item-row" style="display:flex;align-items:center;gap:4px;">
      <button class="recent-item ${session.id === activeSessionId ? "selected" : ""}" data-session-id="${escapeHtml(session.id)}" title="${escapeHtml(session.title)}" style="flex:1;text-align:left;">
        <span class="recent-dot"></span><span>${escapeHtml(session.title)}</span>
      </button>
      <button type="button" class="icon-button session-menu-btn" data-session-menu="${escapeHtml(session.id)}" title="Session actions" aria-label="Session actions">⋯</button>
    </div>`).join("");
  list.querySelectorAll("[data-session-id]").forEach((button) => button.addEventListener("click", () => openSession(button.dataset.sessionId)));
  list.querySelectorAll("[data-session-menu]").forEach((button) => button.addEventListener("click", (e) => {
    e.stopPropagation();
    openSessionMenu(button.dataset.sessionMenu, sessions.find((s) => s.id === button.dataset.sessionMenu));
  }));
}

async function openSessionMenu(sessionId, session) {
  const title = session?.title || "Session";
  const actions = session?.archived ? "restore\n  export\n  delete" : "rename\n  archive\n  export\n  delete";
  const choice = window.prompt(
    `Session: ${title}\n\nType one action:\n  ${actions}`,
    session?.archived ? "restore" : "rename"
  );
  if (!choice) return;
  const action = choice.trim().toLowerCase();
  try {
    if (action === "rename") {
      const next = window.prompt("New title", title);
      if (!next) return;
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: next }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Rename failed");
    } else if (action === "archive" || action === "restore") {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/archive`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archived: action === "archive" }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Session update failed");
      if (action === "archive" && activeSessionId === sessionId) await startNewSession();
    } else if (action === "export") {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/export`, { cache: "no-store" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Export failed");
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `session-${(data.title || "export").replace(/[^a-z0-9-_]+/gi, "_").slice(0, 40)}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    } else if (action === "delete") {
      if (!window.confirm(`Delete session “${title}”? This cannot be undone.`)) return;
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Delete failed");
      if (activeSessionId === sessionId) await startNewSession();
    } else {
      window.alert("Unknown action. Use rename, archive, export, or delete.");
      return;
    }
    await refreshSessionList();
  } catch (err) {
    window.alert(err.message || "Session action failed");
  }
}

async function refreshSessionList() {
  const button = document.querySelector("#toggle-archived-sessions");
  if (button) button.textContent = showArchivedSessions ? "Show recent sessions" : "Show archived sessions";
  const response = await fetch(`/api/sessions${showArchivedSessions ? "?archived=1" : ""}`, { cache: "no-store" });
  const data = await response.json();
  renderSessionList((data.sessions || []).filter((session) => showArchivedSessions ? Boolean(session.archived) : !session.archived));
}

document.querySelector("#toggle-archived-sessions")?.addEventListener("click", async () => {
  showArchivedSessions = !showArchivedSessions;
  await refreshSessionList();
});

function renderSessionImages(images) {
  sessionImages = images || [];
  if (sessionImages.length) {
    const latest = sessionImages[0];
    elements.imageResult.innerHTML = `<div class="generated-image-grid">${sessionImages.slice(0, 4).map((image) => `<article><a href="${escapeHtml(image.url)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(image.url)}" alt="Saved generated image"></a><div class="image-result-caption">${escapeHtml(image.filename)} · seed ${escapeHtml(image.seed ?? "—")}</div></article>`).join("")}</div><div class="image-result-caption">${escapeHtml(latest.checkpoint)}${latest.width ? ` · ${latest.width} × ${latest.height} · ${latest.steps} steps` : ""}</div>`;
  } else {
    elements.imageResult.innerHTML = '<div class="image-result-placeholder"><span>▧</span><strong>Your image will appear here</strong><p>Choose an image model and enter a prompt to begin.</p></div>';
  }
  elements.imageHistory.innerHTML = sessionImages.length
    ? `<div class="image-history-title">SAVED IN THIS SESSION · ${sessionImages.length}</div>${sessionImages.map((image) => `<a class="image-history-item" href="${escapeHtml(image.url)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(image.url)}" alt="Saved generated image"><span><strong>${escapeHtml(image.filename)}</strong><small>${escapeHtml(image.prompt)}</small></span></a>`).join("")}`
    : "";
}

function formatLibrarySize(bytes) {
  if (!bytes) return "0 B";
  if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 ** 2)).toFixed(1)} MB`;
}

function renderLibrary() {
  const grid = document.querySelector("#library-grid");
  const query = document.querySelector("#library-search").value.trim().toLowerCase();
  const visible = libraryFiles.filter((file) => file.filename.toLowerCase().includes(query));
  document.querySelector("#library-count").textContent = `${libraryFiles.length} ${libraryFiles.length === 1 ? "file" : "files"}`;
  if (!visible.length) {
    grid.innerHTML = `<div class="empty-card">${query ? "No files match this search." : "Your Library is ready. Create a PDF, generate an image, or add a file to begin."}</div>`;
    return;
  }
  grid.innerHTML = visible.map((file) => {
    const image = file.content_type?.startsWith("image/");
    const pdf = file.content_type === "application/pdf" || file.filename.toLowerCase().endsWith(".pdf");
    const preview = image ? `<img src="${escapeHtml(file.url)}" alt="">` : pdf ? "▤" : "⌑";
    const cls = image ? "image" : pdf ? "pdf" : "doc";
    const date = new Date(file.modified * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
    return `<article class="library-card"><a class="library-preview ${cls}" href="${escapeHtml(file.url)}" target="_blank" rel="noreferrer">${preview}</a><div class="library-card-copy"><strong title="${escapeHtml(file.filename)}">${escapeHtml(file.filename)}</strong><small>${formatLibrarySize(file.size)} · ${date}</small></div><div class="library-card-actions"><a href="${escapeHtml(file.url)}" target="_blank" rel="noreferrer">Open ↗</a><a href="${escapeHtml(file.url)}" download="${escapeHtml(file.filename)}">Save a copy ↓</a></div></article>`;
  }).join("");
}

async function refreshLibrary() {
  const grid = document.querySelector("#library-grid");
  try {
    const response = await fetch("/api/library", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load the Library.");
    libraryFiles = data.files || [];
    renderLibrary();
    refreshPdfFiles();
  } catch (error) {
    grid.innerHTML = `<div class="empty-card">${escapeHtml(error.message || "Could not load the Library.")}</div>`;
  }
}

function refreshPdfFiles() {
  const pdfs = libraryFiles.filter((file) => file.filename.toLowerCase().endsWith(".pdf"));
  for (const selector of ["#pdf-source", "#pdf-edit-source"]) {
    const select = document.querySelector(selector);
    const previous = select.value;
    select.innerHTML = '<option value="">Choose from Library…</option>' + pdfs.map((file) => `<option value="${escapeHtml(file.filename)}">${escapeHtml(file.filename)}</option>`).join("");
    if (pdfs.some((file) => file.filename === previous)) select.value = previous;
  }
}

async function uploadLibraryFiles(files) {
  for (const file of files) {
    const response = await fetch("/api/library/upload", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-File-Name": encodeURIComponent(file.name) },
      body: file,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Could not add ${file.name}.`);
  }
  await refreshLibrary();
  refreshPdfFiles();
}

async function requestPdfAction(action, payload) {
  const response = await fetch(`/api/pdf/${action}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Could not ${action} this PDF.`);
  return result;
}

function selectedChatTools() {
  return [...document.querySelectorAll("[data-chat-tool]:checked")].map((input) => input.dataset.chatTool);
}

async function refreshChatTools() {
  const host = document.querySelector("#local-chat-tools");
  const searchBox = document.querySelector("[data-chat-tool='web_search']");
  const searchStatus = document.querySelector("#web-search-tool-status");
  try {
    const response = await fetch("/api/chat-tools", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load chat tools.");
    if (searchBox) {
      searchBox.disabled = !data.web_search_configured;
      if (!data.web_search_configured) searchBox.checked = false;
    }
    if (searchStatus) searchStatus.textContent = data.web_search_configured
      ? "Search query goes to Brave Search; results are shared with this local model."
      : "Add a Brave Search key in Settings to enable it.";
    if (host) {
      host.innerHTML = (data.tools || []).map((tool) => `<label><input type="checkbox" data-chat-tool="${escapeHtml(tool.chat_name)}"><span><strong>${escapeHtml(tool.name)}</strong><small>${escapeHtml(tool.description)}</small></span></label>`).join("");
      host.querySelectorAll("[data-chat-tool]").forEach((input) => input.addEventListener("change", scheduleSessionStateSave));
    }
    applyChatToolSelection();
  } catch (error) {
    if (host) host.innerHTML = `<small>${escapeHtml(error.message || "Could not load approved tools.")}</small>`;
  }
}

function applyChatToolSelection() {
  const selected = Array.isArray(sessionState.chat_tools) ? sessionState.chat_tools : sessionState.pdf_tools || [];
  document.querySelectorAll("[data-chat-tool]").forEach((input) => { input.checked = !input.disabled && selected.includes(input.dataset.chatTool); });
}

async function refreshSearchSettings() {
  const state = document.querySelector("#brave-search-state");
  try {
    const response = await fetch("/api/search/config", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not read search settings.");
    if (state) state.textContent = data.configured ? `Configured · ${data.source}` : "Not configured";
  } catch (error) {
    if (state) state.textContent = error.message || "Unavailable";
  }
}

document.querySelector("#save-brave-search-key")?.addEventListener("click", async () => {
  const input = document.querySelector("#brave-search-key");
  const state = document.querySelector("#brave-search-state");
  const key = input?.value?.trim();
  if (!key) {
    if (state) state.textContent = "Paste an API key first.";
    return;
  }
  if (state) state.textContent = "Saving locally…";
  try {
    const response = await fetch("/api/search/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: key }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not save the key.");
    input.value = "";
    if (state) state.textContent = `Configured · ${data.source}`;
    await refreshChatTools();
  } catch (error) {
    if (state) state.textContent = error.message || "Could not save the key.";
  }
});

document.querySelector("#clear-brave-search-key")?.addEventListener("click", async () => {
  const state = document.querySelector("#brave-search-state");
  if (state) state.textContent = "Removing saved key…";
  try {
    const response = await fetch("/api/search/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: "" }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not remove the key.");
    if (state) state.textContent = data.configured ? `Configured · ${data.source}` : "Not configured";
    await refreshChatTools();
  } catch (error) {
    if (state) state.textContent = error.message || "Could not remove the key.";
  }
});

function showPdfOutput(result, label) {
  const output = document.querySelector("#pdf-read-result");
  output.textContent = `${label}: ${result.filename}${result.pages ? ` · ${result.pages} pages` : ""}. Saved in your Library.`;
  const link = document.createElement("a");
  link.className = "pdf-result-link";
  link.href = result.url;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = "Open PDF ↗";
  output.append(document.createElement("br"), link);
}

async function refreshActiveSession() {
  if (!activeSessionId) return;
  const response = await fetch(`/api/sessions/${encodeURIComponent(activeSessionId)}`, { cache: "no-store" });
  if (!response.ok) return;
  const session = await response.json();
  sessionState = session.state || {};
  renderSessionImages(session.images || []);
  await refreshSessionList();
}

async function saveActiveSession() {
  if (!activeSessionId) return;
  const response = await fetch(`/api/sessions/${encodeURIComponent(activeSessionId)}`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: conversation,
      state: {
        model: elements.modelSelect.value,
        image_checkpoint: elements.imageCheckpoint.value,
        loras: [...elements.imageLoraList.querySelectorAll("input:checked")].map((input) => input.value),
        lora_strength: Number(document.querySelector("#lora-strength").value),
        negative_prompt: document.querySelector("#image-negative").value,
        image_prompt: document.querySelector("#image-prompt").value,
        seed_mode: document.querySelector("#seed-mode").value,
        seed: Number(document.querySelector("#image-seed").value),
        width: Number(document.querySelector("#image-width").value),
        height: Number(document.querySelector("#image-height").value),
        steps: Number(document.querySelector("#image-steps").value),
        cfg: Number(document.querySelector("#image-cfg").value),
        sampler: document.querySelector("#image-sampler").value,
        scheduler: document.querySelector("#image-scheduler").value,
        count: Number(document.querySelector("#image-count").value),
        chat_tools: selectedChatTools(),
      },
    }),
  });
  if (response.ok) await refreshSessionList();
}

async function openSession(sessionId) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { cache: "no-store" });
  if (!response.ok) return;
  const session = await response.json();
  activeSessionId = session.id;
  localStorage.setItem("ai-command-center-session", activeSessionId);
  conversation.splice(0, conversation.length, ...(session.messages || []).map(({ role, content }) => ({ role, content })));
  sessionState = session.state || {};
  elements.messages.innerHTML = "";
  conversation.forEach((message) => addMessage(message.role, message.content));
  elements.welcome.hidden = conversation.length > 0;
  elements.input.value = "";
  renderSessionImages(session.images || []);
  applySessionState();
  await refreshSessionList();
  switchView("chat");
}

async function startNewSession() {
  showArchivedSessions = false;
  const response = await fetch("/api/sessions", { method: "POST" });
  const session = await response.json();
  activeSessionId = session.id;
  sessionState = {};
  const starterModel = [...elements.modelSelect.options].some((option) => option.value === "Qwen2.5-Coder-14B-Instruct-MLX-4bit")
    ? "Qwen2.5-Coder-14B-Instruct-MLX-4bit"
    : [...elements.modelSelect.options].some((option) => option.value === "qwen2.5-0.5b-instruct-gguf:latest")
      ? "qwen2.5-0.5b-instruct-gguf:latest"
    : elements.modelSelect.options[0]?.value;
  if (starterModel) elements.modelSelect.value = starterModel;
  updateActiveModel();
  localStorage.setItem("ai-command-center-session", activeSessionId);
  document.querySelector("#image-prompt").value = "";
  document.querySelector("#image-negative").value = "";
  document.querySelector("#lora-strength").value = "0.8";
  document.querySelector("#lora-strength-value").value = "0.80";
  elements.imageLoraList.querySelectorAll("input[type=checkbox]").forEach((input) => { input.checked = false; });
  document.querySelectorAll("[data-chat-tool]").forEach((input) => { input.checked = false; });
  renderSessionImages([]);
  clearConversationView();
  await refreshSessionList();
  switchView("chat");
}

async function initializeSessions() {
  await refreshModels();
  await refreshLibrary();
  await refreshChatTools();
  await refreshSearchSettings();
  refreshPdfFiles();
  const response = await fetch("/api/sessions", { cache: "no-store" });
  const data = await response.json();
  const sessions = data.sessions || [];
  const remembered = localStorage.getItem("ai-command-center-session");
  const target = sessions.find((session) => session.id === remembered) || sessions[0];
  if (target) await openSession(target.id);
  else await startNewSession();
}

function applySessionState() {
  const modelExists = (name) => [...elements.modelSelect.options].some((option) => option.value === name);
  const preferredModel = sessionState.model || (modelExists("Qwen2.5-Coder-14B-Instruct-MLX-4bit") ? "Qwen2.5-Coder-14B-Instruct-MLX-4bit" : modelExists("qwen2.5-0.5b-instruct-gguf:latest") ? "qwen2.5-0.5b-instruct-gguf:latest" : elements.modelSelect.options[0]?.value);
  if (preferredModel && modelExists(preferredModel)) {
    elements.modelSelect.value = preferredModel;
    updateActiveModel();
  }
  if (sessionState.image_checkpoint && [...elements.imageCheckpoint.options].some((option) => option.value === sessionState.image_checkpoint)) {
    elements.imageCheckpoint.value = sessionState.image_checkpoint;
  }
  if (Array.isArray(sessionState.loras)) {
    elements.imageLoraList.querySelectorAll("input[type=checkbox]").forEach((input) => { input.checked = sessionState.loras.includes(input.value); });
  }
  if (typeof sessionState.lora_strength === "number") {
    document.querySelector("#lora-strength").value = String(sessionState.lora_strength);
    document.querySelector("#lora-strength-value").value = sessionState.lora_strength.toFixed(2);
  }
  if (typeof sessionState.negative_prompt === "string") document.querySelector("#image-negative").value = sessionState.negative_prompt;
  if (typeof sessionState.image_prompt === "string") document.querySelector("#image-prompt").value = sessionState.image_prompt;
  applyChatToolSelection();
  for (const [key, selector] of Object.entries({ seed_mode: "#seed-mode", seed: "#image-seed", width: "#image-width", height: "#image-height", steps: "#image-steps", cfg: "#image-cfg", sampler: "#image-sampler", scheduler: "#image-scheduler", count: "#image-count" })) {
    if (sessionState[key] !== undefined) document.querySelector(selector).value = String(sessionState[key]);
  }
}

let stateSaveTimer;
function scheduleSessionStateSave() {
  window.clearTimeout(stateSaveTimer);
  stateSaveTimer = window.setTimeout(() => saveActiveSession(), 350);
}

elements.modelSelect.addEventListener("change", scheduleSessionStateSave);
document.querySelectorAll("[data-chat-tool]").forEach((input) => input.addEventListener("change", scheduleSessionStateSave));
const speakReplies = document.querySelector("#iris-speak-replies");
if (speakReplies) {
  speakReplies.checked = localStorage.getItem("iris-speak-replies") === "true";
  speakReplies.addEventListener("change", () => localStorage.setItem("iris-speak-replies", speakReplies.checked ? "true" : "false"));
}
document.querySelector("#image-form").addEventListener("change", scheduleSessionStateSave);
document.querySelector("#image-prompt").addEventListener("input", scheduleSessionStateSave);
document.querySelector("#image-negative").addEventListener("input", scheduleSessionStateSave);

document.querySelector("[data-action='new-chat']").addEventListener("click", startNewSession);
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.input.value = button.dataset.prompt;
    elements.input.focus();
    elements.input.dispatchEvent(new Event("input"));
  });
});

elements.input.addEventListener("input", () => {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 180)}px`;
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.querySelector("#chat-form").requestSubmit();
  }
});

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = elements.input.value.trim();
  const model = elements.modelSelect.value;
  if (!text) return;
  if (!model) {
    showToast("Connect a local model to start a conversation.");
    switchView("models");
    return;
  }

  elements.welcome.hidden = true;
  conversation.push({ role: "user", content: text });
  addMessage("user", text);
  await saveActiveSession();
  elements.input.value = "";
  elements.input.style.height = "auto";
  const pending = addMessage("assistant", "Thinking…");
  pending.classList.add("pending");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: conversation, tools: selectedChatTools() }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The model could not answer.");
    let answer = data.message?.content || "The model returned an empty response.";
    if (data.tool_actions?.length) {
      const notes = data.tool_actions.map((action) => {
        const result = action.result || {};
        if (result.error) return `${action.name} failed: ${result.error}`;
        if (action.name === "read_pdf") return `Read ${result.filename} from your Library.`;
        if (action.name === "create_pdf" || action.name === "edit_pdf") return `${action.name === "edit_pdf" ? "Revised copy saved" : "PDF saved"} in your Library as ${result.filename}.`;
        if (action.name === "web_search") return "Searched the web with Brave Search.";
        if (action.name.startsWith("local__")) return `Ran approved local tool: ${action.name.slice(7)}.`;
        return `Ran tool: ${action.name}.`;
      });
      answer = `${answer}\n\n${notes.join("\n")}`;
      const sources = [...new Map(data.tool_actions.filter((action) => action.name === "web_search")
        .flatMap((action) => action.result?.results || [])
        .filter((item) => item.url)
        .map((item) => [item.url, item])).values()];
      if (sources.length) answer += `\n\nSources:\n${sources.map((item) => `- ${item.title} — ${item.url}`).join("\n")}`;
      if (data.tool_actions.some((action) => ["create_pdf", "read_pdf", "edit_pdf"].includes(action.name))) {
        await refreshLibrary();
        refreshPdfFiles();
      }
    }
    conversation.push({ role: "assistant", content: answer });
    pending.remove();
    addMessage("assistant", answer);
    await saveActiveSession();
    if (speakReplies?.checked) irisSpeak(answer);
  } catch (error) {
    pending.remove();
    const failure = error.message || "Something went wrong while contacting the local model.";
    conversation.push({ role: "assistant", content: failure });
    addMessage("assistant", failure);
    await saveActiveSession();
  }
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    startNewSession();
  }
  if ((event.metaKey || event.ctrlKey) && /^[1-7]$/.test(event.key)) {
    event.preventDefault();
    const names = ["chat", "code", "models", "images", "library", "tools", "automations"];
    switchView(names[Number(event.key) - 1]);
  }
});

document.querySelector("#refresh-library").addEventListener("click", refreshLibrary);
document.querySelector("#library-search").addEventListener("input", renderLibrary);
document.querySelector("#library-upload").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  if (!files.length) return;
  try {
    await uploadLibraryFiles(files);
    showToast(`${files.length} ${files.length === 1 ? "file" : "files"} added to your Library`);
  } catch (error) {
    showToast(error.message || "Could not add these files.");
  } finally {
    event.target.value = "";
  }
});
document.querySelector("#pdf-create-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const result = await requestPdfAction("create", { title: document.querySelector("#pdf-title").value, text: document.querySelector("#pdf-body").value });
    showPdfOutput(result, "Created");
    await refreshLibrary();
    refreshPdfFiles();
    showToast("PDF saved in your Library");
  } catch (error) {
    document.querySelector("#pdf-read-result").textContent = error.message || "Could not create this PDF.";
  } finally { button.disabled = false; }
});
document.querySelector("#pdf-read").addEventListener("click", async () => {
  const filename = document.querySelector("#pdf-source").value;
  if (!filename) return showToast("Choose a PDF from your Library first.");
  const output = document.querySelector("#pdf-read-result");
  output.textContent = "Reading PDF…";
  try {
    const result = await requestPdfAction("read", { filename });
    output.textContent = `${result.filename} · ${result.pages} pages${result.truncated ? " · text shortened" : ""}\n\n${result.text || "No selectable text was found in this PDF."}`;
  } catch (error) { output.textContent = error.message || "Could not read this PDF."; }
});
document.querySelector("#pdf-edit-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const filename = document.querySelector("#pdf-edit-source").value;
  if (!filename) return showToast("Choose a PDF from your Library first.");
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const result = await requestPdfAction("edit", { filename, title: document.querySelector("#pdf-edit-title").value, note: document.querySelector("#pdf-edit-note").value });
    showPdfOutput(result, "Revised copy created");
    await refreshLibrary();
    refreshPdfFiles();
    showToast("Revised PDF saved in your Library; the original is unchanged");
  } catch (error) {
    document.querySelector("#pdf-read-result").textContent = error.message || "Could not revise this PDF.";
  } finally { button.disabled = false; }
});

initializeSessions();



/* —— Coding workspace MVP —— */
const codeLog = document.querySelector("#code-chat-log");
const codeInput = document.querySelector("#code-input");
let codeMessages = [];

async function refreshCodeProject() {
  try {
    const prefs = await fetch("/api/coding/prefs", { cache: "no-store" }).then((r) => r.json());
    const pathInput = document.querySelector("#code-project-path");
    if (pathInput && prefs.project_root) pathInput.value = prefs.project_root;
    document.querySelector("#code-project-title").textContent = prefs.project_root || "No project folder selected";
    document.querySelector("#code-project-copy").textContent = prefs.project_root
      ? "File tools are scoped to this folder. Terminal commands run with your macOS user permissions."
      : "Choose a folder on this Mac. File tools are scoped to it; terminal commands run with your macOS user permissions.";
    document.querySelector("#code-project-state").textContent = prefs.project_root ? "READY" : "IDLE";
    if (prefs.project_root) await refreshCodeTree();
  } catch (e) {
    document.querySelector("#code-project-state").textContent = "ERROR";
  }
}

async function refreshCodeTree() {
  const tree = document.querySelector("#code-tree");
  try {
    const data = await fetch("/api/coding/tree", { cache: "no-store" }).then((r) => r.json());
    if (data.error) throw new Error(data.error);
    const prefs = await fetch("/api/coding/prefs", { cache: "no-store" }).then((r) => r.json());
    const selected = new Set(prefs.selected_files || []);
    tree.innerHTML = (data.entries || []).map((e) => {
      const mark = e.type === "dir" ? "📁" : "📄";
      const active = selected.has(e.path) ? " style=\"outline:1px solid var(--accent, #6c8);\"" : "";
      return `<button type="button" class="secondary-button" data-code-path="${escapeHtml(e.path)}" data-code-type="${e.type}"${active}>${mark} ${escapeHtml(e.path)}${e.size != null ? " · " + formatSize(e.size) : ""}</button>`;
    }).join("") || '<div class="empty-card">Empty folder.</div>';
    tree.querySelectorAll("[data-code-path]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (btn.dataset.codeType === "dir") return;
        const path = btn.dataset.codePath;
        const prefs = await fetch("/api/coding/prefs", { cache: "no-store" }).then((r) => r.json());
        let files = prefs.selected_files || [];
        if (files.includes(path)) files = files.filter((f) => f !== path);
        else files = [...files, path].slice(0, 40);
        await fetch("/api/coding/select-files", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ files }),
        });
        refreshCodeTree();
      });
    });
  } catch (e) {
    tree.innerHTML = `<div class="empty-card">${escapeHtml(e.message || "Could not list files")}</div>`;
  }
}

document.querySelector("#code-set-project")?.addEventListener("click", async () => {
  const path = document.querySelector("#code-project-path").value.trim();
  document.querySelector("#code-status").textContent = "Setting project…";
  try {
    const res = await fetch("/api/coding/project", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed");
    codeMessages = [];
    if (codeLog) codeLog.textContent = "Project set. Ask the agent to explore or edit.";
    await refreshCodeProject();
    document.querySelector("#code-status").textContent = "Project ready.";
  } catch (e) {
    document.querySelector("#code-status").textContent = e.message || "Could not set project";
  }
});

document.querySelector("#code-refresh-tree")?.addEventListener("click", refreshCodeTree);

document.querySelector("#code-send")?.addEventListener("click", async () => {
  const text = (codeInput?.value || "").trim();
  if (!text) return;
  const model = elements.modelSelect?.value || "";
  codeMessages.push({ role: "user", content: text });
  codeInput.value = "";
  if (codeLog) codeLog.textContent = (codeLog.textContent || "") + "\n\nYou: " + text + "\n\nAgent: …";
  document.querySelector("#code-status").textContent = "Coding agent working…";
  document.querySelector("#code-send").disabled = true;
  try {
    const res = await fetch("/api/coding/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: codeMessages, model }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Coding chat failed");
    const msg = data.message || {};
    const content = msg.content || JSON.stringify(msg).slice(0, 4000);
    codeMessages.push({ role: "assistant", content });
    if (codeLog) codeLog.textContent = (codeLog.textContent || "").replace(/Agent: …$/, "Agent: " + content);
    document.querySelector("#code-status").textContent = "Done.";
    refreshCodeTree();
  } catch (e) {
    document.querySelector("#code-status").textContent = e.message || "Error";
    if (codeLog) codeLog.textContent = (codeLog.textContent || "") + "\n\nError: " + (e.message || e);
  } finally {
    document.querySelector("#code-send").disabled = false;
  }
});

/* —— Local tool create / review —— */
async function refreshLocalTools() {
  const box = document.querySelector("#local-tools-list");
  if (!box) return;
  try {
    const data = await fetch("/api/local-tools", { cache: "no-store" }).then((r) => r.json());
    const tools = data.tools || [];
    if (!tools.length) {
      box.innerHTML = '<div class="empty-card">No local tools saved yet.</div>';
      return;
    }
    box.innerHTML = tools.map((t) => `
      <div class="pdf-tool-card" style="margin-bottom:10px;">
        <h3>${escapeHtml(t.name)}</h3>
        <p>${escapeHtml(t.description || "")}</p>
        <button type="button" class="secondary-button" data-run-local="${escapeHtml(t.name)}">Run</button>
        <button type="button" class="secondary-button" data-del-local="${escapeHtml(t.name)}">Delete</button>
      </div>`).join("");
    box.querySelectorAll("[data-run-local]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        let args = {};
        try { args = JSON.parse(document.querySelector("#tool-args")?.value || "{}"); } catch (_) {}
        const res = await fetch("/api/local-tools/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: btn.dataset.runLocal, arguments: args }),
        });
        const data = await res.json();
        document.querySelector("#tool-review-status").textContent = JSON.stringify(data).slice(0, 500);
      });
    });
    box.querySelectorAll("[data-del-local]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await fetch("/api/local-tools/" + encodeURIComponent(btn.dataset.delLocal), { method: "DELETE" });
        refreshLocalTools();
      });
    });
  } catch (e) {
    box.innerHTML = `<div class="empty-card">${escapeHtml(e.message || "Failed to load tools")}</div>`;
  }
}

document.querySelector("#tool-build-prompt")?.addEventListener("click", async () => {
  const request = document.querySelector("#tool-request")?.value || "";
  const res = await fetch("/api/local-tools/generate-prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request }),
  });
  const data = await res.json();
  document.querySelector("#tool-gen-status").textContent = "Prompt ready — paste into chat or use Generate.";
  if (data.prompt) document.querySelector("#tool-source").value = "# Prompt for coder:\n# " + data.prompt.slice(0, 500) + "\n";
});

document.querySelector("#tool-generate")?.addEventListener("click", async () => {
  const request = document.querySelector("#tool-request")?.value?.trim();
  if (!request) return;
  const status = document.querySelector("#tool-gen-status");
  status.textContent = "Generating with current model…";
  try {
    const promptRes = await fetch("/api/local-tools/generate-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request }),
    });
    const { prompt } = await promptRes.json();
    const model = elements.modelSelect?.value || "";
    // Reuse coding chat endpoint-style via main chat API if present; fallback: ask user to paste
    // Use a minimal generate via /api/coding/chat without project if needed — instead call model through sessions
    // Prefer dedicated: post as single-turn using coding chat with empty project fails — use fetch to a simple completion
    const res = await fetch("/api/local-tools/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "" }),
    }).catch(() => null);
    // Generate via coding agent system by piggybacking chat API
    const chatRes = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: prompt }],
        model,
        tools: [],
      }),
    });
    if (chatRes.ok) {
      const chatData = await chatRes.json();
      const content = chatData.message?.content || chatData.content || chatData.reply || JSON.stringify(chatData);
      const parsedRes = await fetch("/api/local-tools/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: content }),
      });
      const parsedData = await parsedRes.json();
      if (!parsedRes.ok) throw new Error(parsedData.error || "Parse failed");
      const parsed = parsedData.parsed;
      document.querySelector("#tool-name").value = parsed.name || "custom_tool";
      document.querySelector("#tool-description").value = parsed.description || "";
      document.querySelector("#tool-source").value = parsed.source || "";
      status.textContent = "Draft ready — review and approve.";
    } else {
      // Fallback: put prompt in status for manual chat
      status.textContent = "Could not auto-generate (chat API). Copy the request into Chat with a coder model, then paste the reply into Source and click parse via Approve after editing.";
      document.querySelector("#tool-source").value = prompt;
    }
  } catch (e) {
    status.textContent = e.message || "Generate failed";
  }
});

document.querySelector("#tool-test")?.addEventListener("click", async () => {
  const status = document.querySelector("#tool-review-status");
  // Save temp then run — or run from form by saving first
  try {
    const body = {
      name: document.querySelector("#tool-name").value.trim() || "temp_tool",
      description: document.querySelector("#tool-description").value.trim(),
      source: document.querySelector("#tool-source").value,
    };
    const save = await fetch("/api/local-tools", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const saved = await save.json();
    if (!save.ok) throw new Error(saved.error || "Save failed for test");
    let args = {};
    try { args = JSON.parse(document.querySelector("#tool-args").value || "{}"); } catch (_) {}
    const run = await fetch("/api/local-tools/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: body.name, arguments: args }),
    });
    const result = await run.json();
    status.textContent = JSON.stringify(result).slice(0, 800);
    refreshLocalTools();
  } catch (e) {
    status.textContent = e.message || "Test failed";
  }
});

document.querySelector("#tool-approve")?.addEventListener("click", async () => {
  const status = document.querySelector("#tool-review-status");
  try {
    const body = {
      name: document.querySelector("#tool-name").value.trim(),
      description: document.querySelector("#tool-description").value.trim(),
      source: document.querySelector("#tool-source").value,
    };
    const res = await fetch("/api/local-tools", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Save failed");
    status.textContent = "Saved " + data.tool.name;
    refreshLocalTools();
  } catch (e) {
    status.textContent = e.message || "Approve failed";
  }
});

document.querySelector("#tool-refresh-list")?.addEventListener("click", refreshLocalTools);

// Hook view switches
const _origShow = typeof showView === "function" ? showView : null;
document.querySelectorAll("[data-view='code']").forEach((el) => {
  el.addEventListener("click", () => setTimeout(refreshCodeProject, 50));
});
document.querySelectorAll("[data-view='tools']").forEach((el) => {
  el.addEventListener("click", () => setTimeout(refreshLocalTools, 50));
});


/* —— IRIS hybrid navigation —— */
function irisShowHome() {
  const home = document.querySelector("#iris-home");
  const node = document.querySelector("#iris-node-view");
  if (home) home.hidden = false;
  if (node) node.hidden = true;
  refreshIrisStatus();
}

function openIrisNode(name) {
  const home = document.querySelector("#iris-home");
  const node = document.querySelector("#iris-node-view");
  if (home) home.hidden = true;
  if (node) node.hidden = false;
  document.querySelectorAll(".iris-node-nav button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.irisNode === name);
  });
  document.querySelectorAll(".iris-ws").forEach((ws) => { ws.hidden = true; });
  const panel = document.querySelector("#iris-ws-" + name);
  if (panel) panel.hidden = false;
  const label = document.querySelector("#iris-active-node");
  if (label) label.textContent = name;
  const model = document.querySelector("#iris-node-model");
  if (model && elements.modelSelect) model.textContent = elements.modelSelect.value || "—";
}

async function refreshIrisStatus() {
  try {
    const models = await fetch("/api/models", { cache: "no-store" }).then((r) => r.json()).catch(() => ({}));
    const mlx = document.querySelector("#iris-mlx-status");
    if (mlx) {
      const list = models.models || models || [];
      const hasMlx = Array.isArray(list) && list.some((m) => m.backend === "mlx");
      mlx.textContent = hasMlx ? "ready" : "—";
      mlx.style.color = hasMlx ? "#3dffb0" : "#7a93b0";
    }
  } catch (_) {}
  try {
    const cat = await fetch("/api/image-catalog", { cache: "no-store" }).then((r) => r.json());
    const el = document.querySelector("#iris-img-status");
    if (el) {
      const pt = cat.pytorch || {};
      if (pt.device === "mps" && pt.mps_available) el.textContent = "PyTorch MPS";
      else if (cat.sd_cpp_ready || cat.engine_ready) el.textContent = "sd.cpp";
      else el.textContent = "setup";
    }
  } catch (_) {}
  try {
    const prefs = await fetch("/api/coding/prefs", { cache: "no-store" }).then((r) => r.json());
    const el = document.querySelector("#iris-code-status");
    if (el) el.textContent = prefs.project_root ? "linked" : "none";
  } catch (_) {}
  try {
    const tools = await fetch("/api/local-tools", { cache: "no-store" }).then((r) => r.json());
    const el = document.querySelector("#iris-tools-status");
    if (el) el.textContent = String((tools.tools || []).length);
  } catch (_) {}
  const mission = document.querySelector("#iris-mission-title");
  const missionModel = document.querySelector("#iris-mission-model");
  if (mission) mission.textContent = activeSessionId ? "Session active" : "No active session focus";
  if (missionModel && elements.modelSelect) missionModel.textContent = elements.modelSelect.value || "—";
}

document.querySelectorAll("[data-iris-node]").forEach((btn) => {
  btn.addEventListener("click", () => openIrisNode(btn.dataset.irisNode));
});
document.querySelector("#iris-back-home")?.addEventListener("click", irisShowHome);

// When switching to IRIS view, show home
document.querySelectorAll("[data-view='iris']").forEach((el) => {
  el.addEventListener("click", () => setTimeout(irisShowHome, 30));
});

// Workspace buttons inside node view use existing data-view handlers


document.querySelector("#model-path-add")?.addEventListener("click", async () => {
  const path = document.querySelector("#model-path-input")?.value?.trim();
  const status = document.querySelector("#model-path-status");
  if (!path) {
    if (status) status.textContent = "Enter an absolute path to a .gguf file or MLX model folder.";
    return;
  }
  if (status) status.textContent = "Registering model path…";
  try {
    const res = await fetch("/api/models/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not register path");
    if (status) status.textContent = "Added to model list: " + (data.model || path);
    await refreshModels();
  } catch (e) {
    if (status) status.textContent = e.message || "Failed";
  }
});



/* —— IRIS voice: two women voices (STT + TTS) —— */
const irisVoice = {
  pair: { A: null, B: null },
  recognition: null,
  listening: false,
  transcript: "",
  lastError: "",
};

function irisPickWomenVoices() {
  const voices = window.speechSynthesis ? speechSynthesis.getVoices() : [];
  // Prefer known female English voices on macOS / browsers
  const preferA = [/samantha/i, /karen/i, /moira/i, /fiona/i, /tessa/i, /veena/i, /serena/i, /female/i, /woman/i];
  const preferB = [/karen/i, /moira/i, /fiona/i, /tessa/i, /veena/i, /samantha/i, /zira/i, /female/i];
  const en = voices.filter((v) => /en(-|_|$)/i.test(v.lang) || /english/i.test(v.name));
  const pool = en.length ? en : voices;
  const score = (list, prefs) => {
    for (const re of prefs) {
      const hit = list.find((v) => re.test(v.name));
      if (hit) return hit;
    }
    // heuristic: names often used for women on Apple
    const soft = list.find((v) => /a$|ie$|y$/i.test(v.name.split(" ")[0] || ""));
    return soft || list[0] || null;
  };
  let a = score(pool, preferA);
  let b = score(pool.filter((v) => v !== a), preferB);
  if (!b && pool.length > 1) b = pool.find((v) => v !== a) || a;
  if (!a && voices.length) a = voices[0];
  if (!b) b = a;
  irisVoice.pair.A = a;
  irisVoice.pair.B = b;
  const nameEl = document.querySelector("#iris-voice-name");
  const sel = document.querySelector("#iris-voice-select")?.value || "A";
  const active = irisVoice.pair[sel];
  if (nameEl) nameEl.textContent = active ? active.name : "No system voice";
  const hint = document.querySelector("#iris-voice-hint");
  if (hint && a) {
    hint.textContent = `Voice A: ${a.name}${b && b !== a ? " · Voice B: " + b.name : " · Voice B uses the same voice because only one match was found"}. Browser voice names are a best-effort match.`;
  }
  return { A: a, B: b };
}

function irisActiveVoice() {
  if (!irisVoice.pair.A && window.speechSynthesis) irisPickWomenVoices();
  const sel = document.querySelector("#iris-voice-select")?.value || "A";
  return irisVoice.pair[sel] || irisVoice.pair.A;
}

function irisSetVoiceState(state, detail) {
  const el = document.querySelector("#iris-voice-state");
  const orb = document.querySelector("#iris-voice-orb");
  const wave = document.querySelector("#iris-wave");
  if (el) el.textContent = detail || state;
  document.querySelectorAll("#iris-ptt, #iris-ptt-main").forEach((button) => {
    button.setAttribute("aria-pressed", state === "listening" ? "true" : "false");
  });
  if (orb) {
    orb.classList.toggle("listening", state === "listening");
    orb.classList.toggle("speaking", state === "speaking");
  }
  if (wave) wave.classList.toggle("active", state === "listening" || state === "speaking");
}

function irisSpeak(text) {
  if (!window.speechSynthesis || !text) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(String(text).slice(0, 4000));
  const voice = irisActiveVoice();
  if (voice) u.voice = voice;
  u.rate = 1.02;
  u.pitch = 1.0;
  u.onstart = () => irisSetVoiceState("speaking", "Speaking…");
  u.onend = () => irisSetVoiceState("idle", "Idle");
  u.onerror = () => irisSetVoiceState("idle", "Idle");
  speechSynthesis.speak(u);
}

function irisStopSpeak() {
  if (window.speechSynthesis) speechSynthesis.cancel();
  irisSetVoiceState("idle", "Idle");
}

function irisEnsureRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  if (irisVoice.recognition) return irisVoice.recognition;
  const rec = new SR();
  rec.continuous = false;
  rec.interimResults = true;
  rec.lang = "en-US";
  rec.onstart = () => {
    irisVoice.listening = true;
    irisSetVoiceState("listening", "Listening…");
  };
  rec.onend = () => {
    irisVoice.listening = false;
    if (!irisVoice.lastError && !window.speechSynthesis?.speaking) irisSetVoiceState("idle", "Idle");
  };
  rec.onerror = (ev) => {
    irisVoice.listening = false;
    const messages = {
      "not-allowed": "Microphone blocked. Allow microphone access for localhost in browser site settings.",
      "service-not-allowed": "The browser speech service is unavailable for this page.",
      network: "Browser speech service network error. Check the connection and try again.",
      "no-speech": "No speech detected. Hold the button while speaking, then release.",
      "audio-capture": "No microphone is available to the browser.",
    };
    irisVoice.lastError = messages[ev.error] || `Speech recognition error: ${ev.error || "unknown"}`;
    irisSetVoiceState("idle", irisVoice.lastError);
  };
  rec.onresult = (ev) => {
    let text = "";
    for (let i = 0; i < ev.results.length; i++) {
      text += ev.results[i][0].transcript;
    }
    irisVoice.transcript = text.trim();
    const box = document.querySelector("#iris-transcript");
    if (box) box.textContent = irisVoice.transcript || "…";
    // Final result → optional auto-reply short ack with selected woman voice
    if (ev.results[ev.results.length - 1].isFinal && irisVoice.transcript) {
      irisSpeak("Got it.");
    }
  };
  irisVoice.recognition = rec;
  return rec;
}

function irisStartListen() {
  const rec = irisEnsureRecognition();
  if (!rec) {
    irisSetVoiceState("idle", "Speech recognition not available in this browser");
    return;
  }
  irisVoice.lastError = "";
  try {
    rec.start();
  } catch (_) {
    /* already started */
  }
}

function irisStopListen() {
  try {
    irisVoice.recognition?.stop();
  } catch (_) {}
}

function bindIrisPtt(btn) {
  if (!btn) return;
  const down = (e) => {
    e.preventDefault();
    if (e.pointerId !== undefined && btn.setPointerCapture) {
      try { btn.setPointerCapture(e.pointerId); } catch (_) {}
    }
    irisStartListen();
  };
  const up = (e) => {
    e.preventDefault();
    irisStopListen();
  };
  btn.addEventListener("pointerdown", down);
  btn.addEventListener("pointerup", up);
  btn.addEventListener("pointercancel", up);
  btn.addEventListener("keydown", (event) => { if ((event.key === " " || event.key === "Enter") && !event.repeat) down(event); });
  btn.addEventListener("keyup", (event) => { if (event.key === " " || event.key === "Enter") up(event); });
}

bindIrisPtt(document.querySelector("#iris-ptt"));
bindIrisPtt(document.querySelector("#iris-ptt-main"));

document.querySelector("#iris-stop-speak")?.addEventListener("click", irisStopSpeak);
document.querySelector("#iris-test-voice")?.addEventListener("click", () => {
  irisPickWomenVoices();
  const v = irisActiveVoice();
  irisSpeak(v ? `Hello, I am IRIS, using ${v.name}.` : "Hello, I am IRIS.");
});
document.querySelector("#iris-voice-select")?.addEventListener("change", () => {
  irisPickWomenVoices();
  const v = irisActiveVoice();
  if (v) irisSpeak(`Voice selected: ${v.name}`);
});
document.querySelector("#iris-send-transcript")?.addEventListener("click", async () => {
  const text = irisVoice.transcript || document.querySelector("#iris-transcript")?.textContent || "";
  if (!text || text.startsWith("Press")) {
    irisSpeak("I did not catch a transcript yet.");
    return;
  }
  switchView("chat");
  if (elements.input) {
    elements.input.value = text;
    elements.input.dispatchEvent(new Event("input"));
    elements.input.focus();
  }
  irisSpeak("Transcript placed in chat.");
});

if (window.speechSynthesis) {
  speechSynthesis.onvoiceschanged = () => irisPickWomenVoices();
  setTimeout(irisPickWomenVoices, 300);
}
