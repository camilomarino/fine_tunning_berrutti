(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    catalog: null,
    method: "pages",
    split: "validation",
    family: "pages",
    evaluation: "",
    document: "",
    query: "",
    syntheticEpoch: "",
    columns: "",
    listPage: 0,
    linePage: 0,
    lineQuery: "",
    selectedLine: "",
    textMode: "raw",
    zoom: 100,
    index: [],
    indexLoading: false,
    detail: null,
    prediction: null,
    basePrediction: null,
    transcriptionView: "all",
    diffReference: "reference",
    diffCompared: "tuned",
    request: 0,
  };
  const cache = new Map();
  const LIST_SIZE = 40;
  const LINES_SIZE = 24;
  const labels = { pages: "Sólo páginas", lines: "Sólo líneas", combined: "Combinación intercalada", experiment4: "Berrutti + LUISA" };
  const transcriptionViews = {
    all: ["reference", "base", "tuned"],
    "reference-base": ["reference", "base"],
    "reference-tuned": ["reference", "tuned"],
    "base-tuned": ["base", "tuned"],
    reference: ["reference"], base: ["base"], tuned: ["tuned"],
  };
  const descriptions = {
    pages: "125 páginas Berrutti completas por época. 3 épocas.",
    lines: "5.212 recortes de línea Berrutti por época. 3 épocas.",
    combined: "125 páginas y sus 5.212 líneas, intercaladas. 3 épocas.",
    experiment4: "Berrutti completo + 23.774 líneas LUISA y páginas sintéticas. 3 épocas mixtas + 1 de consolidación Berrutti.",
  };
  const int = new Intl.NumberFormat("es-UY", { maximumFractionDigits: 0 });
  const decimal = new Intl.NumberFormat("es-UY", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const percent = (value) => finite(value) ? `${decimal.format(value * 100)} %` : "—";
  const count = (value) => finite(value) ? int.format(value) : "—";
  const normalizedSplit = (value) => value === "val" ? "validation" : value;
  const method = () => state.catalog.methods.find((item) => item.id === state.method);
  const evaluation = () => (state.catalog.evaluations || []).find((item) => item.id === state.evaluation);
  const summaryOf = (value) => value?.summary || value || {};
  const lineList = () => Array.isArray(state.detail?.lines) ? state.detail.lines : [];
  const isBaseline = (item) => item && (item.stage === "baseline" || Number(item.epoch) === 0);
  const tunedPrediction = () => isBaseline(evaluation()) ? null : state.prediction;

  function baselineEvaluation(catalog, methodId, split) {
    const candidates = (catalog.evaluations || []).filter((item) => normalizedSplit(item.split) === split && isBaseline(item) &&
      (methodId === "experiment4" ? ["experiment4", "experiment4_baseline"].includes(item.method) : ["baseline", "shared"].includes(item.method)));
    if (candidates.length > 1) throw new Error("Hay más de una línea base para esta partición.");
    return candidates[0] || null;
  }

  // Pairwise edit alignments share a chosen anchor. Every source character is
  // retained, including whitespace. The table is bounded; large inputs use lines.
  function alignTexts(texts, anchor) {
    const ids = Object.keys(texts).filter((id) => typeof texts[id] === "string");
    if (!ids.includes(anchor)) throw new Error("El texto guía no está disponible.");
    let granularity = "palabras";
    let tokens = Object.fromEntries(ids.map((id) => [id, texts[id].match(/\s*\S+|\s+$/g) || []]));
    const exceedsLimit = () => ids.some((id) => (tokens[anchor].length + 1) * (tokens[id].length + 1) > 1_500_000);
    if (exceedsLimit()) {
      granularity = "líneas";
      tokens = Object.fromEntries(ids.map((id) => [id, texts[id].match(/[^\n]*\n|[^\n]+$/g) || []]));
    }
    if (exceedsLimit()) {
      granularity = "bloques completos";
      tokens = Object.fromEntries(ids.map((id) => [id, texts[id] ? [texts[id]] : []]));
    }
    const a = tokens[anchor];
    const aligned = {};
    for (const id of ids.filter((value) => value !== anchor)) {
      const b = tokens[id];
      const columns = b.length + 1;
      const table = new Uint32Array((a.length + 1) * columns);
      for (let i = 0; i <= a.length; i += 1) table[i * columns + b.length] = a.length - i;
      for (let j = 0; j <= b.length; j += 1) table[a.length * columns + j] = b.length - j;
      const same = (i, j) => a[i].trim() === b[j].trim();
      for (let i = a.length - 1; i >= 0; i -= 1) {
        for (let j = b.length - 1; j >= 0; j -= 1) {
          table[i * columns + j] = Math.min(table[(i + 1) * columns + j + 1] + (same(i, j) ? 0 : 1), table[(i + 1) * columns + j] + 1, table[i * columns + j + 1] + 1);
        }
      }
      const cells = Array(a.length).fill("");
      const gaps = Array.from({ length: a.length + 1 }, () => []);
      let i = 0;
      let j = 0;
      while (i < a.length || j < b.length) {
        const score = table[i * columns + j];
        if (i < a.length && j < b.length && score === table[(i + 1) * columns + j + 1] + (same(i, j) ? 0 : 1)) { cells[i] = b[j]; i += 1; j += 1; }
        else if (i < a.length && score === table[(i + 1) * columns + j] + 1) i += 1;
        else { gaps[i].push(b[j]); j += 1; }
      }
      aligned[id] = { cells, gaps };
    }
    const rows = [];
    const size = granularity === "palabras" ? 12 : 1;
    let slots = 0;
    function append(values) {
      if (slots % size === 0) rows.push({ cells: Object.fromEntries(ids.map((id) => [id, []])), runs: [], changed: false });
      const row = rows.at(-1);
      const changed = ids.some((id) => values[id] !== values[anchor]);
      row.changed ||= changed;
      for (const id of ids) {
        const text = values[id];
        if (!text) continue;
        const kind = changed && (id === anchor || text !== values[anchor]) ? id === anchor ? "del" : "ins" : "same";
        const runs = row.cells[id];
        if (runs.at(-1)?.kind === kind) runs.at(-1).text += text;
        else runs.push({ kind, text });
      }
      if (ids.length === 2) {
        const other = ids.find((id) => id !== anchor);
        const appendRun = (kind, text) => {
          if (!text) return;
          if (row.runs.at(-1)?.kind === kind) row.runs.at(-1).text += text;
          else row.runs.push({ kind, text });
        };
        if (!changed) appendRun("same", values[anchor]);
        else { appendRun("del", values[anchor]); appendRun("ins", values[other]); }
      }
      slots += 1;
    }
    for (let i = 0; i <= a.length; i += 1) {
      const gaps = Math.max(0, ...Object.values(aligned).map((value) => value.gaps[i].length));
      for (let j = 0; j < gaps; j += 1) append(Object.fromEntries(ids.map((id) => [id, id === anchor ? "" : aligned[id].gaps[i][j] || ""])));
      if (i < a.length) append(Object.fromEntries(ids.map((id) => [id, id === anchor ? a[i] : aligned[id].cells[i]])));
    }
    return { rows, granularity };
  }

  function node(tag, text, className) {
    const result = document.createElement(tag);
    if (text !== undefined && text !== null) result.textContent = String(text);
    if (className) result.className = className;
    return result;
  }

  function assetURL(path) {
    if (typeof path !== "string" || !path || /^[\\/]/.test(path) || /[\u0000-\u001f\\]/.test(path) || /^[a-z][a-z0-9+.-]*:/i.test(path)) {
      throw new Error("El recurso publicado no tiene una ubicación válida.");
    }
    const decoded = decodeURIComponent(path.split(/[?#]/)[0]);
    if (decoded.split("/").some((part) => part === "..") || /[\\\u0000-\u001f]/.test(decoded)) {
      throw new Error("El recurso publicado no tiene una ubicación válida.");
    }
    const root = new URL("./", window.location.href);
    const target = new URL(path, root);
    if (target.origin !== root.origin || !target.pathname.startsWith(root.pathname)) {
      throw new Error("El recurso publicado no pertenece a este conjunto.");
    }
    return target.href;
  }

  function fetchJSON(path) {
    const url = assetURL(path);
    if (!cache.has(url)) {
      cache.set(url, fetch(url).then((response) => {
        if (!response.ok) throw new Error("No se pudo cargar un recurso del conjunto publicado.");
        return response.json();
      }).catch((error) => { cache.delete(url); throw error; }));
    }
    return cache.get(url);
  }

  function toast(message) {
    $("toast").textContent = message;
    $("toast").classList.remove("hidden");
    window.clearTimeout(toast.timer);
    toast.timer = window.setTimeout(() => $("toast").classList.add("hidden"), 2600);
  }

  function fatal(message) {
    $("fatalError").textContent = message;
    $("fatalError").classList.remove("hidden");
  }

  function badge(text, kind = "") {
    return node("span", text, `status-badge ${kind}`);
  }

  function chip(label, value) {
    const result = node("span", undefined, "summary-chip");
    result.append(node("span", label), node("strong", value));
    return result;
  }

  function methodState(item) {
    const test = summaryOf(item.test);
    if (item.state === "PLANNED") return ["Definido · sin entrenamiento", ""];
    if (item.state === "PENDING") return ["Enviado · en espera", ""];
    if (item.state === "RUNNING") return ["En curso", "active"];
    if (item.state === "INCOMPLETE_TEST" || (item.state === "COMPLETED" && test.all_pages_completed === false)) return ["Entrenamiento cerrado · test incompleto", "warning"];
    if (item.state === "COMPLETED") return ["Completado", "complete"];
    if (["FAILED", "CANCELLED", "TIMEOUT"].includes(item.state)) return ["Requiere revisión", "warning"];
    return ["Sin cierre confirmado", ""];
  }

  function evaluationStep(item) {
    const epoch = Number(item.epoch);
    return epoch >= 100 ? (Math.floor(epoch / 100) - 1) * 16 + epoch % 100 : epoch * 16;
  }

  function renderMethods() {
    $("methodCards").replaceChildren(...state.catalog.methods.map((item, index) => {
      const button = node("button", undefined, `method-card${item.id === state.method ? " selected" : ""}${item.exploratory ? " exploratory" : ""}`);
      button.type = "button";
      button.dataset.method = item.id;
      button.setAttribute("aria-pressed", String(item.id === state.method));
      button.append(node("span", item.exploratory ? "04 / EXTENSIÓN EXPLORATORIA" : `0${index + 1} / BERRUTTI`, "method-number"));
      button.append(node("span", item.label || labels[item.id] || item.id, "method-title"));
      button.append(node("span", descriptions[item.id] || "", "method-description"));
      const [text, color] = methodState(item);
      const status = node("span", undefined, "method-state");
      status.append(node("span", undefined, `state-dot ${color}`), node("span", text));
      button.append(status);
      const epochs = finite(item.epochs_completed) ? item.epochs_completed : 0;
      const total = finite(item.epochs_total) ? item.epochs_total : item.exploratory ? 4 : 3;
      const progress = node("span", undefined, "method-progress");
      const bar = node("span");
      const fraction = finite(item.optimizer_steps) ? item.optimizer_steps / (total * 16) : epochs / total;
      bar.style.width = `${Math.max(0, Math.min(100, fraction * 100))}%`;
      progress.append(bar);
      const progressCopy = node("span", undefined, "method-progress-copy");
      progressCopy.append(node("span", `${epochs}/${total} épocas cerradas`), node("span", `${count(item.optimizer_steps ?? (item.state === "PLANNED" ? 0 : null))}/${total * 16} actualizaciones`));
      button.append(progress, progressCopy);
      if (item.progress && finite(item.progress.images_completed) && item.progress.images_total > 0) {
        const current = item.progress;
        button.append(node("span", `Época ${current.epoch}/${total} · ${Math.round(current.images_completed / current.images_total * 100)} % confirmado · ${count(current.images_completed)}/${count(current.images_total)} imágenes`, "method-current"));
      }
      const scores = node("span", undefined, "method-scores");
      const validation = node("span");
      const latest = (state.catalog.evaluations || []).filter((row) => row.method === item.id && normalizedSplit(row.split) === "validation" && !isBaseline(row)).sort((a, b) => evaluationStep(a) - evaluationStep(b)).at(-1);
      validation.append(node("span", item.selection ? "CER validación elegida" : "Última validación"), node("strong", percent(item.selection?.validation_cer ?? summaryOf(latest?.summary).cer)));
      if (item.selection?.epoch) validation.append(node("span", `Época ${item.selection.epoch}`));
      else if (latest) validation.append(node("span", Number(latest.epoch) >= 100 ? `Época ${Math.floor(Number(latest.epoch) / 100)} · control intermedio` : `Cierre de época ${latest.epoch}`));
      if (latest && !item.selection) {
        const summary = summaryOf(latest.summary);
        validation.append(node("small", `${count(summary.pages_completed)}/${count(summary.pages)} páginas · ${count(summary.failed_pages)} fallidas · ${count(summary.retried_pages)} reintentadas${summary.provisional ? " · provisional" : ""}`));
      }
      const testSummary = summaryOf(item.test);
      const test = node("span");
      test.append(node("span", "CER test"), node("strong", percent(testSummary.cer)));
      if (testSummary.all_pages_completed === false) test.append(node("small", "Provisional · test incompleto"));
      else if (finite(testSummary.cer)) test.append(node("span", "Descriptivo"));
      else test.append(node("span", "Pendiente del checkpoint final"));
      scores.append(validation, test);
      button.append(scores);
      button.addEventListener("click", () => chooseMethod(item.id));
      return button;
    }));
    $("methodLabel").textContent = method().label || labels[state.method] || state.method;
    $("exploratoryNote").classList.toggle("hidden", !method().exploratory);
    $("protocolNote").textContent = method().exploratory
      ? "Extensión separada: 3 épocas mixtas + 1 de consolidación. Baseline propia con la misma política de generación. Sólo los cierres de época seleccionan checkpoint; test único al final."
      : "Tres modelos Berrutti · 3 épocas · 16 actualizaciones por época · inicialización común. Selección por validación completa; test descriptivo.";
  }

  function renderBaseline() {
    const target = $("baselineSummary");
    const extension = state.method === "experiment4";
    const baseline = extension ? state.catalog.experiment4_baseline : state.catalog.shared_baseline;
    target.replaceChildren();
    if (!baseline) { target.classList.add("hidden"); return; }
    target.classList.remove("hidden");
    target.append(node("span", extension ? "Línea base propia · extensión LUISA" : "Línea base compartida", "baseline-label"));
    const all = (state.catalog.evaluations || []).filter((item) => extension ? item.method === "experiment4_baseline" : item.method === "baseline" || item.method === "shared");
    for (const [split, label] of [["validation", "Validación"], ["test", "Test"]]) {
      let value = all.find((item) => normalizedSplit(item.split) === split)?.summary;
      if (!value) value = summaryOf(baseline[split]);
      if (!finite(value?.cer)) continue;
      const span = node("span", undefined, "baseline-score");
      span.append(node("span", `${label} `), node("strong", percent(value.cer)));
      span.append(node("span", ` · ${count(value.pages)} páginas · ${count(value.failed_pages ?? 0)} fallidas · ${count(value.retried_pages ?? 0)} reintentadas`));
      if (value.all_pages_completed === false) span.append(node("span", " · provisional"));
      target.append(span);
    }
    target.append(node("span", "Antes del ajuste · cero actualizaciones", "quiet"));
  }

  function availableEvaluations() {
    return (state.catalog.evaluations || []).filter((item) => normalizedSplit(item.split) === state.split &&
      (item.method === state.method || (state.method === "experiment4" && item.method === "experiment4_baseline") || (state.method !== "experiment4" && ["shared", "baseline"].includes(item.method))));
  }

  function evaluationLabel(item) {
    if (item.label) return item.label;
    if (item.stage === "baseline" || Number(item.epoch) === 0) return ["experiment4", "experiment4_baseline"].includes(item.method) ? "Línea base propia · sin entrenar" : "Línea base compartida · sin entrenar";
    if (Number(item.epoch) >= 100) return `Época ${Math.floor(Number(item.epoch) / 100)} · update ${Number(item.epoch) % 100} · control`;
    const chosen = Number(item.epoch) === Number(method().selection?.epoch);
    const test = state.split === "test" ? " · checkpoint elegido" : chosen ? " · elegida" : "";
    return `Época ${item.epoch}${test}`;
  }

  function renderControls(resetEvaluation = false) {
    document.querySelectorAll("[data-split]").forEach((button) => {
      const selected = button.dataset.split === state.split;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    const training = state.split === "train";
    const options = availableEvaluations().sort((a, b) => evaluationStep(a) - evaluationStep(b));
    if (resetEvaluation || !options.some((item) => item.id === state.evaluation)) {
      const preferred = options.find((item) => Number(item.epoch) === Number(method().selection?.epoch) && item.stage !== "baseline");
      state.evaluation = preferred?.id || options.at(-1)?.id || "";
    }
    $("evaluationSelect").replaceChildren(...options.map((item) => {
      const option = node("option", evaluationLabel(item));
      option.value = item.id;
      return option;
    }));
    if (!options.length) {
      const option = node("option", "Sin evaluación publicada");
      option.value = "";
      $("evaluationSelect").append(option);
    }
    $("evaluationSelect").value = state.evaluation;
    $("evaluationSelect").disabled = !options.length;
    $("evaluationField").classList.toggle("hidden", training);
    $("familyField").classList.toggle("hidden", !training);
    for (const value of ["synthetic", "luisa"]) {
      $("familySelect").querySelector(`[value="${value}"]`).disabled = state.method !== "experiment4";
    }
    $("familySelect").value = state.family;
    $("syntheticEpochField").classList.toggle("hidden", !training || state.family !== "synthetic");
    $("columnsField").classList.toggle("hidden", !training || state.family !== "synthetic");
    renderContext();
    renderEvaluationSummary();
  }

  function renderContext() {
    let text;
    if (state.split !== "train") {
      text = state.split === "validation" ? "25 páginas · validación al final de cada época." : "25 páginas · test del checkpoint elegido por validación.";
      if (state.method === "experiment4" && !availableEvaluations().length) text = "Sin evaluaciones publicadas de esta extensión.";
    } else if (state.family === "synthetic") {
      text = "Composiciones de líneas Berrutti y LUISA, por época y número de columnas.";
    } else if (state.family === "luisa") {
      text = "23.774 líneas LUISA, incluido su test de origen.";
    } else if (state.method === "lines" && state.family === "pages") {
      text = "Páginas de origen de los 5.212 recortes usados por este modelo.";
    } else if (state.method === "pages" && state.family === "lines") {
      text = "Recortes de las 125 páginas train; este modelo entrena con la página completa.";
    } else {
      text = state.family === "lines" ? "5.212 recortes de las 125 páginas train." : "125 páginas train y sus recortes de línea.";
    }
    if (state.split === "train") text += " Sin OCR evaluado de entrenamiento.";
    $("contextNote").textContent = text;
  }

  function renderEvaluationSummary() {
    const target = $("evaluationSummary");
    target.replaceChildren();
    if (state.split === "train") {
      const training = method().training_history || {};
      const history = Array.isArray(training) ? training : training.epochs || [];
      const source = state.catalog.training_sources || {};
      const expected = state.family === "luisa" ? source.luisa_lines : state.family === "synthetic" ? source.synthetic_pages : state.family === "lines" ? source.berrutti_lines : source.berrutti_pages;
      target.append(chip(state.family === "lines" || state.family === "luisa" ? "Líneas" : "Páginas", count(expected)));
      target.append(badge(methodState(method())[0], ["COMPLETED", "INCOMPLETE_TEST"].includes(method().state) ? "" : "neutral"));
      if (method().progress) {
        const current = method().progress;
        target.append(chip("Época en curso", `${current.epoch}/${method().epochs_total}`), chip("Imágenes confirmadas", `${count(current.images_completed)}/${count(current.images_total)}`));
      }
      if (history.length) {
        const wrap = node("div", undefined, "training-history");
        const table = node("table");
        const head = node("thead");
        const row = node("tr");
        for (const label of ["Época", "Pérdida media", "Updates", "Imágenes presentadas", "Páginas de origen"]) row.append(node("th", label));
        head.append(row);
        const body = node("tbody");
        for (const entry of history) {
          const line = node("tr");
          for (const value of [entry.epoch, finite(entry.mean_loss) ? decimal.format(entry.mean_loss) : "—", count(entry.optimizer_steps), count(entry.samples), count(entry.source_pages)]) line.append(node("td", value));
          body.append(line);
        }
        table.append(head, body);
        wrap.append(table);
        target.append(wrap);
        if (training.note) target.append(node("span", training.note, "quiet"));
      }
      return;
    }
    const item = evaluation();
    if (!item) { target.append(badge("Sin OCR publicado · referencias disponibles", "neutral")); return; }
    const summary = summaryOf(item);
    const complete = summary.all_pages_completed === true;
    const completedPages = summary.pages_completed ?? (finite(summary.pages) && finite(summary.failed_pages) ? summary.pages - summary.failed_pages : null);
    target.append(chip("CER", percent(summary.cer)), chip("Páginas completas", `${count(completedPages)}/${count(summary.pages)}`), chip("Fallidas", count(summary.failed_pages ?? 0)), chip("Reintentadas", count(summary.retried_pages ?? 0)));
    target.append(badge(complete ? "Evaluación completa" : "CER provisional · evaluación incompleta", complete ? "" : "warning"));
    if (item.stage === "baseline" || Number(item.epoch) === 0) target.append(node("span", "Evaluación previa, sin entrenamiento", "quiet"));
    else if (Number(item.epoch) >= 100 || item.stage === "progress") target.append(node("span", "Control de avance · excluido de la selección", "quiet"));
    if (!complete) target.append(node("span", "No acredita mejoras; se conserva el estado de cada página.", "quiet"));
  }

  async function chooseMethod(id) {
    if (state.method === id) return;
    state.method = id;
    if (state.method !== "experiment4" && ["synthetic", "luisa"].includes(state.family)) state.family = "pages";
    if (state.split === "train") state.family = id === "lines" ? "lines" : "pages";
    state.query = "";
    $("documentSearch").value = "";
    renderMethods();
    renderBaseline();
    renderControls(true);
    await refreshIndex();
  }

  async function chooseSplit(split) {
    state.split = split;
    state.family = split === "train" && state.method === "lines" ? "lines" : "pages";
    state.query = "";
    $("documentSearch").value = "";
    renderControls(true);
    await refreshIndex();
  }

  function filteredIndex() {
    const query = state.query.toLocaleLowerCase("es").trim();
    return state.index.filter((item) => (!query || String(item.id).toLocaleLowerCase("es").includes(query)) &&
      (!state.syntheticEpoch || state.family !== "synthetic" || Number(item.epoch) === Number(state.syntheticEpoch)) &&
      (!state.columns || state.family !== "synthetic" || Number(item.columns) === Number(state.columns)));
  }

  function renderDocumentList() {
    const items = filteredIndex();
    state.listPage = Math.max(0, Math.min(state.listPage, Math.max(0, Math.ceil(items.length / LIST_SIZE) - 1)));
    const start = state.listPage * LIST_SIZE;
    $("documentCount").textContent = `${count(items.length)} imágenes`;
    $("listRange").textContent = items.length ? `${start + 1}–${Math.min(start + LIST_SIZE, items.length)}` : "";
    $("listPage").textContent = items.length ? `${state.listPage + 1} / ${Math.ceil(items.length / LIST_SIZE)}` : "0 / 0";
    $("listPrevious").disabled = state.listPage === 0;
    $("listNext").disabled = start + LIST_SIZE >= items.length;
    $("documentList").replaceChildren(...items.slice(start, start + LIST_SIZE).map((item) => {
      const button = node("button", item.id, `document-item${item.id === state.document ? " selected" : ""}`);
      button.type = "button";
      button.dataset.document = item.id;
      button.setAttribute("aria-pressed", String(item.id === state.document));
      const detail = state.family === "synthetic" ? `Época ${item.epoch} · ${item.columns} ${Number(item.columns) === 1 ? "columna" : "columnas"}` : finite(item.line_count) ? `${count(item.line_count)} líneas` : state.family === "luisa" ? "Línea LUISA" : "Página Berrutti";
      button.append(node("small", detail));
      button.addEventListener("click", () => selectDocument(item.id));
      return button;
    }));
    const current = items.findIndex((item) => item.id === state.document);
    $("documentPrevious").disabled = current <= 0;
    $("documentNext").disabled = current < 0 || current >= items.length - 1;
  }

  function clearDocument(message, error = false) {
    state.detail = null;
    state.prediction = null;
    state.basePrediction = null;
    $("documentContent").classList.add("hidden");
    $("documentLoading").classList.remove("hidden");
    $("documentLoading").classList.toggle("error", error);
    $("documentLoading").textContent = message;
    $("pageStatus").replaceChildren();
    $("lineGallery").replaceChildren();
    $("referenceText").textContent = "";
    $("predictionText").textContent = "";
    $("baseText").textContent = "";
    $("diffText").replaceChildren();
    $("pageImage").removeAttribute("src");
    $("lineBoxes").replaceChildren();
  }

  async function refreshIndex() {
    const request = ++state.request;
    const oldDocument = state.document;
    state.indexLoading = true;
    state.index = [];
    state.listPage = 0;
    clearDocument("Cargando imágenes…");
    renderDocumentList();
    try {
      let items;
      if (state.split === "train" && state.family === "synthetic") {
        if (!state.catalog.synthetic_index) items = [];
        else items = (await fetchJSON(state.catalog.synthetic_index)).pages || [];
      } else if (state.split === "train" && state.family === "luisa") {
        if (!state.catalog.luisa_index) items = [];
        else items = (await fetchJSON(state.catalog.luisa_index)).lines || [];
      } else {
        items = state.catalog.pages.filter((item) => normalizedSplit(item.split) === state.split);
      }
      if (request !== state.request) return;
      state.indexLoading = false;
      state.index = items;
      const filtered = filteredIndex();
      state.document = filtered.some((item) => item.id === oldDocument) ? oldDocument : filtered[0]?.id || "";
      state.listPage = Math.max(0, Math.floor(filtered.findIndex((item) => item.id === state.document) / LIST_SIZE));
      renderDocumentList();
      if (state.document) await selectDocument(state.document);
      else { $("documentTitle").textContent = "Sin imágenes publicadas"; clearDocument("Todavía no hay imágenes publicadas para esta selección."); }
    } catch (_) {
      if (request !== state.request) return;
      state.indexLoading = false;
      clearDocument("No se pudo cargar este conjunto de imágenes. Recargá la página para volver a intentarlo.", true);
    }
  }

  async function selectDocument(id) {
    const item = state.index.find((entry) => entry.id === id);
    if (!item) return;
    state.document = id;
    state.selectedLine = "";
    state.linePage = 0;
    state.lineQuery = "";
    state.zoom = 100;
    $("lineSearch").value = "";
    $("documentTitle").textContent = item.id;
    $("documentKind").textContent = state.family === "synthetic" ? `Sintética · época ${item.epoch} · ${item.columns} ${Number(item.columns) === 1 ? "columna" : "columnas"}` : state.family === "luisa" ? "Línea LUISA" : "Página Berrutti";
    renderDocumentList();
    clearDocument("Cargando imagen y transcripciones…");
    const request = ++state.request;
    const evalItem = state.split !== "train" ? evaluation()?.documents?.find((entry) => entry.page_id === id || entry.id === id) : null;
    try {
      const baseItem = state.split !== "train" ? baselineEvaluation(state.catalog, state.method, state.split)?.documents?.find((entry) => entry.page_id === id || entry.id === id) : null;
      const [detail, prediction, basePrediction] = await Promise.all([
        item.detail ? fetchJSON(item.detail) : Promise.resolve(item),
        evalItem?.detail ? fetchJSON(evalItem.detail) : Promise.resolve(null),
        baseItem?.detail ? fetchJSON(baseItem.detail) : Promise.resolve(null),
      ]);
      if (request !== state.request) return;
      for (const value of [prediction, basePrediction]) {
        if (value && (value.page_id !== id || value.reference !== detail.reference || value.normalization_policy !== state.catalog.normalization_policy)) throw new Error("La evaluación no coincide con esta página y su referencia.");
      }
      state.detail = detail;
      state.prediction = prediction;
      state.basePrediction = basePrediction;
      const imagePath = item.image || detail.image;
      if (!imagePath) throw new Error("Imagen no disponible.");
      $("pageImage").src = assetURL(imagePath);
      $("pageImage").alt = `Imagen ${id}`;
      $("pageImage").onload = renderBoxes;
      $("pageImage").onerror = () => {
        if (request === state.request) toast("No se pudo cargar la imagen; las transcripciones siguen disponibles.");
      };
      setZoom(100);
      $("documentLoading").classList.add("hidden");
      $("documentContent").classList.remove("hidden");
      renderPageStatus();
      renderText();
      const lines = lineList();
      $("linesDetails").classList.toggle("hidden", !lines.length);
      $("linesDetails").open = state.family === "lines";
      $("boxesField").classList.toggle("hidden", !lines.some((line) => Array.isArray(line.box)));
      $("linesTitle").textContent = state.family === "synthetic" ? "Líneas que componen esta página" : "Recortes y referencias de líneas";
      $("linesCount").textContent = `${count(lines.length)} líneas`;
      $("linesNote").textContent = state.family === "synthetic" ? "Líneas Berrutti y LUISA con su procedencia." : "La referencia de cada recorte puede diferir de la de la página.";
      if ($("linesDetails").open) renderLines();
      renderBoxes();
    } catch (_) {
      if (request !== state.request) return;
      clearDocument("No se pudieron cargar los detalles de esta imagen. Elegí otra o recargá la página para volver a intentarlo.", true);
    }
  }

  function renderPageStatus() {
    const target = $("pageStatus");
    target.replaceChildren();
    const prediction = state.prediction;
    if (state.family === "luisa") {
      const source = state.detail.source_split;
      target.append(badge(source === "test" ? "LUISA · test de origen incluido" : "LUISA · referencia original", source === "test" ? "warning" : "neutral"));
      return;
    }
    if (state.split === "train") { target.append(badge("Dato de entrenamiento · referencia humana", "neutral")); return; }
    if (!prediction) { target.append(badge("Sin transcripción publicada para este checkpoint", "neutral")); return; }
    const complete = prediction.complete === true && !prediction.failed;
    target.append(badge(complete ? "Generación completa" : "Generación incompleta · provisional", complete ? "" : "warning"));
    if (finite(prediction.cer)) target.append(node("span", `CER página ${percent(prediction.cer)}`));
    if (finite(prediction.wer)) target.append(node("span", `WER ${percent(prediction.wer)}`));
    const attempts = Array.isArray(prediction.attempts) ? prediction.attempts : [];
    if (attempts.length) {
      const reasons = { eos: "fin de secuencia", repetition: "repetición", time_limit: "límite de tiempo", token_limit: "límite de tokens", error: "error" };
      target.append(node("span", `${attempts.length}/2 intentos · ${attempts.map((attempt) => reasons[attempt.stop_reason] || "estado registrado").join(" → ")}`));
    }
    if (prediction.retried) target.append(node("span", "Página reintentada"));
  }

  function textValue(value, name) {
    const key = state.textMode === "normalized" ? `${name}_normalized` : name;
    return typeof value?.[key] === "string" ? value[key] : null;
  }

  function renderText() {
    const reference = textValue(state.detail, "reference");
    const base = textValue(state.basePrediction, "prediction");
    const tuned = tunedPrediction();
    const prediction = textValue(tuned, "prediction");
    $("referenceText").textContent = reference ?? "Referencia no disponible en esta representación.";
    const training = state.split === "train";
    const visible = training ? ["reference"] : transcriptionViews[state.transcriptionView] || transcriptionViews.all;
    for (const [id, pane] of [["reference", "referencePane"], ["base", "basePane"], ["tuned", "predictionPane"]]) $(pane).classList.toggle("hidden", !visible.includes(id));
    document.querySelector(".text-grid").dataset.count = visible.length;
    $("transcriptionView").value = training ? "reference" : state.transcriptionView;
    $("transcriptionView").disabled = training;
    $("baseText").textContent = base ?? "Todavía no hay una transcripción base publicada para esta página.";
    $("predictionText").textContent = prediction ?? (isBaseline(evaluation()) ? "Seleccioná una época ajustada para compararla con el modelo base." : "Todavía no hay una transcripción ajustada publicada para esta selección.");
    $("predictionLabel").textContent = tuned ? `Modelo ajustado · ${evaluationLabel(evaluation())}` : "Modelo ajustado";
    const status = (value) => value ? `${value.complete === true && !value.failed ? "Completa" : "Incompleta · provisional"}${finite(value.cer) ? ` · CER ${percent(value.cer)}` : ""}${value.retried ? " · reintentada" : ""}` : "Sin evaluación publicada";
    $("baseTextStatus").textContent = state.basePrediction ? `Cero actualizaciones · ${status(state.basePrediction)}` : status(null);
    $("predictionTextStatus").textContent = status(tuned);
    $("copyReference").disabled = reference === null;
    $("copyBase").disabled = base === null;
    $("copyPrediction").disabled = prediction === null;
    $("normalizationNote").classList.toggle("hidden", state.textMode !== "normalized");
    $("normalizationNote").textContent = `Normalización: ${state.catalog.normalization_policy || "publicada"}.`;
    document.querySelectorAll("[data-text-mode]").forEach((button) => {
      const selected = button.dataset.textMode === state.textMode;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    $("diffDetails").classList.toggle("hidden", training || [reference, base, prediction].filter((value) => value !== null).length < 2);
    $("diffText").replaceChildren();
    if ($("diffDetails").open && !training) renderDiff();
  }

  function renderDiff() {
    const target = $("diffText");
    target.replaceChildren();
    const texts = { reference: textValue(state.detail, "reference"), base: textValue(state.basePrediction, "prediction"), tuned: textValue(tunedPrediction(), "prediction") };
    const available = Object.keys(texts).filter((id) => texts[id] !== null);
    if (available.length < 2) return;
    if (!available.includes(state.diffReference)) state.diffReference = available[0];
    if (!available.includes(state.diffCompared) || state.diffCompared === state.diffReference) state.diffCompared = available.find((id) => id !== state.diffReference);
    for (const option of $("diffReference").options) option.disabled = !available.includes(option.value);
    for (const option of $("diffCompared").options) option.disabled = !available.includes(option.value) || option.value === state.diffReference;
    $("diffReference").value = state.diffReference;
    $("diffCompared").value = state.diffCompared;
    const ids = [state.diffReference, state.diffCompared];
    const { rows, granularity } = alignTexts(Object.fromEntries(ids.map((id) => [id, texts[id]])), state.diffReference);
    const onlyChanges = $("showOnlyChanges").checked;
    const changed = rows.filter((row) => row.changed).length;
    const provisional = ids.map((id) => id === "base" ? state.basePrediction : id === "tuned" ? tunedPrediction() : null).some((value) => value && (value.complete !== true || value.failed));
    $("alignmentStatus").textContent = `${changed}/${rows.length} fragmentos con diferencias · ${granularity}.${provisional ? " Generación incompleta: comparación provisional." : ""}`;
    const body = node("pre", undefined, "diff-content");
    body.classList.toggle("filtered", onlyChanges);
    rows.forEach((row, index) => {
      if (onlyChanges && !row.changed) return;
      const fragment = node("span", undefined, row.changed ? "changed" : "unchanged");
      fragment.dataset.fragment = index + 1;
      fragment.append(...row.runs.map((run) => run.kind === "same" ? document.createTextNode(run.text) : node(run.kind, run.text)));
      body.append(fragment);
    });
    target.append(body);
    if (!body.childNodes.length) target.append(node("p", onlyChanges ? "Sin diferencias." : "Ambos textos están vacíos.", "alignment-empty"));
  }

  function setZoom(value) {
    state.zoom = Math.max(25, Math.min(400, value));
    $("imageCanvas").style.width = `${state.zoom}%`;
    $("zoomValue").textContent = `${state.zoom} %`;
    $("zoomOut").disabled = state.zoom === 25;
    $("zoomIn").disabled = state.zoom === 400;
  }

  function renderBoxes() {
    const image = $("pageImage");
    const target = $("lineBoxes");
    target.replaceChildren();
    if (!$("showBoxes").checked || !image.naturalWidth || !image.naturalHeight) return;
    lineList().forEach((line, index) => {
      if (!Array.isArray(line.box) || line.box.length !== 4 || !line.box.every(finite)) return;
      const [x, y, width, height] = line.box;
      if (width <= 0 || height <= 0) return;
      const box = node("button", undefined, `line-box${state.selectedLine === String(line.id) ? " active" : ""}`);
      box.type = "button";
      box.setAttribute("aria-label", `Seleccionar línea ${index + 1}`);
      box.title = `Línea ${index + 1}`;
      box.style.left = `${x / image.naturalWidth * 100}%`;
      box.style.top = `${y / image.naturalHeight * 100}%`;
      box.style.width = `${width / image.naturalWidth * 100}%`;
      box.style.height = `${height / image.naturalHeight * 100}%`;
      box.addEventListener("click", () => {
        state.selectedLine = String(line.id);
        state.lineQuery = "";
        $("lineSearch").value = "";
        state.linePage = Math.floor(index / LINES_SIZE);
        $("linesDetails").open = true;
        renderLines();
        renderBoxes();
        $("lineGallery").querySelector(".selected")?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
      target.append(box);
    });
  }

  function renderLines() {
    const query = state.lineQuery.trim().toLocaleLowerCase("es");
    const lines = lineList().filter((line) => !query || `${line.id} ${line.reference || ""}`.toLocaleLowerCase("es").includes(query));
    state.linePage = Math.max(0, Math.min(state.linePage, Math.max(0, Math.ceil(lines.length / LINES_SIZE) - 1)));
    const start = state.linePage * LINES_SIZE;
    $("lineRange").textContent = lines.length ? `${start + 1}–${Math.min(start + LINES_SIZE, lines.length)} de ${count(lines.length)}` : "Sin coincidencias";
    $("linesPrevious").disabled = state.linePage === 0;
    $("linesNext").disabled = start + LINES_SIZE >= lines.length;
    $("lineGallery").replaceChildren(...lines.slice(start, start + LINES_SIZE).map((line, index) => {
      const card = node("section", undefined, `line-card${state.selectedLine === String(line.id) ? " selected" : ""}`);
      const heading = node("div", undefined, "line-card-header");
      heading.append(node("span", `${start + index + 1} · ${line.id}`));
      if (line.image) {
        const link = node("a", "Ampliar ↗");
        link.href = assetURL(line.image);
        link.target = "_blank";
        link.rel = "noopener";
        heading.append(link);
        const image = node("img");
        image.src = assetURL(line.image);
        image.loading = "lazy";
        image.decoding = "async";
        image.alt = `Recorte de línea ${start + index + 1}`;
        image.addEventListener("click", () => showImage(line.image, `Línea ${line.id}`));
        card.append(heading, image);
      } else card.append(heading);
      card.append(node("pre", textValue(line, "reference") ?? "Referencia no disponible."));
      if (typeof line.source === "string") card.append(node("p", `Origen: ${line.source}`, "source"));
      return card;
    }));
  }

  function showImage(path, title) {
    if (!path) return;
    $("dialogTitle").textContent = title;
    $("dialogImage").src = assetURL(path);
    $("dialogImage").alt = title;
    $("originalImageLink").href = assetURL(path);
    $("imageDialog").showModal();
  }

  function currentImage() {
    return state.index.find((item) => item.id === state.document)?.image || state.detail?.image;
  }

  function navigateDocument(direction) {
    const items = filteredIndex();
    const index = items.findIndex((item) => item.id === state.document) + direction;
    if (index < 0 || index >= items.length) return;
    state.listPage = Math.floor(index / LIST_SIZE);
    selectDocument(items[index].id);
  }

  async function copyText(id) {
    try { await navigator.clipboard.writeText($(id).textContent); toast("Texto copiado"); }
    catch (_) { toast("Seleccioná el texto para copiarlo; el navegador no habilitó el portapapeles."); }
  }

  function bindEvents() {
    document.querySelectorAll("[data-split]").forEach((button) => button.addEventListener("click", () => chooseSplit(button.dataset.split)));
    $("familySelect").addEventListener("change", async () => {
      state.family = $("familySelect").value;
      state.query = "";
      $("documentSearch").value = "";
      renderControls();
      await refreshIndex();
    });
    $("evaluationSelect").addEventListener("change", () => {
      state.evaluation = $("evaluationSelect").value;
      renderEvaluationSummary();
      if (state.document) selectDocument(state.document);
    });
    for (const [id, key] of [["syntheticEpochSelect", "syntheticEpoch"], ["columnsSelect", "columns"]]) {
      $(id).addEventListener("change", () => { state[key] = $(id).value; filterDocuments(); });
    }
    $("documentSearch").addEventListener("input", () => { state.query = $("documentSearch").value; filterDocuments(); });
    $("listPrevious").addEventListener("click", () => { state.listPage -= 1; renderDocumentList(); });
    $("listNext").addEventListener("click", () => { state.listPage += 1; renderDocumentList(); });
    $("documentPrevious").addEventListener("click", () => navigateDocument(-1));
    $("documentNext").addEventListener("click", () => navigateDocument(1));
    $("zoomOut").addEventListener("click", () => setZoom(state.zoom - 25));
    $("zoomIn").addEventListener("click", () => setZoom(state.zoom + 25));
    $("zoomReset").addEventListener("click", () => setZoom(100));
    $("showBoxes").addEventListener("change", renderBoxes);
    $("expandImage").addEventListener("click", () => showImage(currentImage(), state.document));
    $("pageImage").addEventListener("dblclick", () => showImage(currentImage(), state.document));
    $("closeImageDialog").addEventListener("click", () => $("imageDialog").close());
    $("imageDialog").addEventListener("click", (event) => { if (event.target === $("imageDialog")) $("imageDialog").close(); });
    document.querySelectorAll("[data-text-mode]").forEach((button) => button.addEventListener("click", () => {
      state.textMode = button.dataset.textMode;
      renderText();
      if ($("linesDetails").open) renderLines();
    }));
    $("diffDetails").addEventListener("toggle", () => {
      if ($("diffDetails").open && state.detail) renderDiff();
    });
    $("transcriptionView").addEventListener("change", () => { state.transcriptionView = $("transcriptionView").value; renderText(); });
    $("diffReference").addEventListener("change", () => {
      const previous = state.diffReference;
      state.diffReference = $("diffReference").value;
      if (state.diffCompared === state.diffReference) state.diffCompared = previous;
      renderDiff();
    });
    $("diffCompared").addEventListener("change", () => { state.diffCompared = $("diffCompared").value; renderDiff(); });
    $("swapComparison").addEventListener("click", () => { [state.diffReference, state.diffCompared] = [state.diffCompared, state.diffReference]; renderDiff(); });
    $("showOnlyChanges").addEventListener("change", renderDiff);
    $("linesDetails").addEventListener("toggle", () => { if ($("linesDetails").open) renderLines(); });
    $("lineSearch").addEventListener("input", () => { state.lineQuery = $("lineSearch").value; state.linePage = 0; renderLines(); });
    $("linesPrevious").addEventListener("click", () => { state.linePage -= 1; renderLines(); });
    $("linesNext").addEventListener("click", () => { state.linePage += 1; renderLines(); });
    $("copyReference").addEventListener("click", () => copyText("referenceText"));
    $("copyBase").addEventListener("click", () => copyText("baseText"));
    $("copyPrediction").addEventListener("click", () => copyText("predictionText"));
  }

  function filterDocuments() {
    if (state.indexLoading) return;
    state.listPage = 0;
    const items = filteredIndex();
    renderDocumentList();
    if (!items.some((item) => item.id === state.document)) {
      if (items.length) selectDocument(items[0].id);
      else { state.request += 1; state.document = ""; $("documentTitle").textContent = "Sin coincidencias"; clearDocument("No hay imágenes que coincidan con estos filtros."); }
    }
  }

  async function init() {
    try {
      const catalog = await fetchJSON("catalog.json");
      if (catalog.schema_version !== 1 || !Array.isArray(catalog.methods) || !catalog.methods.length || !Array.isArray(catalog.pages)) throw new Error("Catálogo no compatible.");
      state.catalog = catalog;
      state.method = catalog.methods.some((item) => item.id === "pages") ? "pages" : catalog.methods[0].id;
      if (catalog.title) document.title = `${catalog.title} · Qwen–Berrutti`;
      const date = new Date(catalog.observed_at || catalog.generated_at);
      const formatter = new Intl.DateTimeFormat("es-UY", { dateStyle: "medium", timeStyle: "short", timeZone: "Europe/Paris" });
      $("snapshotDate").textContent = Number.isNaN(date.valueOf()) ? "Publicación de resultados" : `${catalog.observed_at ? "Evidencia del" : "Publicado"} ${formatter.format(date)} · París`;
      const published = new Date(catalog.generated_at);
      if (!Number.isNaN(published.valueOf())) $("snapshotDate").title = `Publicación cerrada: ${formatter.format(published)} · París`;
      if (catalog.manifest) { $("manifestLink").href = assetURL(catalog.manifest); $("manifestLink").classList.remove("hidden"); }
      renderMethods();
      renderBaseline();
      renderControls(true);
      bindEvents();
      await refreshIndex();
    } catch (_) {
      clearDocument("No se pudo abrir esta publicación.", true);
      fatal(window.location.protocol === "file:" ? "Abrí el viewer a través de un servidor web para cargar sus imágenes y transcripciones." : "No se pudo cargar el catálogo de esta publicación. Recargá la página o verificá que el conjunto esté completo.");
    }
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { alignTexts, baselineEvaluation, evaluationStep };
  if (typeof document !== "undefined") init();
})();
