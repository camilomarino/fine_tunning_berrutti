(() => {
  "use strict";
  const data = window.FINETUNING_RESULTS;
  const models = new Map(data.models.map(model => [model.id, model]));
  const split = data.split;
  const pct = value => `${(100 * value).toLocaleString("es-ES", {minimumFractionDigits: 3, maximumFractionDigits: 3})} %`;

  window.OCR_DIFF_VIEWER_DATA.ready.then(index => {
    function displayNames() {
      for (const button of document.querySelectorAll("#summaryTable .model-doc-sort")) {
        const model = models.get(button.dataset.model);
        if (!model) continue;
        if (button.textContent.startsWith(model.id)) {
          button.textContent = model.name + button.textContent.slice(model.id.length);
        }
        button.title = model.training_description;
      }
      for (const option of document.querySelectorAll("#modelSelect option")) {
        const model = models.get(option.value);
        if (model && option.textContent.endsWith(model.id)) {
          option.textContent = option.textContent.slice(0, -model.id.length) + model.name;
          option.title = model.training_description;
        }
      }
      const ocrTitle = document.getElementById("ocrTitle");
      const current = models.get(ocrTitle.textContent);
      if (current) ocrTitle.textContent = current.name;
      const methodTitle = document.getElementById("method-details-title");
      for (const model of models.values()) {
        if (methodTitle.textContent.startsWith(model.id + " · ")) {
          methodTitle.textContent = model.name;
          document.getElementById("method-details-description").textContent = model.training_description;
          break;
        }
      }
    }
    // Keep the v2 IDs, data and UI assets intact, including existing deep links.
    const originalSearchMatcher = searchMatcher;
    searchMatcher = query => {
      const matches = originalSearchMatcher(query);
      return value => matches(value) || (models.has(value) && matches(models.get(value).name));
    };
    const originalHomeModels = homeModels;
    homeModels = () => {
      const rows = originalHomeModels();
      if (state.homeSort.type === "alpha") {
        const direction = state.homeSort.direction === "desc" ? -1 : 1;
        rows.sort((a, b) => models.get(a.id).name.localeCompare(models.get(b.id).name, "es") * direction);
      }
      return rows;
    };
    const originalRenderHomeTable = renderHomeTable;
    renderHomeTable = () => { originalRenderHomeTable(); displayNames(); };
    const originalRenderModels = renderModels;
    renderModels = () => { originalRenderModels(); displayNames(); };
    // In v2 the figures controls are initialized only with entity evidence.
    // This comparison contains CER/WER receipts, so omit the inactive control.
    if (!index.coverage) document.getElementById("summaryViewSelector").classList.add("hidden");
    const header = document.querySelector(".topbar");
    const controls = document.createElement("div");
    controls.className = "finetuning-controls";
    const label = document.createElement("label");
    label.textContent = "Partición ";
    const select = document.createElement("select");
    select.id = "finetuningSplit";
    select.setAttribute("aria-label", "Partición de evaluación");
    for (const [id, name] of [["validation", "Validación"], ["test", "Test"]]) {
      const option = document.createElement("option");
      option.value = id; option.textContent = name; select.append(option);
    }
    select.value = split; label.append(select); controls.append(label);
    const baseline = document.createElement("button");
    baseline.type = "button"; baseline.id = "finetuningBaseline";
    baseline.className = "subtle-button"; baseline.textContent = "Ver base";
    controls.append(baseline);
    const scores = document.createElement("button");
    scores.type = "button"; scores.id = "finetuningScores";
    scores.className = "subtle-button"; scores.textContent = "Resultados";
    controls.append(scores); header.insertBefore(controls, document.getElementById("homeButton"));

    const status = document.createElement("p");
    status.id = "finetuningStatus"; status.setAttribute("role", "status");
    header.after(status);
    const dialog = document.createElement("dialog");
    dialog.id = "finetuningResults"; dialog.setAttribute("aria-labelledby", "finetuningResultsTitle");
    const shell = document.createElement("div"); shell.className = "finetuning-results-shell";
    const head = document.createElement("div"); head.className = "finetuning-results-head";
    const title = document.createElement("h2"); title.id = "finetuningResultsTitle";
    title.textContent = "Resultados auditados · CER micro";
    const close = document.createElement("button"); close.type = "button";
    close.className = "icon-button"; close.textContent = "×";
    close.setAttribute("aria-label", "Cerrar resultados"); close.addEventListener("click", () => dialog.close());
    head.append(title, close); shell.append(head);
    const note = document.createElement("p");
    note.textContent = "25 páginas de validación y 25 de test. El CER corresponde a la época elegida por validación. Qwen sin entrenamiento usa generación determinista; la prueba con LUISA usa otra configuración.";
    shell.append(note);
    const scroll = document.createElement("div"); scroll.className = "finetuning-results-scroll";
    const table = document.createElement("table"); table.id = "finetuningResultsTable";
    const thead = document.createElement("thead"), tr = document.createElement("tr");
    for (const text of ["Modelo", "Época elegida", "CER validación", "CER test", "Test completo", "Fallidas / reintentadas"]) {
      const th = document.createElement("th"); th.scope = "col"; th.textContent = text; tr.append(th);
    }
    thead.append(tr); table.append(thead);
    const tbody = document.createElement("tbody");
    for (const model of data.models) {
      const row = document.createElement("tr"); row.dataset.model = model.id;
      row.title = model.training_description;
      const test = model.evaluations.test;
      const values = [model.name, model.epoch || "—", pct(model.evaluations.validation.cer),
        pct(test.cer) + (test.provisional ? " · provisional" : ""), `${test.completed}/25`,
        `${test.failed_pages} / ${test.retried_pages}`];
      values.forEach(value => { const td = document.createElement("td"); td.textContent = value; row.append(td); });
      if (model.exploratory) row.className = "finetuning-exploratory";
      tbody.append(row);
    }
    table.append(tbody); scroll.append(table); shell.append(scroll);
    const caveat = document.createElement("p");
    caveat.textContent = "La matriz del viewer muestra medias/medianas por página. Este cuadro muestra CER micro. Líneas conserva un test incompleto. LUISA es exploratorio: test contaminado y generación distinta.";
    shell.append(caveat); dialog.append(shell); document.body.append(dialog);
    scores.addEventListener("click", () => dialog.showModal());
    let lastAdjusted = "ajuste-combinado";

    function update() {
      displayNames();
      const isDetail = !document.getElementById("detailView").classList.contains("hidden");
      const route = new URLSearchParams(location.hash.slice(1));
      const id = route.get("model") || state.currentModel;
      const pid = route.get("doc") || route.get("page") || state.currentPage?.id;
      const model = models.get(id);
      if (model?.baseline && models.has(model.baseline)) lastAdjusted = id;
      baseline.hidden = Boolean(model?.baseline_not_displayed);
      baseline.disabled = !isDetail || !model || Boolean(model.baseline_not_displayed);
      baseline.textContent = model?.baseline ? "Ver Qwen sin entrenamiento" : "Volver al ajuste";
      let message = split === "test" ? "Test · " : "Validación · ";
      if (isDetail && model && pid) {
        const row = data.page_status[split][id]?.[pid];
        message += `${model.name} · ${row?.status === "complete" ? "completa" : "INCOMPLETA · CER provisional"} · ${row?.attempts || 0} intento(s)`;
        if (model.exploratory) message += " · Exploratorio; test contaminado y generación distinta";
        if (model.evaluations[split].provisional) message += " · Test de líneas: 24/25 completas";
        status.className = model.exploratory || model.evaluations[split].provisional ? "finetuning-warn" : "";
      } else {
        message += "25 páginas · Líneas: test 24/25, provisional · LUISA: exploratorio, test contaminado";
        status.className = "";
      }
      if (status.textContent !== message) status.textContent = message;
      const summary = document.getElementById("datasetSummary");
      const text = `Qwen · ${split === "test" ? "Test" : "Validación"} · 25 documentos · ${models.size} modelos`;
      if (summary.textContent !== text) summary.textContent = text;
    }
    select.addEventListener("change", () => {
      const root = new URL(data.publication_root, location.href);
      const target = new URL(select.value === "test" ? "test/" : "./", root);
      const isDetail = !document.getElementById("detailView").classList.contains("hidden");
      if (isDetail) target.hash = new URLSearchParams({doc: data.pages[select.value][0], model: state.currentModel}).toString();
      else target.hash = "home";
      location.assign(target.href);
    });
    baseline.addEventListener("click", () => {
      const model = models.get(state.currentModel);
      const next = model?.baseline || lastAdjusted;
      const control = document.getElementById("modelSelect");
      control.value = next; control.dispatchEvent(new Event("change", {bubbles: true}));
    });
    const observer = new MutationObserver(update);
    for (const id of ["pageTitle", "ocrTitle", "datasetSummary", "method-details-title"]) {
      observer.observe(document.getElementById(id), {childList: true, characterData: true, subtree: true});
    }
    for (const id of ["homeView", "detailView"]) {
      observer.observe(document.getElementById(id), {attributes: true, attributeFilter: ["class"]});
    }
    window.addEventListener("hashchange", update); window.addEventListener("popstate", update);
    const resize = new ResizeObserver(() => document.documentElement.style.setProperty("--finetuning-top-height", `${header.getBoundingClientRect().height + status.getBoundingClientRect().height}px`));
    resize.observe(header); resize.observe(status); update();
  }).catch(error => { console.error(error); });
})();
