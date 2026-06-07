from __future__ import annotations


RUN_REVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepResearch Run Review</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #f7f8fa; color: #1f2933; }
    header { height: 54px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; border-bottom: 1px solid #d9dee7; background: #fff; }
    h1 { margin: 0; font-size: 16px; font-weight: 650; }
    main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 55px); }
    aside { border-right: 1px solid #d9dee7; background: #fff; padding: 14px; overflow: auto; }
    section { padding: 18px; overflow: auto; }
    label { display: block; font-size: 12px; font-weight: 650; margin: 12px 0 6px; color: #52616f; }
    textarea, input, select { box-sizing: border-box; width: 100%; border: 1px solid #cbd3df; border-radius: 6px; padding: 9px; font: inherit; background: #fff; }
    textarea { min-height: 90px; resize: vertical; }
    button { border: 1px solid #aeb8c5; border-radius: 6px; background: #fff; padding: 8px 11px; cursor: pointer; font-weight: 600; }
    button.primary { background: #1f6feb; color: #fff; border-color: #1f6feb; }
    button.danger { color: #b42318; border-color: #f0b8b2; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .run { border: 1px solid #d9dee7; border-radius: 6px; padding: 10px; margin: 8px 0; cursor: pointer; background: #fbfcfe; }
    .run.active { border-color: #1f6feb; background: #eef5ff; }
    .muted { color: #687789; font-size: 12px; }
    .status { font-size: 12px; padding: 3px 7px; border-radius: 999px; background: #eef1f5; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .panel { border: 1px solid #d9dee7; border-radius: 8px; background: #fff; padding: 14px; margin-bottom: 14px; }
    .panel h2 { margin: 0 0 10px; font-size: 14px; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f6f8; border-radius: 6px; padding: 10px; max-height: 320px; overflow: auto; }
    .source, .claim { border-top: 1px solid #e3e7ee; padding: 9px 0; }
    @media (max-width: 820px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #d9dee7; } .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>DeepResearch Run Review</h1>
    <div class="row"><button id="refresh">Refresh</button><span id="connection" class="muted"></span></div>
  </header>
  <main>
    <aside>
      <label for="query">Query</label>
      <textarea id="query">How should agent tools fail safely?</textarea>
      <div class="row">
        <label><input id="requireApproval" type="checkbox" checked style="width:auto"> Require approval</label>
      </div>
      <div class="row">
        <button id="create" class="primary">Create Run</button>
        <button id="clear">Clear</button>
      </div>
      <label>Recent Runs</label>
      <div id="runs"></div>
    </aside>
    <section>
      <div class="panel">
        <div class="row">
          <h2 style="flex:1">Run</h2>
          <span id="runStatus" class="status">none</span>
        </div>
        <div id="runMeta" class="muted">No run selected.</div>
      </div>
      <div class="grid">
        <div class="panel">
          <h2>Plan</h2>
          <textarea id="planEditor"></textarea>
          <div class="row">
            <button id="approve" class="primary" disabled>Approve</button>
            <button id="edit" disabled>Save Edit</button>
            <button id="reject" class="danger" disabled>Reject</button>
            <button id="cancel" class="danger" disabled>Cancel</button>
          </div>
        </div>
        <div class="panel">
          <h2>Events</h2>
          <pre id="events"></pre>
        </div>
      </div>
      <div class="panel">
        <h2>Report</h2>
        <textarea id="reportEditor"></textarea>
      </div>
      <div class="grid">
        <div class="panel">
          <h2>Sources</h2>
          <div id="sources"></div>
        </div>
        <div class="panel">
          <h2>Citations</h2>
          <div id="citations"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    let selectedRunId = null;
    let eventSource = null;
    const $ = id => document.getElementById(id);

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function refreshRuns() {
      const runs = await api("/runs");
      $("runs").innerHTML = runs.map(run => `
        <div class="run ${run.run_id === selectedRunId ? "active" : ""}" data-run="${run.run_id}">
          <div><strong>${run.run_id}</strong> <span class="status">${run.status}</span></div>
          <div class="muted">${escapeHtml(run.query).slice(0, 120)}</div>
        </div>`).join("");
      document.querySelectorAll(".run").forEach(item => item.onclick = () => selectRun(item.dataset.run));
    }

    async function selectRun(runId) {
      selectedRunId = runId;
      const run = await api(`/runs/${runId}`);
      renderRun(run);
      subscribe(runId);
      await refreshRuns();
    }

    function renderRun(run) {
      $("runStatus").textContent = run.status;
      $("runMeta").textContent = `${run.run_id} | ${run.current_stage} | tokens ${run.total_tokens} | cost ${run.total_cost}`;
      const subquestions = run.plan_json?.subquestions || [];
      $("planEditor").value = JSON.stringify(subquestions, null, 2);
      $("approve").disabled = run.status !== "waiting_approval";
      $("edit").disabled = run.status !== "waiting_approval";
      $("reject").disabled = run.status !== "waiting_approval";
      $("cancel").disabled = ["succeeded", "failed", "cancelled"].includes(run.status);
      const result = run.result_json;
      $("reportEditor").value = result?.answer || "";
      renderSources(result?.sources || []);
      renderCitations(result?.citation_check?.assessments || []);
    }

    function subscribe(runId) {
      if (eventSource) eventSource.close();
      $("events").textContent = "";
      eventSource = new EventSource(`/runs/${runId}/events`);
      eventSource.onopen = () => $("connection").textContent = "events connected";
      eventSource.onerror = () => $("connection").textContent = "events closed";
      eventSource.onmessage = event => appendEvent(event.data);
      ["planner.planner_done", "approval.waiting_approval", "researcher.succeeded", "synthesizer.succeeded", "verifier.succeeded", "run.succeeded"].forEach(name => {
        eventSource.addEventListener(name, event => {
          appendEvent(event.data);
          api(`/runs/${runId}`).then(renderRun).then(refreshRuns);
        });
      });
    }

    function appendEvent(data) {
      $("events").textContent += data + "\\n";
      $("events").scrollTop = $("events").scrollHeight;
    }

    function renderSources(sources) {
      $("sources").innerHTML = sources.map(source => `
        <div class="source"><strong>${source.id} ${escapeHtml(source.title)}</strong>
        <div class="muted">${escapeHtml(source.provider)} | ${escapeHtml(source.url)}</div>
        <div>${escapeHtml(source.content || "").slice(0, 500)}</div></div>`).join("");
    }

    function renderCitations(items) {
      $("citations").innerHTML = items.map(item => `
        <div class="claim"><strong>${item.support_level || (item.supported ? "supported" : "unsupported")}</strong>
        <div>${escapeHtml(item.claim)}</div>
        <div class="muted">overlap ${item.overlap_score} | ${escapeHtml(item.reason)}</div>
        <pre>${escapeHtml(JSON.stringify(item.evidence_quotes || [], null, 2))}</pre></div>`).join("");
    }

    $("create").onclick = async () => {
      const run = await api("/runs", {
        method: "POST",
        body: JSON.stringify({
          query: $("query").value,
          search_provider: "mock",
          llm_provider: "mock",
          max_researchers: 1,
          max_results_per_researcher: 1,
          require_approval: $("requireApproval").checked
        })
      });
      await selectRun(run.run_id);
    };
    $("approve").onclick = async () => selectRun((await api(`/runs/${selectedRunId}/approve`, { method: "POST" })).run_id);
    $("edit").onclick = async () => {
      const subquestions = JSON.parse($("planEditor").value || "[]");
      await api(`/runs/${selectedRunId}/edit`, { method: "POST", body: JSON.stringify({ subquestions }) });
      await selectRun(selectedRunId);
    };
    $("reject").onclick = async () => selectRun((await api(`/runs/${selectedRunId}/reject`, { method: "POST", body: JSON.stringify({ reason: "rejected in UI" }) })).run_id);
    $("cancel").onclick = async () => selectRun((await api(`/runs/${selectedRunId}/cancel`, { method: "POST" })).run_id);
    $("refresh").onclick = refreshRuns;
    $("clear").onclick = () => { $("query").value = ""; };

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
    }

    refreshRuns();
  </script>
</body>
</html>
"""
