/**
 * Basketry — Frontend Application Script
 * Live force-directed canvas graph, receipt generator, algorithm race,
 * custom canvas charts, parameter tuning sweep, and manual metric derivations.
 */

(function () {
  "use strict";

  // Helper selector
  const $ = (id) => document.getElementById(id);

  // App State
  let state = {
    algorithm: "scratch",
    minSupport: 0.03,
    minConfidence: 0.35,
    minLift: 1.2,
    minRating: 4.0,
    maxLen: 3,
    datasetSource: "demo_small.csv",
    activeRuleIndex: 0,
    rules: [],
    itemsets: [],
    runtimeMs: 0,
    summary: null,
  };

  // Graph Simulation State
  let nodes = [];
  let edges = [];
  let selectedNodeId = null;
  let draggedNode = null;
  let canvasWidth = 0;
  let canvasHeight = 0;
  let dpr = window.devicePixelRatio || 1;
  let simulationTicks = 0;
  let rippleTime = 0;
  let animFrameId = null;
  const isReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Counter tweens
  const counterValues = {};

  // Setup Canvas
  const cv = $("cv");
  const cx = cv.getContext("2d");

  // Gradient Color interpolation: Periwinkle (#8b9cff) to Apricot (#ffb86b)
  function getLiftColor(t, alpha = 1.0) {
    const a = [139, 156, 255];
    const b = [255, 184, 107];
    const clamped = Math.max(0, Math.min(1, t));
    const r = Math.round(a[0] + (b[0] - a[0]) * clamped);
    const g = Math.round(a[1] + (b[1] - a[1]) * clamped);
    const bl = Math.round(a[2] + (b[2] - a[2]) * clamped);
    return `rgba(${r}, ${g}, ${bl}, ${alpha})`;
  }

  // Number tween helper
  function tweenValue(id, targetVal, decimals = 0, suffix = "") {
    const el = $(id);
    if (!el) return;
    const fromVal = counterValues[id] || 0;
    const duration = isReducedMotion ? 50 : 1200;
    const startTime = performance.now();
    counterValues[id] = targetVal;

    cancelAnimationFrame(el._tweenFrame);

    function step(now) {
      const progress = Math.min(1, (now - startTime) / duration);
      // Ease out quartic
      const ease = 1 - Math.pow(1 - progress, 4);
      const current = fromVal + (targetVal - fromVal) * ease;
      el.textContent = current.toFixed(decimals) + suffix;

      if (progress < 1) {
        el._tweenFrame = requestAnimationFrame(step);
      } else {
        el.textContent = targetVal.toFixed(decimals) + suffix;
      }
    }
    el._tweenFrame = requestAnimationFrame(step);
  }

  // =========================================================================
  // Graph Physics & Rendering
  // =========================================================================

  function resizeCanvas() {
    const rect = cv.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    canvasWidth = rect.width;
    canvasHeight = rect.height;
    cv.width = canvasWidth * dpr;
    cv.height = canvasHeight * dpr;
    simulationTicks = 160;
  }

  function rebuildGraph() {
    const topRules = state.rules.slice(0, 80);
    const edgeMap = new Map();
    const productNodes = new Map();

    // Collect involved items
    topRules.forEach((rule) => {
      rule.antecedent.forEach((a) => {
        rule.consequent.forEach((b) => {
          if (!productNodes.has(a)) {
            const title = rule.antecedent_titles[rule.antecedent.indexOf(a)] || a;
            productNodes.set(a, { id: a, title, support: rule.support });
          }
          if (!productNodes.has(b)) {
            const title = rule.consequent_titles[rule.consequent.indexOf(b)] || b;
            productNodes.set(b, { id: b, title, support: rule.support });
          }

          const edgeKey = a < b ? `${a}_${b}` : `${b}_${a}`;
          if (!edgeMap.has(edgeKey) || edgeMap.get(edgeKey).lift < rule.lift) {
            edgeMap.set(edgeKey, {
              sourceId: a,
              targetId: b,
              lift: rule.lift,
              confidence: rule.confidence,
            });
          }
        });
      });
    });

    // Reuse existing node coordinates for smooth morphing
    const oldNodes = new Map(nodes.map((n) => [n.id, n]));
    nodes = Array.from(productNodes.values()).map((p) => {
      const existing = oldNodes.get(p.id);
      if (existing) {
        existing.support = p.support;
        existing.title = p.title;
        return existing;
      }
      return {
        id: p.id,
        title: p.title,
        support: p.support,
        x: canvasWidth / 2 + (Math.random() - 0.5) * 50,
        y: canvasHeight / 2 + (Math.random() - 0.5) * 50,
        vx: 0,
        vy: 0,
        bloom: isReducedMotion ? 1 : 0,
        radius: 12,
      };
    });

    const nodeIndex = new Map(nodes.map((n, i) => [n.id, i]));
    const maxLift = Math.max(...Array.from(edgeMap.values()).map((e) => e.lift), 1.05);

    edges = Array.from(edgeMap.values())
      .filter((e) => nodeIndex.has(e.sourceId) && nodeIndex.has(e.targetId))
      .map((e) => ({
        source: nodeIndex.get(e.sourceId),
        target: nodeIndex.get(e.targetId),
        lift: e.lift,
        weight: (e.lift - 1.0) / (maxLift - 1.0 || 1.0),
      }));

    simulationTicks = 240;
  }

  function simulatePhysicsStep() {
    const k = nodes.length;
    if (k === 0) return;

    // Repulsion between nodes
    for (let i = 0; i < k; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < k; j++) {
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dSq = dx * dx + dy * dy + 0.1;
        let force = Math.min(2800 / dSq, 6);
        let dist = Math.sqrt(dSq);
        dx /= dist;
        dy /= dist;
        a.vx += dx * force;
        a.vy += dy * force;
        b.vx -= dx * force;
        b.vy -= dy * force;
      }
      // Gentle gravity towards canvas center
      a.vx += (canvasWidth / 2 - a.x) * 0.0035;
      a.vy += (canvasHeight / 2 - a.y) * 0.0035;
    }

    // Spring attraction along association edges
    edges.forEach((edge) => {
      const a = nodes[edge.source];
      const b = nodes[edge.target];
      if (!a || !b) return;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      let dist = Math.hypot(dx, dy) || 1;
      let restDist = 125 - edge.weight * 55;
      let force = (dist - restDist) * 0.012;
      dx /= dist;
      dy /= dist;
      a.vx += dx * force;
      a.vy += dy * force;
      b.vx -= dx * force;
      b.vy -= dy * force;
    });

    // Apply velocities and bounds
    nodes.forEach((n) => {
      if (n === draggedNode) return;
      n.vx *= 0.82;
      n.vy *= 0.82;
      n.x += n.vx;
      n.y += n.vy;
      n.x = Math.max(35, Math.min(canvasWidth - 35, n.x));
      n.y = Math.max(35, Math.min(canvasHeight - 35, n.y));
    });
  }

  function drawGraph() {
    const now = performance.now();

    if (simulationTicks > 0 || draggedNode) {
      simulatePhysicsStep();
      simulationTicks--;
    }

    // Subtle ambient floating breathing
    if (!isReducedMotion) {
      nodes.forEach((n) => {
        if (n !== draggedNode) {
          n.x += Math.sin(now / 1600 + n.x * 0.01) * 0.08;
          n.y += Math.cos(now / 2000 + n.y * 0.01) * 0.08;
        }
      });
    }

    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Active connected neighbor set
    const neighborSet = new Set();
    if (selectedNodeId) {
      edges.forEach((e) => {
        const src = nodes[e.source];
        const tgt = nodes[e.target];
        if (src.id === selectedNodeId) neighborSet.add(e.target);
        if (tgt.id === selectedNodeId) neighborSet.add(e.source);
      });
    }

    // 1. Draw Edges
    edges.forEach((edge) => {
      const a = nodes[edge.source];
      const b = nodes[edge.target];
      if (!a || !b) return;

      const isConnected = !selectedNodeId || a.id === selectedNodeId || b.id === selectedNodeId;
      const grad = cx.createLinearGradient(a.x, a.y, b.x, b.y);
      grad.addColorStop(0, getLiftColor(edge.weight, isConnected ? 0.85 : 0.06));
      grad.addColorStop(1, getLiftColor(Math.min(1, edge.weight + 0.2), isConnected ? 0.85 : 0.06));

      cx.strokeStyle = grad;
      cx.lineWidth = isConnected ? 1.5 + edge.weight * 4.5 : 0.8;
      cx.beginPath();
      cx.moveTo(a.x, a.y);
      cx.lineTo(b.x, b.y);
      cx.stroke();

      // Traveling glowing pulse on strong edges
      if (!isReducedMotion && isConnected && edge.weight > 0.2) {
        const pulseCycle = ((now / (3000 - edge.weight * 1600)) + edge.source * 0.31) % 1;
        const px = a.x + (b.x - a.x) * pulseCycle;
        const py = a.y + (b.y - a.y) * pulseCycle;

        cx.fillStyle = getLiftColor(edge.weight, 0.95);
        cx.shadowColor = getLiftColor(edge.weight, 0.8);
        cx.shadowBlur = 10;
        cx.beginPath();
        cx.arc(px, py, 2.5 + edge.weight * 2, 0, Math.PI * 2);
        cx.fill();
        cx.shadowBlur = 0;
      }
    });

    // 2. Draw Nodes
    const isDark = document.documentElement.dataset.theme !== "light";
    const labelColor = isDark ? "#f2eefc" : "#1a1538";
    cx.font = "500 12px 'DM Sans', system-ui, sans-serif";
    cx.textAlign = "center";

    nodes.forEach((n, idx) => {
      // Bloom animation
      n.bloom = Math.min(1, n.bloom + 0.024);
      const isSelected = n.id === selectedNodeId;
      const isHighlighted = !selectedNodeId || isSelected || neighborSet.has(idx);

      const baseRadius = 8 + Math.sqrt(n.support || 0.02) * 26;
      const breathing = isReducedMotion ? 0 : 0.04 * Math.sin(now / 950 + idx);
      const r = baseRadius * n.bloom * (1 + breathing);
      n.radius = r;

      // Glow halo
      const maxEdgeWeight = edges
        .filter((e) => e.source === idx || e.target === idx)
        .reduce((acc, e) => Math.max(acc, e.weight), 0.1);

      const haloGrad = cx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 2.8);
      haloGrad.addColorStop(0, getLiftColor(maxEdgeWeight, isHighlighted ? 0.35 : 0.05));
      haloGrad.addColorStop(1, "transparent");

      cx.fillStyle = haloGrad;
      cx.beginPath();
      cx.arc(n.x, n.y, r * 2.8, 0, Math.PI * 2);
      cx.fill();

      // Node Body
      cx.fillStyle = getLiftColor(maxEdgeWeight, isHighlighted ? 1.0 : 0.2);
      cx.beginPath();
      cx.arc(n.x, n.y, r, 0, Math.PI * 2);
      cx.fill();

      // Ripple Ring when selected
      if (isSelected) {
        cx.strokeStyle = labelColor;
        cx.lineWidth = 2.2;
        cx.beginPath();
        cx.arc(n.x, n.y, r + 5, 0, Math.PI * 2);
        cx.stroke();

        const rippleProgress = Math.min(1, (now - rippleTime) / 1100);
        if (rippleProgress < 1 && !isReducedMotion) {
          cx.strokeStyle = getLiftColor(maxEdgeWeight, (1 - rippleProgress) * 0.7);
          cx.lineWidth = 1.5;
          cx.beginPath();
          cx.arc(n.x, n.y, r + 5 + rippleProgress * 65, 0, Math.PI * 2);
          cx.stroke();
        }
      }

      // Title Label
      if (isHighlighted) {
        cx.fillStyle = labelColor;
        const shortTitle = n.title.length > 22 ? n.title.substring(0, 20) + "…" : n.title;
        cx.fillText(shortTitle, n.x, n.y + r + 15);
      }
    });

    animFrameId = requestAnimationFrame(drawGraph);
  }

  // Pointer interaction
  function getCanvasPos(e) {
    const rect = cv.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  function findNodeHit(x, y) {
    return nodes.find((n) => Math.hypot(n.x - x, n.y - y) <= (n.radius || 12) + 8);
  }

  cv.addEventListener("pointerdown", (e) => {
    const [x, y] = getCanvasPos(e);
    const hit = findNodeHit(x, y);
    if (hit) {
      draggedNode = hit;
      selectedNodeId = hit.id;
      rippleTime = performance.now();
      cv.setPointerCapture(e.pointerId);
      cv.style.cursor = "grabbing";
      renderReceipt(hit.id);
    } else {
      selectedNodeId = null;
      renderReceipt(null);
    }
  });

  cv.addEventListener("pointermove", (e) => {
    const [x, y] = getCanvasPos(e);
    if (draggedNode) {
      draggedNode.x = x;
      draggedNode.y = y;
      draggedNode.vx = 0;
      draggedNode.vy = 0;
    } else {
      cv.style.cursor = findNodeHit(x, y) ? "pointer" : "grab";
    }
  });

  cv.addEventListener("pointerup", () => {
    draggedNode = null;
    simulationTicks = 120;
    cv.style.cursor = "grab";
  });

  // Spotlight mouse effect on canvas parent container
  cv.parentElement.addEventListener("pointermove", (e) => {
    const box = cv.parentElement.getBoundingClientRect();
    cv.parentElement.style.setProperty("--mx", `${e.clientX - box.left}px`);
    cv.parentElement.style.setProperty("--my", `${e.clientY - box.top}px`);
  });

  // =========================================================================
  // Receipt Printing
  // =========================================================================

  function renderReceipt(productId) {
    const receiptEl = $("receipt-card");
    if (!productId) {
      receiptEl.classList.add("empty");
      return;
    }

    const matchingRules = state.rules.filter(
      (r) => r.antecedent.includes(productId) || r.consequent.includes(productId)
    );

    const targetNode = nodes.find((n) => n.id === productId);
    const productTitle = targetNode ? targetNode.title : productId;

    if (matchingRules.length === 0) {
      receiptEl.innerHTML = `
        <h3>${escapeHtml(productTitle)}</h3>
        <div class="sub">goes well with</div>
        <div class="rl" style="--i:0">
          <span class="nm">No rules meet current thresholds.</span>
          <div class="m"><span>Lower min support or confidence</span></div>
        </div>
      `;
      receiptEl.classList.remove("empty");
      return;
    }

    const displayedRules = matchingRules.slice(0, 5);
    const ruleLinesHtml = displayedRules
      .map((r, idx) => {
        const antStr = r.antecedent_titles.join(" + ");
        const conStr = r.consequent_titles.join(" + ");
        const suppPct = (r.support * 100).toFixed(1);
        const confPct = (r.confidence * 100).toFixed(1);
        const liftStr = r.lift.toFixed(1);

        return `
          <div class="rl" style="--i:${idx}">
            <span class="nm">${escapeHtml(antStr)} → ${escapeHtml(conStr)}</span>
            <div class="m">
              <span>seen in ${suppPct}%</span>
              <span>${liftStr}× lift</span>
            </div>
            <div class="m">
              <span>${confPct}% confidence</span>
            </div>
            <div class="bar">
              <u style="width: ${confPct}%"></u>
            </div>
          </div>
        `;
      })
      .join("");

    receiptEl.innerHTML = `
      <h3>${escapeHtml(productTitle)}</h3>
      <div class="sub">goes well with</div>
      ${ruleLinesHtml}
      <div class="sub" style="margin-top: 14px; border: 0;">
        ${matchingRules.length} bundle rule${matchingRules.length === 1 ? "" : "s"} found
      </div>
    `;
    receiptEl.classList.remove("empty");
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // =========================================================================
  // API Calls & Data Mining
  // =========================================================================

  async function fetchSummary() {
    try {
      const res = await fetch("/api/summary");
      if (!res.ok) return;
      const data = await res.json();
      state.summary = data;
      renderSummaryDetails(data);
    } catch (err) {
      console.warn("Failed fetching summary:", err);
    }
  }

  function renderSummaryDetails(data) {
    const s = data.dataset_summary || {};
    if ($("sum-rows")) $("sum-rows").textContent = s.rows ? s.rows.toLocaleString() : "—";
    if ($("sum-reviewers")) $("sum-reviewers").textContent = s.reviewers ? s.reviewers.toLocaleString() : "—";
    if ($("sum-products")) $("sum-products").textContent = s.products ? s.products.toLocaleString() : "—";
    if ($("sum-nulls")) $("sum-nulls").textContent = s.null_count !== undefined ? s.null_count.toLocaleString() : "0";
    if ($("sum-dups")) $("sum-dups").textContent = s.duplicate_count !== undefined ? s.duplicate_count.toLocaleString() : "0";
    if ($("upload-status")) $("upload-status").textContent = `Active dataset: ${data.source || state.datasetSource}`;
  }

  let mineDebounceTimer = null;
  function triggerMine(immediate = false) {
    clearTimeout(mineDebounceTimer);
    mineDebounceTimer = setTimeout(runMineRequest, immediate ? 0 : 80);
  }

  async function runMineRequest() {
    const payload = {
      algorithm: state.algorithm,
      min_support: state.minSupport,
      min_confidence: state.minConfidence,
      min_lift: state.minLift,
      min_rating: state.minRating,
      max_len: state.maxLen,
    };

    try {
      const res = await fetch("/api/mine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        console.error("Mining error:", err);
        return;
      }

      const result = await res.json();
      state.rules = result.rules || [];
      state.itemsets = result.itemsets || [];
      state.runtimeMs = result.runtime_ms || 0;

      // Update Top stats
      const numReviewers = state.summary?.preprocessed_meta?.qualifying_baskets || 1809;
      const numProds = state.summary?.preprocessed_meta?.unique_products_in_baskets || 40;
      const strongestLift = state.rules.length ? state.rules[0].lift : 0;

      tweenValue("stat-reviewers", numReviewers);
      tweenValue("stat-products", numProds);
      tweenValue("stat-rules", state.rules.length);
      tweenValue("stat-lift", strongestLift, 1, "×");
      tweenValue("stat-runtime", state.runtimeMs, 0, " ms");

      // Rebuild graph & lists
      rebuildGraph();
      renderTopBundlesList();
      renderCustomCharts();

      // Update badge
      const algoLabels = {
        scratch: "Scratch Apriori",
        apriori: "mlxtend Apriori",
        fpgrowth: "mlxtend FP-Growth",
      };
      $("active-algo-badge").textContent = `Algorithm: ${algoLabels[state.algorithm] || state.algorithm}`;
    } catch (err) {
      console.error("Mining request failed:", err);
    }
  }

  // =========================================================================
  // Top Bundles List & Product Search
  // =========================================================================

  function renderTopBundlesList() {
    const listEl = $("top-bundles-list");
    const query = ($("product-search")?.value || "").toLowerCase().trim();

    let filtered = state.rules;
    if (query) {
      filtered = filtered.filter((r) => {
        const full = (r.antecedent_titles.join(" ") + " " + r.consequent_titles.join(" ")).toLowerCase();
        return full.includes(query);
      });
    }

    if (filtered.length === 0) {
      listEl.innerHTML = `<li><span>No rules found. Lower support/confidence.</span></li>`;
      return;
    }

    const itemsHtml = filtered
      .slice(0, 8)
      .map((r) => {
        const antStr = r.antecedent_titles.join(" + ");
        const conStr = r.consequent_titles.join(" + ");
        const primaryId = r.antecedent[0];

        return `
          <li data-id="${primaryId}" tabindex="0">
            <span>${escapeHtml(antStr)} → ${escapeHtml(conStr)}</span>
            <em>${r.lift.toFixed(1)}×</em>
          </li>
        `;
      })
      .join("");

    listEl.innerHTML = itemsHtml;
  }

  $("top-bundles-list")?.addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (!li || !li.dataset.id) return;
    const prodId = li.dataset.id;
    selectedNodeId = prodId;
    rippleTime = performance.now();
    simulationTicks = 160;
    renderReceipt(prodId);
    cv.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  $("product-search")?.addEventListener("input", () => {
    renderTopBundlesList();
  });

  // =========================================================================
  // Custom Hand-Drawn Canvas Charts
  // =========================================================================

  function renderCustomCharts() {
    renderScatterChart();
    renderBarsChart();
  }

  function renderScatterChart() {
    const cScatter = $("chart-scatter");
    if (!cScatter) return;
    const ctx = cScatter.getContext("2d");
    const w = cScatter.width;
    const h = cScatter.height;
    ctx.clearRect(0, 0, w, h);

    const padLeft = 45;
    const padBottom = 35;
    const padTop = 15;
    const padRight = 20;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const isDark = document.documentElement.dataset.theme !== "light";
    const axisColor = isDark ? "rgba(255, 255, 255, 0.18)" : "rgba(26, 21, 56, 0.18)";
    const textColor = isDark ? "#9a93c4" : "#5d5790";

    // Draw Axes
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Axis Labels
    ctx.fillStyle = textColor;
    ctx.font = "11px 'DM Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Support (%)", padLeft + plotW / 2, h - 8);

    ctx.save();
    ctx.translate(14, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Confidence (%)", 0, 0);
    ctx.restore();

    if (!state.rules || state.rules.length === 0) {
      ctx.textAlign = "center";
      ctx.fillText("No association rules mined yet", padLeft + plotW / 2, padTop + plotH / 2);
      return;
    }

    const maxSupp = Math.max(...state.rules.map((r) => r.support), 0.08);
    const maxLift = Math.max(...state.rules.map((r) => r.lift), 2.0);

    // Plot Points
    state.rules.forEach((r) => {
      const px = padLeft + (r.support / maxSupp) * plotW;
      const py = padTop + plotH - (r.confidence / 1.0) * plotH;
      const normLift = (r.lift - 1.0) / (maxLift - 1.0 || 1.0);

      ctx.fillStyle = getLiftColor(normLift, 0.85);
      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function renderBarsChart() {
    const cBars = $("chart-bars");
    if (!cBars) return;
    const ctx = cBars.getContext("2d");
    const w = cBars.width;
    const h = cBars.height;
    ctx.clearRect(0, 0, w, h);

    const padLeft = 140;
    const padRight = 45;
    const padTop = 15;
    const padBottom = 25;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const isDark = document.documentElement.dataset.theme !== "light";
    const textColor = isDark ? "#f2eefc" : "#1a1538";
    const muteColor = isDark ? "#9a93c4" : "#5d5790";

    const topItemsets = (state.itemsets || []).slice(0, 6);
    if (topItemsets.length === 0) {
      ctx.fillStyle = muteColor;
      ctx.textAlign = "center";
      ctx.font = "12px 'DM Sans', sans-serif";
      ctx.fillText("No itemsets available", w / 2, h / 2);
      return;
    }

    const maxSupp = Math.max(...topItemsets.map((x) => x.support), 0.05);
    const barHeight = Math.min(22, (plotH / topItemsets.length) * 0.65);
    const stepY = plotH / topItemsets.length;

    topItemsets.forEach((it, idx) => {
      const y = padTop + idx * stepY + (stepY - barHeight) / 2;
      const barW = (it.support / maxSupp) * plotW;

      // Label
      const label = it.titles.join(" + ");
      const shortLabel = label.length > 18 ? label.substring(0, 16) + "…" : label;
      ctx.fillStyle = textColor;
      ctx.font = "11.5px 'DM Sans', sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(shortLabel, padLeft - 10, y + barHeight * 0.72);

      // Bar fill with gradient
      const grad = ctx.createLinearGradient(padLeft, 0, padLeft + barW, 0);
      grad.addColorStop(0, "rgba(139, 156, 255, 0.7)");
      grad.addColorStop(1, "rgba(255, 184, 107, 0.95)");

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(padLeft, y, Math.max(4, barW), barHeight, 4);
      ctx.fill();

      // Percentage value text
      ctx.fillStyle = muteColor;
      ctx.textAlign = "left";
      ctx.fillText(`${(it.support * 100).toFixed(1)}%`, padLeft + barW + 8, y + barHeight * 0.72);
    });
  }

  // =========================================================================
  // Algorithm Race & Comparison
  // =========================================================================

  $("btn-run-race")?.addEventListener("click", runAlgorithmRace);

  async function runAlgorithmRace() {
    const btn = $("btn-run-race");
    btn.disabled = true;
    btn.textContent = "Racing algorithms...";

    const url = `/api/compare?min_support=${state.minSupport}&min_confidence=${state.minConfidence}&min_lift=${state.minLift}&min_rating=${state.minRating}&max_len=${state.maxLen}`;

    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error("Comparison request failed");
      const data = await res.json();

      // Reset bars
      $("bar-scratch").style.width = "0%";
      $("bar-apriori").style.width = "0%";
      $("bar-fpgrowth").style.width = "0%";

      const tScratch = data.scratch.runtime_ms;
      const tApriori = data.apriori.runtime_ms;
      const tFp = data.fpgrowth.runtime_ms;
      const maxT = Math.max(tScratch, tApriori, tFp, 1.0);

      // Update text
      $("time-scratch").textContent = `${tScratch.toFixed(1)} ms`;
      $("time-apriori").textContent = `${tApriori.toFixed(1)} ms`;
      $("time-fpgrowth").textContent = `${tFp.toFixed(1)} ms`;

      $("items-scratch").textContent = `Itemsets: ${data.scratch.num_itemsets}`;
      $("rules-scratch").textContent = `Rules: ${data.scratch.num_rules}`;

      $("items-apriori").textContent = `Itemsets: ${data.apriori.num_itemsets}`;
      $("rules-apriori").textContent = `Rules: ${data.apriori.num_rules}`;

      $("items-fpgrowth").textContent = `Itemsets: ${data.fpgrowth.num_itemsets}`;
      $("rules-fpgrowth").textContent = `Rules: ${data.fpgrowth.num_rules}`;

      // Animate progress bar widths proportional to runtime
      setTimeout(() => {
        $("bar-scratch").style.width = `${Math.max(12, (tScratch / maxT) * 100)}%`;
        $("bar-apriori").style.width = `${Math.max(12, (tApriori / maxT) * 100)}%`;
        $("bar-fpgrowth").style.width = `${Math.max(12, (tFp / maxT) * 100)}%`;
      }, 100);

      // Verification badge
      const verif = data.verification || {};
      const tag = $("verif-tag");
      if (verif.itemsets_match && verif.rules_match) {
        tag.className = "verif-tag success";
        tag.textContent = "Equivalence Verified: Scratch == mlxtend (100% Match)";
      } else {
        tag.className = "verif-tag";
        tag.textContent = `Equivalence: Scratch found ${verif.scratch_rule_count} rules, mlxtend found ${verif.mlx_rule_count}`;
      }
    } catch (err) {
      console.error("Race error:", err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run benchmark race";
    }
  }

  // =========================================================================
  // Parameter Tuning Grid (Sweep Heatmap)
  // =========================================================================

  async function fetchSweepGrid() {
    try {
      const res = await fetch(`/api/sweep?min_rating=${state.minRating}`);
      if (!res.ok) return;
      const data = await res.json();
      renderSweepTable(data);
    } catch (err) {
      console.warn("Sweep error:", err);
    }
  }

  function renderSweepTable(data) {
    const table = $("sweep-table");
    if (!table) return;

    const suppLabels = data.supports || [];
    const confLabels = data.confidences || [];
    const grid = data.grid || [];

    // Find max cell value for color scaling
    let maxVal = 1;
    grid.forEach((row) => row.forEach((v) => { if (v > maxVal) maxVal = v; }));

    let html = `<thead><tr><th>Support \\ Conf</th>`;
    confLabels.forEach((c) => { html += `<th>${c}</th>`; });
    html += `</tr></thead><tbody>`;

    suppLabels.forEach((sLabel, rowIdx) => {
      html += `<tr><th>${sLabel}</th>`;
      confLabels.forEach((_, colIdx) => {
        const val = grid[rowIdx]?.[colIdx] || 0;
        const ratio = val / maxVal;
        const bg = `rgba(255, 184, 107, ${Math.max(0.08, ratio * 0.75)})`;
        html += `<td style="background: ${bg}" data-s="${data.supports_raw[rowIdx]}" data-c="${data.confidences_raw[colIdx]}">${val} rules</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody>`;

    table.innerHTML = html;

    // Click cell to apply parameters
    table.querySelectorAll("td").forEach((cell) => {
      cell.addEventListener("click", () => {
        const s = parseFloat(cell.dataset.s);
        const c = parseFloat(cell.dataset.c);
        if (s && c) {
          $("slider-sup").value = (s * 100).toFixed(1);
          $("slider-conf").value = Math.round(c * 100);
          updateSliderLabels();
          triggerMine(true);
        }
      });
    });
  }

  // =========================================================================
  // Manual Worked Example Derivation
  // =========================================================================

  async function fetchManualExample(idx = 0) {
    try {
      const res = await fetch(`/api/manual-example?rule_index=${idx}`);
      if (!res.ok) return;
      const data = await res.json();
      renderManualExample(data);
    } catch (err) {
      console.warn("Manual example error:", err);
    }
  }

  function renderManualExample(data) {
    if ($("manual-rule-display")) {
      $("manual-rule-display").textContent = data.rule_display || "Rule Calculation";
    }

    const grid = $("manual-steps-grid");
    if (!grid) return;

    const stepsHtml = (data.steps || [])
      .map((s) => `
        <div class="manual-step-card">
          <h4>${escapeHtml(s.metric)}</h4>
          <span class="step-formula">${escapeHtml(s.formula)}</span>
          <span class="step-calc">${escapeHtml(s.calculation)}</span>
          <span class="step-desc">${escapeHtml(s.interpretation)}</span>
        </div>
      `)
      .join("");

    grid.innerHTML = stepsHtml;
  }

  $("btn-next-rule")?.addEventListener("click", () => {
    state.activeRuleIndex = (state.activeRuleIndex + 1) % Math.max(1, state.rules.length);
    fetchManualExample(state.activeRuleIndex);
  });

  // =========================================================================
  // Upload & File Handling
  // =========================================================================

  const dropZone = $("drop-zone");
  const fileInput = $("file-input");

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    ["dragenter", "dragover"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add("on");
      });
    });

    ["dragleave", "drop"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove("on");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadFile(e.target.files[0]);
      }
    });
  }

  async function uploadFile(file) {
    if (!file.name.endsWith(".csv")) {
      alert("Please upload a .csv file.");
      return;
    }

    const statusEl = $("upload-status");
    if (statusEl) statusEl.textContent = `Uploading and parsing ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      state.datasetSource = data.source;
      renderSummaryDetails(data);
      if (statusEl) statusEl.textContent = `Active dataset: ${data.source} (${data.dataset_summary.rows.toLocaleString()} rows)`;

      triggerMine(true);
      fetchSweepGrid();
      fetchManualExample(0);
    } catch (err) {
      if (statusEl) statusEl.textContent = `Upload error: ${err.message}`;
    }
  }

  $("btn-load-demo")?.addEventListener("click", () => loadSampleDataset("demo"));
  $("btn-load-full")?.addEventListener("click", () => loadSampleDataset("sample"));

  async function loadSampleDataset(name) {
    const statusEl = $("upload-status");
    if (statusEl) statusEl.textContent = `Loading ${name} dataset...`;

    try {
      const res = await fetch(`/api/sample?name=${name}`, { method: "POST" });
      if (!res.ok) throw new Error("Sample load failed");
      const data = await res.json();
      state.datasetSource = data.source;
      renderSummaryDetails(data);
      if (statusEl) statusEl.textContent = `Active dataset: ${data.source}`;
      triggerMine(true);
      fetchSweepGrid();
      fetchManualExample(0);
    } catch (err) {
      if (statusEl) statusEl.textContent = `Load failed: ${err.message}`;
    }
  }

  // =========================================================================
  // Controls & Theme Switcher
  // =========================================================================

  function updateSliderLabels() {
    state.minSupport = parseFloat($("slider-sup").value) / 100;
    state.minConfidence = parseFloat($("slider-conf").value) / 100;
    state.minLift = parseFloat($("slider-lift").value);
    state.minRating = parseFloat($("slider-rating").value);

    $("val-sup").textContent = `${$("slider-sup").value}%`;
    $("val-conf").textContent = `${$("slider-conf").value}%`;
    $("val-lift").textContent = `${state.minLift.toFixed(1)}x`;
    $("val-rating").textContent = `${state.minRating} star${state.minRating > 1 ? "s" : ""} and up`;
  }

  ["slider-sup", "slider-conf", "slider-lift", "slider-rating"].forEach((id) => {
    $(id)?.addEventListener("input", () => {
      updateSliderLabels();
      triggerMine();
    });
  });

  // Segmented Algorithm Selector
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-checked", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-checked", "true");
      state.algorithm = btn.dataset.algo;
      triggerMine(true);
    });
  });

  // Theme Switcher
  const themeBtn = $("theme-btn");
  themeBtn?.addEventListener("click", () => {
    const doc = document.documentElement;
    const isDark = doc.dataset.theme === "dark" || (!doc.dataset.theme && !window.matchMedia("(prefers-color-scheme: light)").matches);
    const newTheme = isDark ? "light" : "dark";
    doc.dataset.theme = newTheme;
    themeBtn.textContent = newTheme === "dark" ? "Switch to light" : "Switch to dark";
    renderCustomCharts();
  });

  // =========================================================================
  // Initialization
  // =========================================================================

  window.addEventListener("resize", () => {
    resizeCanvas();
    renderCustomCharts();
  });

  // Kick off on load
  resizeCanvas();
  updateSliderLabels();
  fetchSummary();
  triggerMine(true);
  fetchSweepGrid();
  fetchManualExample(0);
  drawGraph();
})();
