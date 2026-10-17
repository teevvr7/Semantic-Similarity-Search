/**
 * Semantic Similarity Search Client Application
 * Handles state management, REST API orchestration, token attribution, and interactive UI components.
 */

// Application State
const state = {
  currentMode: "hybrid", // "hybrid", "compare", "word2vec", "fasttext", "tfidf"
  query: "",
  topK: 5,
  category: "",
  isLoading: false,
  categories: [],
  systemStats: null,
};

// DOM References
const queryInput = document.getElementById("query-input");
const clearQueryBtn = document.getElementById("clear-query-btn");
const modeSelector = document.getElementById("mode-selector");
const categorySelect = document.getElementById("category-select");
const topkSelect = document.getElementById("topk-select");

const statusPill = document.getElementById("system-status-pill");
const statusText = document.getElementById("system-status-text");

const inspectionRibbon = document.getElementById("inspection-ribbon");
const metricLatency = document.getElementById("metric-latency");
const metricCorpus = document.getElementById("metric-corpus");
const tokensContainer = document.getElementById("tokens-container");
const fallbackAlert = document.getElementById("fallback-alert");
const fallbackMessage = document.getElementById("fallback-message");

const singleResultsContainer = document.getElementById("single-results-container");
const comparativeContainer = document.getElementById("comparative-container");
const resultsCountSub = document.getElementById("results-count-sub");
const emptyState = document.getElementById("empty-state");

const drawerBackdrop = document.getElementById("drawer-backdrop");
const drawerCategory = document.getElementById("drawer-category");
const drawerDocId = document.getElementById("drawer-doc-id");
const drawerTitle = document.getElementById("drawer-title");
const drawerContent = document.getElementById("drawer-content");
const drawerKeywords = document.getElementById("drawer-keywords");
const drawerCleanedTokens = document.getElementById("drawer-cleaned-tokens");

const themeToggleBtn = document.getElementById("theme-toggle-btn");
const themeIconSun = document.getElementById("theme-icon-sun");
const themeIconMoon = document.getElementById("theme-icon-moon");

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initEventListeners();
  checkSystemHealth();
  loadCategories();
});

/* ==========================================================================
   Theme Management
   ========================================================================== */
function initTheme() {
  const savedTheme = localStorage.getItem("search-theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcons(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  const target = current === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", target);
  localStorage.setItem("search-theme", target);
  updateThemeIcons(target);
}

function updateThemeIcons(theme) {
  if (theme === "dark") {
    themeIconSun.style.display = "block";
    themeIconMoon.style.display = "none";
  } else {
    themeIconSun.style.display = "none";
    themeIconMoon.style.display = "block";
  }
}

/* ==========================================================================
   Event Listeners Setup
   ========================================================================== */
function initEventListeners() {
  // Theme Toggle
  themeToggleBtn.addEventListener("click", toggleTheme);

  // Search Input Controls
  queryInput.addEventListener("input", () => {
    state.query = queryInput.value.trim();
    clearQueryBtn.style.display = state.query.length > 0 ? "block" : "none";
  });

  clearQueryBtn.addEventListener("click", () => {
    queryInput.value = "";
    state.query = "";
    clearQueryBtn.style.display = "none";
    queryInput.focus();
    resetResultsView();
  });

  // Global Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    // Focus search on '/' or 'Ctrl+K'
    if ((e.key === "/" || (e.ctrlKey && e.key === "k")) && document.activeElement !== queryInput) {
      e.preventDefault();
      queryInput.focus();
      queryInput.select();
    }
    // Close Drawer or Clear on 'Escape'
    if (e.key === "Escape") {
      closeDrawer();
    }
  });

  // Retrieval Mode Switcher Tabs
  modeSelector.querySelectorAll(".mode-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      modeSelector.querySelectorAll(".mode-tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentMode = btn.getAttribute("data-mode");

      // Auto re-execute query if text exists
      if (state.query) {
        triggerSearch();
      }
    });
  });

  // Category and TopK Selectors
  categorySelect.addEventListener("change", (e) => {
    state.category = e.target.value;
    if (state.query) triggerSearch();
  });

  topkSelect.addEventListener("change", (e) => {
    state.topK = parseInt(e.target.value, 10);
    if (state.query) triggerSearch();
  });
}

/* ==========================================================================
   System Health & Categories Loading
   ========================================================================== */
async function checkSystemHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.models_ready) {
      statusText.textContent = `Engine Ready (${data.total_documents.toLocaleString()} Docs)`;
      statusPill.querySelector(".status-dot").style.backgroundColor = "var(--success)";
    } else {
      statusText.textContent = "Loading Models (30-40s)...";
      statusPill.querySelector(".status-dot").style.backgroundColor = "var(--warning)";
      setTimeout(checkSystemHealth, 3000);
    }
  } catch (err) {
    statusText.textContent = "Server Offline";
    statusPill.querySelector(".status-dot").style.backgroundColor = "var(--danger)";
    setTimeout(checkSystemHealth, 5000);
  }
}

async function loadCategories() {
  try {
    const res = await fetch("/api/categories");
    if (!res.ok) return;
    const categories = await res.json();
    state.categories = categories;

    // Populate dropdown
    categories.forEach((cat) => {
      const opt = document.createElement("option");
      opt.value = cat;
      opt.textContent = cat;
      categorySelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load categories:", err);
  }
}

/* ==========================================================================
   Quick Presets
   ========================================================================== */
function setQueryPreset(text) {
  queryInput.value = text;
  state.query = text;
  clearQueryBtn.style.display = "block";
  triggerSearch();
}

/* ==========================================================================
   Search Execution Pipeline
   ========================================================================== */
async function triggerSearch() {
  const query = queryInput.value.trim();
  if (!query) return;

  state.query = query;
  setLoadingState(true);

  if (state.currentMode === "compare") {
    await executeComparativeSearch(query);
  } else {
    await executeSingleModelSearch(query);
  }

  setLoadingState(false);
}

function setLoadingState(loading) {
  state.isLoading = loading;
  if (loading) {
    resultsCountSub.textContent = "Querying semantic vector space...";
  }
}

function resetResultsView() {
  inspectionRibbon.style.display = "none";
  singleResultsContainer.innerHTML = "";
  singleResultsContainer.appendChild(emptyState);
  singleResultsContainer.style.display = "flex";
  comparativeContainer.style.display = "none";
  resultsCountSub.textContent = "";
}

/* ==========================================================================
   Single Model Search
   ========================================================================== */
async function executeSingleModelSearch(query) {
  try {
    const payload = {
      query: query,
      mode: state.currentMode,
      top_k: state.topK,
      category: state.category || null,
      min_score: 0.0,
    };

    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      renderError(err.detail || "Search request failed");
      return;
    }

    const data = await res.json();
    renderInspectionRibbon(data.preprocessing, data.latency_ms, data.total_corpus);
    renderSingleResults(data.results, data.mode);
  } catch (err) {
    renderError("Network error or server unavailable.");
  }
}

/* ==========================================================================
   Comparative Multi-Model Search (Requirement 4.5)
   ========================================================================== */
async function executeComparativeSearch(query) {
  try {
    const payload = {
      query: query,
      top_k: state.topK,
      category: state.category || null,
    };

    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      renderError(err.detail || "Comparative search failed");
      return;
    }

    const data = await res.json();
    renderInspectionRibbon(data.preprocessing, data.latency_ms, 50000);
    renderComparativeResults(data);
  } catch (err) {
    renderError("Network error executing comparison.");
  }
}

/* ==========================================================================
   Explainability & Token Attribution Ribbon Rendering
   ========================================================================== */
function renderInspectionRibbon(preprocessing, latencyMs, totalCorpus) {
  inspectionRibbon.style.display = "block";
  metricLatency.textContent = `${latencyMs.toFixed(1)} ms`;
  metricCorpus.textContent = totalCorpus.toLocaleString();

  // Render Token Pills
  tokensContainer.innerHTML = "";
  if (preprocessing.cleaned_tokens.length === 0) {
    tokensContainer.innerHTML = '<span style="font-size:0.78rem; color:var(--text-dim);">(No valid tokens remaining after stopword removal)</span>';
  } else {
    preprocessing.token_attributions.forEach((attr) => {
      const span = document.createElement("span");
      span.className = `token-badge ${attr.is_fallback ? "token-fallback" : "token-primary"}`;
      span.title = attr.details;
      span.innerHTML = `
        <span>${escapeHtml(attr.token)}</span>
        <small style="font-size:0.65rem; opacity:0.85;">${attr.is_fallback ? "(FastText OOV)" : "(Word2Vec)"}</small>
      `;
      tokensContainer.appendChild(span);
    });
  }

  // Fallback Callout Alert
  if (preprocessing.fallback_triggered) {
    fallbackAlert.style.display = "flex";
    const unseenList = preprocessing.fallback_tokens.map(t => `'${t}'`).join(", ");
    fallbackMessage.innerHTML = `
      Term(s) <strong>${escapeHtml(unseenList)}</strong> not in Word2Vec training vocabulary.
      FastText character n-grams extracted semantic sub-word vectors as fallback.
    `;
  } else {
    fallbackAlert.style.display = "none";
  }
}

/* ==========================================================================
   Render Single Model Results
   ========================================================================== */
function renderSingleResults(results, modeName) {
  comparativeContainer.style.display = "none";
  singleResultsContainer.style.display = "flex";
  singleResultsContainer.innerHTML = "";

  resultsCountSub.textContent = `— ${results.length} ranked match${results.length === 1 ? "" : "es"} retrieved via ${modeName}`;

  if (results.length === 0) {
    singleResultsContainer.innerHTML = `
      <div class="empty-state-box">
        <h3 class="empty-state-title">No Relevant Documents Found</h3>
        <p class="empty-state-desc">The query did not yield any document matches exceeding the similarity threshold.</p>
      </div>
    `;
    return;
  }

  results.forEach((item) => {
    const card = document.createElement("div");
    card.className = "result-card";

    const scorePct = Math.max(0, Math.min(100, (item.similarity_score * 100))).toFixed(1);
    const scoreColorClass = getScoreColorClass(item.similarity_score);

    const keywordsHtml = (item.keywords || [])
      .map((kw) => `<span class="keyword-chip">${escapeHtml(kw)}</span>`)
      .join("");

    card.innerHTML = `
      <div class="card-top-row">
        <div class="card-meta-left">
          <span class="rank-badge">#${item.rank}</span>
          <span class="category-tag">${escapeHtml(item.category)}</span>
          <span class="doc-id-pill">${escapeHtml(item.document_id)}</span>
        </div>
        <div class="similarity-meter-wrap">
          <div class="score-bar-bg" title="Cosine Similarity: ${item.similarity_score.toFixed(4)}">
            <div class="score-bar-fill ${scoreColorClass}" style="width: ${scorePct}%;"></div>
          </div>
          <span class="similarity-score-text">${scorePct}%</span>
        </div>
      </div>

      <h3 class="card-title">${escapeHtml(item.title)}</h3>
      <p class="card-snippet">${highlightKeywords(escapeHtml(item.content_snippet), state.query)}</p>

      <div class="card-footer">
        <div class="keywords-chips">
          ${keywordsHtml}
        </div>
        <button type="button" class="view-doc-btn" onclick="openDocumentDrawer('${item.document_id}')">
          <span>Inspect Document</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12 5 19 12 12 19"></polyline>
          </svg>
        </button>
      </div>
    `;

    singleResultsContainer.appendChild(card);
  });
}

/* ==========================================================================
   Render Comparative Multi-Column Results (Requirement 4.5)
   ========================================================================== */
function renderComparativeResults(data) {
  singleResultsContainer.style.display = "none";
  comparativeContainer.style.display = "grid";

  resultsCountSub.textContent = `— Side-by-side comparative evaluation across 4 retrieval strategies`;

  renderCompColumn("comp-hybrid-list", data.hybrid_results);
  renderCompColumn("comp-w2v-list", data.word2vec_results);
  renderCompColumn("comp-ft-list", data.fasttext_results);
  renderCompColumn("comp-tfidf-list", data.tfidf_results);
}

function renderCompColumn(elementId, items) {
  const container = document.getElementById(elementId);
  container.innerHTML = "";

  if (!items || items.length === 0) {
    container.innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem; text-align:center; padding:1.5rem 0;">No matches (0.00 score)</div>';
    return;
  }

  items.forEach((item) => {
    const mini = document.createElement("div");
    mini.className = "comp-mini-card";
    const scorePct = Math.max(0, Math.min(100, item.similarity_score * 100)).toFixed(1);
    const scoreClass = getScoreColorClass(item.similarity_score);

    mini.innerHTML = `
      <div class="comp-mini-top">
        <span class="rank-badge">#${item.rank}</span>
        <span style="font-weight:700; font-size:0.78rem; font-variant-numeric:tabular-nums;">${scorePct}%</span>
      </div>
      <div class="comp-mini-title">${escapeHtml(item.title)}</div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
        <span class="comp-mini-category">${escapeHtml(item.category)}</span>
        <button type="button" class="view-doc-btn" style="padding:2px 6px; font-size:0.7rem;" onclick="openDocumentDrawer('${item.document_id}')">
          View
        </button>
      </div>
    `;
    container.appendChild(mini);
  });
}

/* ==========================================================================
   Document Detail Slide-Over Drawer
   ========================================================================== */
async function openDocumentDrawer(docId) {
  drawerBackdrop.classList.add("active");
  drawerDocId.textContent = docId;
  drawerTitle.textContent = "Loading document...";
  drawerContent.textContent = "Fetching full article content from repository...";
  drawerKeywords.innerHTML = "";
  drawerCleanedTokens.textContent = "";

  try {
    const res = await fetch(`/api/document/${encodeURIComponent(docId)}`);
    if (!res.ok) {
      drawerContent.textContent = "Failed to load document details.";
      return;
    }
    const doc = await res.json();
    drawerCategory.textContent = doc.category;
    drawerTitle.textContent = doc.title;
    drawerContent.textContent = doc.content;
    drawerCleanedTokens.textContent = doc.cleaned_text_str || "(No preprocessed tokens)";

    drawerKeywords.innerHTML = (doc.keywords || [])
      .map((k) => `<span class="keyword-chip">${escapeHtml(k)}</span>`)
      .join("");
  } catch (err) {
    drawerContent.textContent = "Network error while loading document.";
  }
}

function closeDrawer() {
  drawerBackdrop.classList.remove("active");
}

function closeDocumentDrawer(e) {
  if (e.target === drawerBackdrop) {
    closeDrawer();
  }
}

/* ==========================================================================
   Utility Helpers
   ========================================================================== */
function getScoreColorClass(score) {
  if (score >= 0.7) return "score-high";
  if (score >= 0.4) return "score-medium";
  if (score > 0.05) return "score-low";
  return "score-zero";
}

function renderError(msg) {
  singleResultsContainer.style.display = "flex";
  comparativeContainer.style.display = "none";
  singleResultsContainer.innerHTML = `
    <div class="empty-state-box">
      <h3 class="empty-state-title" style="color:var(--danger);">Error</h3>
      <p class="empty-state-desc">${escapeHtml(msg)}</p>
    </div>
  `;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function highlightKeywords(text, query) {
  if (!query || !text) return text;
  const words = query.trim().split(/\s+/).filter(w => w.length > 2);
  if (words.length === 0) return text;

  try {
    const pattern = new RegExp(`(${words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
    return text.replace(pattern, "<mark style='background-color:rgba(99,102,241,0.18); color:inherit; padding:1px 3px; border-radius:2px;'>$1</mark>");
  } catch (e) {
    return text;
  }
}
