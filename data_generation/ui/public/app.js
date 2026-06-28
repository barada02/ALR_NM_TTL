// Core Application Client State
const state = {
    datasets: [],
    selectedDataset: null,
    selectedSample: null,
    currentPage: 1,
    totalPages: 1,
    limit: 10,
    searchQuery: '',
    samples: []
};

// DOM Elements
const elements = {
    datasetList: document.getElementById('datasetList'),
    samplesList: document.getElementById('samplesList'),
    viewerContent: document.getElementById('viewerContent'),
    searchInput: document.getElementById('searchInput'),
    btnPrev: document.getElementById('btnPrev'),
    btnNext: document.getElementById('btnNext'),
    pageIndicator: document.getElementById('pageIndicator'),
    itemsCount: document.getElementById('itemsCount'),
    schemaBadge: document.getElementById('schemaBadge')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    fetchDatasets();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Search input
    let searchTimeout;
    elements.searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        state.searchQuery = e.target.value;
        searchTimeout = setTimeout(() => {
            state.currentPage = 1;
            fetchDatasetSamples();
        }, 300);
    });

    // Pagination
    elements.btnPrev.addEventListener('click', () => {
        if (state.currentPage > 1) {
            state.currentPage--;
            fetchDatasetSamples();
        }
    });

    elements.btnNext.addEventListener('click', () => {
        if (state.currentPage < state.totalPages) {
            state.currentPage++;
            fetchDatasetSamples();
        }
    });
}

// Fetch available datasets
async function fetchDatasets() {
    try {
        const response = await fetch('/api/datasets');
        state.datasets = await response.json();
        renderDatasets();
        
        // Auto-select first dataset if available
        if (state.datasets.length > 0) {
            selectDataset(state.datasets[0]);
        } else {
            elements.datasetList.innerHTML = '<div class="loading-spinner-small">No datasets found. Run pipeline first.</div>';
        }
    } catch (err) {
        console.error('Failed to load datasets', err);
        elements.datasetList.innerHTML = '<div class="loading-spinner-small">Error loading datasets.</div>';
    }
}

// Render datasets list in sidebar
function renderDatasets() {
    elements.datasetList.innerHTML = '';
    state.datasets.forEach(db => {
        const sizeMb = (db.sizeBytes / (1024 * 1024)).toFixed(2);
        const div = document.createElement('div');
        div.className = 'dataset-item';
        if (state.selectedDataset && state.selectedDataset.name === db.name) {
            div.className += ' active';
        }
        
        div.innerHTML = `
            <div class="name">${db.displayName}</div>
            <div class="meta">${sizeMb} MB • JSONL</div>
        `;
        
        div.addEventListener('click', () => selectDataset(db));
        elements.datasetList.appendChild(div);
    });
}

// Select a dataset
function selectDataset(db) {
    state.selectedDataset = db;
    state.currentPage = 1;
    state.selectedSample = null;
    state.searchQuery = '';
    elements.searchInput.value = '';
    elements.searchInput.disabled = false;
    
    // Re-render sidebar to update active state
    renderDatasets();
    
    // Fetch samples
    fetchDatasetSamples();
    
    // Reset Detail view
    resetDetailViewer();
}

// Fetch samples
async function fetchDatasetSamples() {
    if (!state.selectedDataset) return;
    
    elements.samplesList.innerHTML = `
        <div class="loading-spinner">
            <div class="spinner"></div>
            <span>Streaming samples...</span>
        </div>
    `;
    
    const url = `/api/dataset/${state.selectedDataset.name}?page=${state.currentPage}&limit=${state.limit}&query=${encodeURIComponent(state.searchQuery)}`;
    
    try {
        const response = await fetch(url);
        const result = await response.json();
        
        state.samples = result.items;
        state.totalPages = result.totalPages || 1;
        state.currentPage = result.page;
        
        renderSamplesList(result.totalItems);
        updatePagination();
    } catch (err) {
        console.error('Failed to load samples', err);
        elements.samplesList.innerHTML = '<div class="empty-state"><h3>Error</h3><p>Could not read samples.</p></div>';
    }
}

// Render Left Panel List
function renderSamplesList(totalCount) {
    elements.itemsCount.textContent = `${totalCount.toLocaleString()} samples found`;
    elements.samplesList.innerHTML = '';
    
    if (state.samples.length === 0) {
        elements.samplesList.innerHTML = '<div class="empty-state"><h3>No results</h3><p>Try refining your search keyword.</p></div>';
        return;
    }
    
    state.samples.forEach(sample => {
        const card = document.createElement('div');
        card.className = 'sample-card';
        if (state.selectedSample && state.selectedSample.index === sample.index) {
            card.className += ' selected';
        }
        
        // Schema configurations
        let tag = 'SFT';
        let tagClass = 'tag-lc';
        let title = 'Sample Entry';
        let description = '';
        
        const type = sample.data._type || '';
        
        if (type === 'long_context') {
            tag = 'Long Context';
            tagClass = 'tag-lc';
            title = sample.data.question || 'Long Context Story';
            description = sample.data.context || '';
        } else if (type === 'dialogue') {
            tag = 'Dialogue';
            tagClass = 'tag-diag';
            title = sample.data.test_question || 'Dialogue Scenario';
            const firstTurn = sample.data.turns && sample.data.turns[0] ? sample.data.turns[0].content : '';
            description = firstTurn ? `User: "${firstTurn}"` : '';
        } else if (type === 'agentic') {
            tag = 'Agentic';
            tagClass = 'tag-agent';
            title = sample.data.task || 'Agentic Flow';
            description = sample.data.decision_point ? `Requires memory of step ${sample.data.decision_point.requires_memory_of_step}` : '';
        }
        
        card.innerHTML = `
            <div class="sample-card-header">
                <span class="sample-card-index">#${sample.index + 1}</span>
                <span class="sample-card-tag ${tagClass}">${tag}</span>
            </div>
            <div class="sample-card-topic" title="${title}">${title}</div>
            <div class="sample-card-desc">${description}</div>
        `;
        
        card.addEventListener('click', () => selectSample(sample));
        elements.samplesList.appendChild(card);
    });
}

// Pagination updates
function updatePagination() {
    elements.btnPrev.disabled = state.currentPage <= 1;
    elements.btnNext.disabled = state.currentPage >= state.totalPages;
    elements.pageIndicator.textContent = `Page ${state.currentPage} of ${state.totalPages}`;
}

// Select sample
function selectSample(sample) {
    state.selectedSample = sample;
    
    // Update active list card selection style
    const cards = elements.samplesList.querySelectorAll('.sample-card');
    cards.forEach((card, idx) => {
        if (state.samples[idx] && state.samples[idx].index === sample.index) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
    
    renderDetailViewer();
}

// Reset viewer content
function resetDetailViewer() {
    elements.schemaBadge.textContent = 'No Data';
    elements.schemaBadge.className = 'badge';
    elements.viewerContent.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <h3>No Sample Selected</h3>
            <p>Select a dataset and click on a sample from the list to view its complete structure.</p>
        </div>
    `;
}

// Render Detailed View
function renderDetailViewer() {
    if (!state.selectedSample) return;
    
    const sample = state.selectedSample.data;
    const type = sample._type || '';
    
    elements.schemaBadge.textContent = type.replace('_', ' ');
    elements.schemaBadge.className = 'badge active';
    
    let html = `<div class="viewer-card">`;
    
    if (type === 'long_context') {
        html += renderLongContext(sample);
    } else if (type === 'dialogue') {
        html += renderDialogue(sample);
    } else if (type === 'agentic') {
        html += renderAgentic(sample);
    } else {
        // Fallback raw JSON viewer
        html += `
            <div class="viewer-section">
                <h2>Raw JSON Content</h2>
                <pre style="white-space: pre-wrap; font-family: monospace; font-size: 12px; color: var(--text-secondary);">${JSON.stringify(sample, null, 2)}</pre>
            </div>
        `;
    }
    
    html += `</div>`;
    elements.viewerContent.innerHTML = html;
}

// ── Long Context Renderer ────────────────────────────────────────────────────
function renderLongContext(data) {
    // Escape regex characters
    const escapeRegExp = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    
    // Highlight facts inside context
    let highlightedContext = data.context || '';
    if (data.facts && data.facts.length > 0) {
        data.facts.forEach(fact => {
            // If the fact matches text in context, highlight it. 
            // Often facts are summarized, so we highlight matches of keywords or numbers.
            // As a simple safety, we do exact substring replacement for exact sentences if possible.
            try {
                // Find potential numbers or unique keys in the fact to highlight
                const numRegex = /\b\d+(?:[.,]\d+)?\b/g;
                let match;
                const numbers = [];
                while ((match = numRegex.exec(fact)) !== null) {
                    numbers.push(match[0]);
                }
                
                // Highlight exact matches of fact phrases if found
                if (highlightedContext.includes(fact)) {
                    highlightedContext = highlightedContext.split(fact).join(`<span class="highlight-fact">${fact}</span>`);
                } else {
                    // Try to highlight numbers/identifiers from the fact that appear in the text
                    numbers.forEach(num => {
                        const escapedNum = escapeRegExp(num);
                        const regex = new RegExp(`\\b${escapedNum}\\b`, 'g');
                        highlightedContext = highlightedContext.replace(regex, `<span class="highlight-fact">${num}</span>`);
                    });
                }
            } catch (e) {
                // Fallback if regex fails
            }
        });
    }
    
    return `
        <div class="viewer-meta-row">
            <div class="meta-pill">Position: <strong>${data.fact_position || 'unknown'}</strong></div>
            <div class="meta-pill">Distractor Count: <strong>${data.distractor_count || 0}</strong></div>
        </div>

        <div class="viewer-section">
            <h2>📖 Document Context</h2>
            <div class="story-text">${highlightedContext}</div>
        </div>

        <div class="viewer-section">
            <h2>🔑 Core Facts (Gold Standard)</h2>
            <div class="facts-list">
                ${(data.facts || []).map((fact, i) => `
                    <div class="fact-item">
                        <strong>Fact #${i + 1}:</strong> ${fact}
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="viewer-section">
            <h2>❓ SFT Memory Evaluation</h2>
            <div style="font-size: 14px; line-height: 1.6;">
                <p style="margin-bottom: 12px; font-weight: 600; color: white;">Question:</p>
                <p style="margin-bottom: 20px; color: var(--text-secondary); background: rgba(0,0,0,0.15); padding: 12px; border-radius: 6px; border-left: 3px solid var(--accent);">${data.question || 'No question generated'}</p>
                
                <p style="margin-bottom: 6px; font-weight: 600; color: white;">Target Answer (Verbatim):</p>
                <p style="font-size: 15px; color: var(--success); font-weight: 700; background: rgba(16, 185, 129, 0.05); padding: 12px; border-radius: 6px; border: 1px dashed rgba(16, 185, 129, 0.2);">${data.answer || 'No answer'}</p>
            </div>
        </div>
    `;
}

// ── Dialogue Renderer ────────────────────────────────────────────────────────
function renderDialogue(data) {
    const turns = data.turns || [];
    
    // Check if turns exist
    let chatHtml = '<div class="chat-container">';
    turns.forEach((turn, i) => {
        const isUser = turn.role === 'user';
        const speaker = isUser ? 'User' : 'Assistant';
        const bubbleClass = isUser ? 'user' : 'assistant';
        
        // Find if this turn is a memory anchor
        const anchor = (data.memory_anchors || []).find(a => a.established_at_turn === i || a.recalled_at_turn === i);
        let anchorBorder = '';
        let anchorBadge = '';
        if (anchor) {
            if (anchor.established_at_turn === i) {
                anchorBorder = 'style="border: 1px dashed var(--warning);"';
                anchorBadge = `<span class="badge" style="background: rgba(245, 158, 11, 0.15); color: #f59e0b; margin-left: 8px;">Anchor Established</span>`;
            } else if (anchor.recalled_at_turn === i) {
                anchorBorder = 'style="border: 1px dashed var(--accent-secondary);"';
                anchorBadge = `<span class="badge" style="background: rgba(236, 72, 153, 0.15); color: #ec4899; margin-left: 8px;">Anchor Recalled</span>`;
            }
        }
        
        chatHtml += `
            <div class="chat-bubble ${bubbleClass}" ${anchorBorder}>
                <div class="speaker-tag">${speaker}${anchorBadge}</div>
                <div>${turn.content}</div>
                ${anchor && anchor.established_at_turn === i ? `<div style="font-size: 10px; color: #f59e0b; margin-top: 6px; font-weight: 600;">Stored info: "${anchor.info}"</div>` : ''}
            </div>
        `;
    });
    chatHtml += '</div>';

    return `
        <div class="viewer-meta-row">
            <div class="meta-pill">Turns: <strong>${turns.length}</strong></div>
            <div class="meta-pill">Anchors: <strong>${(data.memory_anchors || []).length}</strong></div>
        </div>

        <div class="viewer-section">
            <h2>💬 Stateful Dialogue</h2>
            ${chatHtml}
        </div>

        <div class="viewer-section">
            <h2>❓ Memory Test Challenge</h2>
            <div style="font-size: 14px; line-height: 1.6;">
                <p style="margin-bottom: 12px; font-weight: 600; color: white;">Final Question:</p>
                <p style="margin-bottom: 20px; color: var(--text-secondary); background: rgba(0,0,0,0.15); padding: 12px; border-radius: 6px; border-left: 3px solid var(--accent);">${data.test_question || 'No test question generated'}</p>
                
                <p style="margin-bottom: 6px; font-weight: 600; color: white;">Gold Answer:</p>
                <p style="font-size: 15px; color: var(--success); font-weight: 700; background: rgba(16, 185, 129, 0.05); padding: 12px; border-radius: 6px; border: 1px dashed rgba(16, 185, 129, 0.2);">${data.correct_answer || 'No answer'}</p>
            </div>
        </div>
    `;
}

// ── Agentic Renderer ─────────────────────────────────────────────────────────
function renderAgentic(data) {
    const steps = data.steps || [];
    const dp = data.decision_point || {};
    
    return `
        <div class="viewer-meta-row">
            <div class="meta-pill">Total Steps: <strong>${steps.length}</strong></div>
            <div class="meta-pill">Decision Step: <strong>${dp.at_step || 'unknown'}</strong></div>
            <div class="meta-pill">Requires Step: <strong>${dp.requires_memory_of_step || 'unknown'}</strong></div>
        </div>

        <div class="viewer-section">
            <h2>🛠️ High-Level Task</h2>
            <p style="font-size: 15px; font-weight: 500; color: white; line-height: 1.5; background: rgba(255, 255, 255, 0.01); padding: 16px; border-radius: 8px; border: 1px solid var(--border-color);">${data.task || 'No task description'}</p>
        </div>

        <div class="viewer-section">
            <h2>📈 Agent Action Sequence</h2>
            <div class="timeline">
                ${steps.map(step => {
                    const isDecisionPoint = step.step === dp.at_step;
                    const borderStyle = isDecisionPoint ? 'style="border-color: var(--accent); box-shadow: 0 0 10px var(--accent-glow);"' : '';
                    return `
                        <div class="timeline-step">
                            <div class="timeline-dot" ${isDecisionPoint ? 'style="border-color: var(--accent-secondary);"' : ''}></div>
                            <div class="step-card" ${borderStyle}>
                                <div class="step-header">
                                    <span class="step-title">Step ${step.step}</span>
                                    <span class="step-action">${step.action}</span>
                                </div>
                                <div class="step-grid">
                                    <div class="step-label">Input:</div>
                                    <div class="step-val">${step.input || 'None'}</div>
                                    
                                    <div class="step-label">Output:</div>
                                    <div class="step-val">${step.output || 'None'}</div>
                                </div>
                                ${step.key_result ? `
                                    <div class="step-key-res">
                                        💡 <strong>Key Result:</strong> ${step.key_result}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>

        <div class="viewer-section decision-alert">
            <h2>⚠️ SFT Logic Evaluation: Decision Point (Step ${dp.at_step})</h2>
            <p style="font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
                At this step, the agent must recall the output from <strong>Step ${dp.requires_memory_of_step}</strong> to make the correct choice.
            </p>
            
            <div class="decision-grid">
                <div class="decision-card" style="border-color: rgba(16, 185, 129, 0.3);">
                    <div class="decision-card-title correct">Correct Action (Perfect Memory)</div>
                    <div class="decision-card-body">${dp.correct_decision}</div>
                </div>
                <div class="decision-card" style="border-color: rgba(239, 68, 68, 0.3);">
                    <div class="decision-card-title wrong">Memoryless Action (Forgot Step ${dp.requires_memory_of_step})</div>
                    <div class="decision-card-body">${dp.wrong_decision_if_forgotten}</div>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <p style="font-size: 12px; font-weight: 700; color: white; margin-bottom: 6px;">Final Task Completion Answer:</p>
                <div style="font-size: 14px; font-weight: 600; color: var(--success); background: rgba(16, 185, 129, 0.05); padding: 10px; border-radius: 6px; border: 1px dashed rgba(16, 185, 129, 0.2);">${data.final_answer}</div>
            </div>
        </div>
    `;
}
