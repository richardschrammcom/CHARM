// dashboard.js — Three-panel agent dashboard

// CHARM 3.0: every /api call carries the per-boot UI token embedded in the page. Other sites
// can't read it or send this custom header, so they can't drive the dashboard from your browser.
const CHARM_UI_TOKEN = document.querySelector('meta[name="charm-ui-token"]')?.content || '';
const _charmFetch = window.fetch.bind(window);
window.fetch = (url, opts = {}) => {
    opts.headers = Object.assign({}, opts.headers || {}, { 'X-CHARM-UI': CHARM_UI_TOKEN });
    return _charmFetch(url, opts);
};

let contextId = null;
let traceSource = null;
let agentPollInterval = null;
let sending = false; // retained for compatibility; chat is no longer globally blocked
let traceMode = 'standard'; // 'standard' or 'conversation'
let availableModels = []; // populated from server
let thinkAloud = false;
let chatInFlight = false; // true while ANY chat request is awaiting a result
let pendingRequests = new Set(); // request_ids of in-flight chats awaiting SSE results
let cloudUsage = { spent: 0, remaining: null, count: 0 }; // browser-use Cloud cost accumulated this batch
let actionWords = []; // flattened list of action words from all agents

// Whether a request is slow is a fact about the request, not about its wording, so
// there is no keyword guessing here: the notice below is driven purely by elapsed
// time and cancelled if the answer arrives first.
// How long a request may run before the chat says it's still working. Tune here.
// 5s was too eager: measured chat requests answer in 1-6s (median 3s), so the notice
// kept appearing a second or two before the answer it was apologising for. Genuinely
// slow requests delegate to the Web Agent and run for minutes — nothing falls in
// between. 10s clears the slowest ordinary reply and still lands before the Web
// Agent's own 20s browsing heartbeats take over the display.
const BACKGROUND_NOTICE_DELAY_MS = 10000;
const backgroundNotices = new Map(); // request_id -> pending timer

function initDashboard() {
    loadAgents();
    loadAssistantProfile();
    loadProfile();
    loadActionWords();
    connectTraceStream();
    setupChatForm();
    setupAssistantProfileToggle();
    setupProfileToggle();
    setupResizeHandle();

    // Poll agent status every 5 seconds, but skip if user is interacting with a dropdown
    agentPollInterval = setInterval(() => {
        if (!document.activeElement || !document.activeElement.classList.contains('agent-model-select')) {
            loadAgents();
            loadActionWords(); // Refresh in case agents registered/unregistered
        }
    }, 5000);

    document.getElementById('refresh-agents').addEventListener('click', loadAgents);
    document.getElementById('clear-trace').addEventListener('click', clearTrace);
    document.getElementById('save-assistant-profile').addEventListener('click', saveAssistantProfile);
    document.getElementById('save-profile').addEventListener('click', saveProfile);

    // Trace mode toggle
    const traceModeSelect = document.getElementById('trace-mode');
    traceModeSelect.addEventListener('change', (e) => {
        traceMode = e.target.value;
    });

    // Think out loud toggle
    const thinkCheckbox = document.getElementById('think-aloud');
    thinkCheckbox.addEventListener('change', (e) => {
        thinkAloud = e.target.checked;
    });
}

// ---- LEFT PANEL: Agent Roster ----

async function loadAgents() {
    try {
        const resp = await fetch('/api/agents');
        const data = await resp.json();
        availableModels = data.available_models || [];
        renderAgents(data.agents || []);
    } catch (e) {
        console.error('Failed to load agents:', e);
    }
}

function renderAgents(agents) {
    const container = document.getElementById('agent-list');
    container.innerHTML = '';

    agents.forEach(agent => {
        const card = document.createElement('div');
        card.className = 'agent-card';
        card.id = `agent-${agent.id}`;

        const statusText = agent.running
            ? (agent.registered ? 'Running & Registered' : 'Running')
            : 'Stopped';
        const statusDotClass = agent.running
            ? (agent.registered ? 'registered' : 'running')
            : 'stopped';

        const toggleDisabled = agent.required ? 'disabled' : '';
        const toggleChecked = agent.running ? 'checked' : '';

        // Build model dropdown options
        let modelOptions = '';
        availableModels.forEach(m => {
            const selected = m.id === agent.model ? 'selected' : '';
            modelOptions += `<option value="${m.id}" ${selected}>${m.label}</option>`;
        });

        card.innerHTML = `
            <div class="agent-card-header">
                <span class="agent-name">${agent.name}</span>
                <div class="agent-controls">
                    <button class="btn-restart" onclick="restartAgent('${agent.id}')" title="Restart">
                        <i class="fas fa-redo-alt"></i>
                    </button>
                    <label class="toggle-switch">
                        <input type="checkbox" ${toggleChecked} ${toggleDisabled}
                               onchange="toggleAgent('${agent.id}', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <span class="agent-loading"><i class="fas fa-spinner fa-spin"></i></span>
            </div>
            <div class="agent-model-row">
                <select class="agent-model-select" onchange="changeModel('${agent.id}', this.value)">
                    ${modelOptions}
                </select>
            </div>
            <div class="agent-status">
                <span class="status-dot ${statusDotClass}"></span>
                <span class="agent-status-text">${statusText}</span>
                <span class="agent-status-text" style="margin-left:auto">:${agent.port}</span>
            </div>
        `;

        container.appendChild(card);
    });
}

async function changeModel(agentId, modelId) {
    const card = document.getElementById(`agent-${agentId}`);
    if (card) card.classList.add('loading');

    try {
        const resp = await fetch(`/api/agents/${agentId}/model`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelId }),
        });
        const result = await resp.json();
        console.log(`Model change for ${agentId}:`, result);
    } catch (e) {
        console.error(`Failed to change model for ${agentId}:`, e);
    }

    // Wait for restart, then refresh
    setTimeout(() => {
        if (card) card.classList.remove('loading');
        loadAgents();
    }, 5000);
}

async function toggleAgent(agentId, shouldRun) {
    const card = document.getElementById(`agent-${agentId}`);
    if (card) card.classList.add('loading');

    const action = shouldRun ? 'start' : 'stop';
    try {
        const resp = await fetch(`/api/agents/${agentId}/${action}`, { method: 'POST' });
        const result = await resp.json();
        console.log(`Agent ${agentId} ${action}:`, result);
    } catch (e) {
        console.error(`Failed to ${action} agent ${agentId}:`, e);
    }

    // Refresh status after a brief delay
    setTimeout(() => {
        if (card) card.classList.remove('loading');
        loadAgents();
    }, 2000);
}

async function restartAgent(agentId) {
    const card = document.getElementById(`agent-${agentId}`);
    if (card) card.classList.add('loading');

    try {
        const resp = await fetch(`/api/agents/${agentId}/restart`, { method: 'POST' });
        const result = await resp.json();
        console.log(`Agent ${agentId} restart:`, result);
    } catch (e) {
        console.error(`Failed to restart agent ${agentId}:`, e);
    }

    setTimeout(() => {
        if (card) card.classList.remove('loading');
        loadAgents();
    }, 3000);
}

// ---- LEFT PANEL: Assistant Profile ----

function setupAssistantProfileToggle() {
    document.getElementById('assistant-profile-toggle').addEventListener('click', () => {
        toggleProfileSection('assistant');
    });
}

async function loadAssistantProfile() {
    try {
        const resp = await fetch('/api/assistant-profile');
        const profile = await resp.json();
        document.getElementById('assistant-name').value = profile.name || '';
        document.getElementById('assistant-personality').value = profile.personality || '';
    } catch (e) {
        console.error('Failed to load assistant profile:', e);
    }
}

async function saveAssistantProfile() {
    const profile = {
        name: document.getElementById('assistant-name').value,
        personality: document.getElementById('assistant-personality').value,
    };

    try {
        const resp = await fetch('/api/assistant-profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile),
        });
        const result = await resp.json();
        if (result.status === 'saved') {
            const msg = document.getElementById('assistant-profile-saved-msg');
            msg.textContent = result.restarted ? 'Saved & Chat Agent restarted!' : 'Saved!';
            msg.classList.add('show');
            setTimeout(() => msg.classList.remove('show'), 4000);
        }
    } catch (e) {
        console.error('Failed to save assistant profile:', e);
    }
}

// ---- LEFT PANEL: User Profile ----

function setupProfileToggle() {
    document.getElementById('profile-toggle').addEventListener('click', () => {
        toggleProfileSection('user');
    });
}

function toggleProfileSection(which) {
    // Accordion: close the other, toggle this one
    const sections = {
        assistant: {
            body: document.getElementById('assistant-profile-body'),
            chevron: document.getElementById('assistant-profile-chevron'),
        },
        user: {
            body: document.getElementById('profile-body'),
            chevron: document.getElementById('profile-chevron'),
        },
    };

    const target = sections[which];
    const other = which === 'assistant' ? sections.user : sections.assistant;
    const isOpening = target.body.style.display === 'none';

    // Close the other section
    other.body.style.display = 'none';
    other.chevron.style.transform = 'rotate(-90deg)';

    // Toggle this section
    target.body.style.display = isOpening ? 'block' : 'none';
    target.chevron.style.transform = isOpening ? 'rotate(0deg)' : 'rotate(-90deg)';
}

async function loadProfile() {
    try {
        const resp = await fetch('/api/profile');
        const profile = await resp.json();
        document.getElementById('profile-name').value = profile.name || '';
        document.getElementById('profile-email').value = profile.email || '';
        document.getElementById('profile-phone').value = profile.phone || '';
        document.getElementById('profile-role').value = profile.role || '';
        document.getElementById('profile-notes').value = profile.notes || '';
    } catch (e) {
        console.error('Failed to load profile:', e);
    }
}

async function saveProfile() {
    const profile = {
        name: document.getElementById('profile-name').value,
        email: document.getElementById('profile-email').value,
        phone: document.getElementById('profile-phone').value,
        role: document.getElementById('profile-role').value,
        notes: document.getElementById('profile-notes').value,
    };

    try {
        const resp = await fetch('/api/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile),
        });
        const result = await resp.json();
        if (result.status === 'saved') {
            const msg = document.getElementById('profile-saved-msg');
            msg.textContent = 'Saved! Restart Chat Agent to apply.';
            msg.classList.add('show');
            setTimeout(() => msg.classList.remove('show'), 3000);
        }
    } catch (e) {
        console.error('Failed to save profile:', e);
    }
}

// ---- ACTION WORDS ----

async function loadActionWords() {
    try {
        const resp = await fetch('/api/action-words');
        const data = await resp.json();
        // Flatten all agent action words into one set
        const allWords = new Set();
        for (const words of Object.values(data)) {
            if (Array.isArray(words)) {
                words.forEach(w => allWords.add(w.toLowerCase()));
            }
        }
        actionWords = Array.from(allWords);
        console.log('Action words loaded:', actionWords);
    } catch (e) {
        console.error('Failed to load action words:', e);
        actionWords = [];
    }
}

function isActionRequest(message) {
    const lower = message.toLowerCase();
    return actionWords.some(word => lower.includes(word));
}

// ---- MIDDLE PANEL: Chat ----

function autoGrowInput(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function setupChatForm() {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');

    // Grow the textarea (and wrap text) as the user types.
    input.addEventListener('input', () => autoGrowInput(input));

    // Enter sends; Shift+Enter inserts a newline.
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (form.requestSubmit) form.requestSubmit();
            else form.dispatchEvent(new Event('submit', { cancelable: true }));
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = input.value.trim();
        if (!message) return;  // input is never globally blocked — concurrent requests allowed

        input.value = '';
        autoGrowInput(input);  // shrink back to one line after sending
        addChatBubble(message, 'user');
        await sendChat(message);
    });
}

function addChatBubble(text, type) {
    const container = document.getElementById('chat-messages');

    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${type}`;
    bubble.textContent = text;

    const ts = document.createElement('span');
    ts.className = 'timestamp';
    ts.textContent = new Date().toLocaleTimeString();
    bubble.appendChild(ts);

    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;

    return bubble;
}

function addThinkingBubble(text) {
    const container = document.getElementById('chat-messages');
    const typing = document.getElementById('typing-indicator');

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble thinking';
    bubble.textContent = text;

    // Insert before the typing indicator if it exists, otherwise append
    if (typing) {
        container.insertBefore(bubble, typing);
    } else {
        container.appendChild(bubble);
    }
    container.scrollTop = container.scrollHeight;
}

// Live progress for long-running work (currently Web Agent browsing). A single
// bubble is reused and rewritten in place rather than one per heartbeat, so a
// multi-minute browse reads as steady progress instead of a stalled chat.
function updateProgressBubble(text) {
    const container = document.getElementById('chat-messages');
    const typing = document.getElementById('typing-indicator');

    let bubble = document.getElementById('chat-progress');
    if (!bubble) {
        bubble = document.createElement('div');
        bubble.id = 'chat-progress';
        bubble.className = 'chat-bubble thinking';
        if (typing) {
            container.insertBefore(bubble, typing);
        } else {
            container.appendChild(bubble);
        }
    }
    bubble.textContent = text;
    container.scrollTop = container.scrollHeight;
}

function removeProgressBubble() {
    const el = document.getElementById('chat-progress');
    if (el) el.remove();
}

function showTyping() {
    const container = document.getElementById('chat-messages');
    if (document.getElementById('typing-indicator')) return;  // avoid duplicates with concurrent requests
    const typing = document.createElement('div');
    typing.className = 'chat-typing';
    typing.id = 'typing-indicator';
    typing.innerHTML = 'Agent is thinking<span class="dots">...</span>';
    container.appendChild(typing);
    container.scrollTop = container.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

async function sendChat(message) {
    if (isActionRequest(message)) {
        const acks = [
            "On it!",
            "Let me take care of that.",
            "Working on it now.",
            "Sure, give me a moment.",
            "Got it, let me see what I can do.",
            "Let me look into that for you.",
            "Right away!",
            "I'll handle that.",
        ];
        addChatBubble(acks[Math.floor(Math.random() * acks.length)], 'agent');
    }

    // Add a trace separator for this new request
    addTraceSeparator();

    // Launch the request; the reply arrives asynchronously via the 'chat_result' SSE event.
    // The input is NOT blocked — the user can send more messages while this one runs.
    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, context_id: contextId }),
        });
        const data = await resp.json();
        if (data && data.request_id) {
            pendingRequests.add(data.request_id);
            chatInFlight = true;
            showTyping();
            // Any request still pending after the delay gets the notice. Guessing from
            // the wording missed slow ones: "what time is it in brazil?" reads like a
            // trivial question but was delegated to the Web Agent, which browsed for
            // minutes with nothing shown in the chat but "Agent is thinking...".
            const requestId = data.request_id;
            backgroundNotices.set(requestId, setTimeout(() => {
                backgroundNotices.delete(requestId);
                // While an approval card is open the request is waiting on the human, not working.
                const awaitingHuman = document.querySelector('.chat-bubble.approval:not([data-resolved])');
                if (pendingRequests.has(requestId) && !awaitingHuman) {
                    addChatBubble("Still working on this one — it's running in the background and I'll post the answer here when it's done. You can keep chatting in the meantime.", 'agent');
                }
            }, BACKGROUND_NOTICE_DELAY_MS));
        } else {
            addChatBubble((data && data.message) || 'Failed to start request.', 'error');
        }
    } catch (e) {
        addChatBubble('Failed to connect to the server.', 'error');
    }
}

// Called when a chat result arrives over SSE.
function handleChatResult(data) {
    // Only handle results for requests we launched.
    if (data.request_id && pendingRequests.size && !pendingRequests.has(data.request_id)) return;
    if (data.request_id) {
        pendingRequests.delete(data.request_id);
        // The answer beat the notice — cancel it so it never appears.
        const notice = backgroundNotices.get(data.request_id);
        if (notice) {
            clearTimeout(notice);
            backgroundNotices.delete(data.request_id);
        }
    }

    chatInFlight = pendingRequests.size > 0;
    if (!chatInFlight) {
        removeTyping();
        removeProgressBubble();
    }

    if (data.error) {
        addChatBubble(data.message, 'error');
    } else {
        addChatBubble(data.message, 'agent');
        if (data.context_id) contextId = data.context_id;
    }

    // When all in-flight requests have finished, surface any cloud usage in the chat.
    if (!chatInFlight && cloudUsage.count > 0) {
        const spent = cloudUsage.spent.toFixed(4);
        const scope = cloudUsage.count > 1 ? ` across ${cloudUsage.count} cloud browses` : '';
        let note = `💳 Browser-use Cloud: $${spent} spent${scope}`;
        if (cloudUsage.remaining !== null) note += ` · $${cloudUsage.remaining.toFixed(2)} credits remaining`;
        addChatBubble(note, 'usage');
        cloudUsage = { spent: 0, remaining: null, count: 0 };
    }

    const input = document.getElementById('chat-input');
    if (input) input.focus();
}

// Accumulate a cloud-cost trace event (carries structured numbers in metadata).
function accumulateCloudUsage(data) {
    const u = data && data.metadata && data.metadata.cloud_usage;
    if (!u) return;
    if (typeof u.spent_usd === 'number') cloudUsage.spent += u.spent_usd;
    if (typeof u.remaining_usd === 'number') cloudUsage.remaining = u.remaining_usd;
    cloudUsage.count += 1;
}

// Map trace event types to user-friendly thinking messages
function traceToThinkingMessage(data) {
    const agent = data.agent_name || '';
    const msg = data.message || '';
    const eventType = data.event_type || '';

    // Only show ChatAgent's key decision points and inter-agent communication
    if (agent === 'ChatAgent') {
        if (eventType === 'tool_call' && msg.includes('query_registry')) {
            const match = msg.match(/query_registry\("(.+?)"\)/);
            const query = match ? match[1] : 'help';
            return `🔍 I need help with this. Searching for an agent that can: "${query}"`;
        }
        if (eventType === 'tool_response' && msg.includes('Registry returned')) {
            return `📋 ${msg}`;
        }
        if (eventType === 'tool_call' && msg.includes('call_remote_agent')) {
            const match = msg.match(/→ (.+)/);
            const target = match ? match[1] : 'another agent';
            return `📞 Sending request to ${target}...`;
        }
        if (eventType === 'tool_response' && msg.includes('responded')) {
            return `📨 ${msg}`;
        }
        if (eventType === 'tool_error') {
            return `⚠️ ${msg}`;
        }
    }

    // Registry scoring results
    if (agent === 'RegistryAgent' && eventType === 'search_results') {
        return `${msg}`;
    }

    return null; // Don't show this event as a thinking bubble
}

// ---- RESIZE HANDLE ----

function setupResizeHandle() {
    const handle = document.getElementById('resize-handle');
    const chatPanel = document.getElementById('panel-chat');
    const tracePanel = document.getElementById('panel-trace');
    const container = document.querySelector('.panels-container');

    let isDragging = false;
    let startX = 0;
    let startChatWidth = 0;
    let startTraceWidth = 0;

    handle.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startChatWidth = chatPanel.getBoundingClientRect().width;
        startTraceWidth = tracePanel.getBoundingClientRect().width;

        handle.classList.add('dragging');
        document.body.classList.add('resizing');
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const dx = e.clientX - startX;
        const newChatWidth = startChatWidth + dx;
        const newTraceWidth = startTraceWidth - dx;

        // Enforce minimum widths
        if (newChatWidth < 200 || newTraceWidth < 200) return;

        chatPanel.style.flex = 'none';
        chatPanel.style.width = newChatWidth + 'px';
        tracePanel.style.width = newTraceWidth + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        handle.classList.remove('dragging');
        document.body.classList.remove('resizing');
    });
}

// ---- RIGHT PANEL: Trace Stream ----

function connectTraceStream() {
    const statusBadge = document.getElementById('trace-status');

    if (traceSource) {
        traceSource.close();
    }

    traceSource = new EventSource('/api/trace/stream?ui=' + encodeURIComponent(CHARM_UI_TOKEN));

    traceSource.addEventListener('connected', () => {
        statusBadge.textContent = 'Trace: live';
        statusBadge.className = 'badge bg-success';
    });

    traceSource.addEventListener('trace', (e) => {
        try {
            const data = JSON.parse(e.data);
            addTraceEntry(data);

            // CHARM 3.0 guard events: approvals and blocks always surface in the chat.
            if (data.event_type === 'approval_request') renderApprovalCard(data.metadata || {});
            if (data.event_type === 'approval_resolved') markApprovalResolved(data.metadata || {});
            if (data.event_type === 'guard_block') addChatBubble(data.message, 'guard');

            // Track browser-use Cloud spend so it can be summarized in the chat on completion.
            if (data.event_type === 'cloud_cost') accumulateCloudUsage(data);

            // Surface long-running work in the chat panel, not just the trace, so the
            // user can see a slow browse progressing instead of a silent "thinking".
            if (chatInFlight) {
                if (data.event_type === 'browsing') {
                    updateProgressBubble(`🌐 ${data.message}`);
                } else if (data.event_type === 'cloud_switch') {
                    updateProgressBubble('☁️ The site blocked the local browser — retrying on the cloud browser. This takes longer.');
                }
            }

            // If "Think out loud" is on and a chat request is in flight,
            // render key events as thinking bubbles in the chat panel
            if (thinkAloud && chatInFlight) {
                const thinkMsg = traceToThinkingMessage(data);
                if (thinkMsg) {
                    // Insert thinking bubble before the typing indicator
                    addThinkingBubble(thinkMsg);
                }
            }
        } catch (err) {
            console.error('Failed to parse trace event:', err);
        }
    });

    traceSource.addEventListener('chat_result', (e) => {
        try {
            handleChatResult(JSON.parse(e.data));
        } catch (err) {
            console.error('Failed to parse chat_result event:', err);
        }
    });

    traceSource.addEventListener('keepalive', () => {
        // Just keep the connection alive
    });

    traceSource.onerror = () => {
        statusBadge.textContent = 'Trace: disconnected';
        statusBadge.className = 'badge bg-danger';
        if (traceSource) traceSource.close();
        // Reconnect after 3 seconds
        setTimeout(connectTraceStream, 3000);
    };
}

function addTraceEntry(data) {
    // Filter by mode: in standard mode, skip conversation-level entries
    if (traceMode === 'standard' && data.level === 'conversation') {
        return;
    }

    const container = document.getElementById('trace-log');

    // Remove placeholder
    const placeholder = container.querySelector('.text-muted');
    if (placeholder) placeholder.remove();

    const entry = document.createElement('div');
    entry.className = 'trace-entry';
    entry.setAttribute('data-agent', data.agent_name || '');

    // Format timestamp (just time with seconds)
    let timeStr = '';
    if (data.timestamp) {
        try {
            const d = new Date(data.timestamp);
            timeStr = d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch {
            timeStr = '';
        }
    }

    // Clean agent name: "ChatAgent" -> "Chat", "chat_agent" -> "Chat"
    let displayName = data.agent_name || '?';
    displayName = displayName.replace(/_agent$/i, '').replace(/Agent$/i, '');
    displayName = displayName.charAt(0).toUpperCase() + displayName.slice(1);

    // Format: [Name] [HH:MM:SS]: message
    entry.innerHTML = `<span class="trace-agent">[${displayName}]</span> <span class="trace-time">[${timeStr}]</span><span class="trace-separator-colon">:</span> <span class="trace-message">${escapeHtml(data.message)}</span>`;

    container.appendChild(entry);

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;

    // Limit entries to prevent memory issues
    while (container.children.length > 500) {
        container.removeChild(container.firstChild);
    }
}

function addTraceSeparator() {
    const container = document.getElementById('trace-log');
    if (container.children.length > 0 && !container.lastElementChild?.classList.contains('trace-separator')) {
        const sep = document.createElement('div');
        sep.className = 'trace-separator';
        container.appendChild(sep);
    }
}

function clearTrace() {
    const container = document.getElementById('trace-log');
    container.innerHTML = '<div class="text-center text-muted p-3">Trace cleared. Waiting for agent activity...</div>';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


// ---------------------------------------------------------------------------
// CHARM 3.0: human approvals
// The orchestrator pauses a side effect and emits a signed approval_request trace event. This card
// shows exactly what would be authorized. Approve makes the dashboard (not the model) sign a
// capability token for those recipients only.
// ---------------------------------------------------------------------------

function renderApprovalCard(meta) {
    if (!meta.approval_id || document.getElementById(`approval-${meta.approval_id}`)) return;
    const container = document.getElementById('chat-messages');
    const card = document.createElement('div');
    card.className = 'chat-bubble approval';
    card.id = `approval-${meta.approval_id}`;

    const title = document.createElement('div');
    title.className = 'approval-title';
    title.textContent = `✋ ${meta.agent} wants to ${meta.action}`;
    card.appendChild(title);

    const rcpt = document.createElement('div');
    rcpt.className = 'approval-recipients';
    rcpt.textContent = `To: ${(meta.recipients || []).join(', ')}`;
    card.appendChild(rcpt);

    if (meta.reason) {
        const why = document.createElement('div');
        why.className = 'approval-reason';
        why.textContent = `Why you're being asked: ${meta.reason}`;
        card.appendChild(why);
    }

    const preview = document.createElement('pre');
    preview.className = 'approval-preview';
    preview.textContent = meta.preview || '';
    card.appendChild(preview);

    const actions = document.createElement('div');
    actions.className = 'approval-actions';
    const approve = document.createElement('button');
    approve.className = 'btn btn-sm btn-success';
    approve.textContent = 'Approve';
    const deny = document.createElement('button');
    deny.className = 'btn btn-sm btn-outline-danger';
    deny.textContent = 'Deny';
    approve.addEventListener('click', () => sendApproval(meta.approval_id, 'approve', card));
    deny.addEventListener('click', () => sendApproval(meta.approval_id, 'deny', card));
    actions.appendChild(approve);
    actions.appendChild(deny);
    card.appendChild(actions);

    const typing = document.getElementById('typing-indicator');
    if (typing) container.insertBefore(card, typing); else container.appendChild(card);
    container.scrollTop = container.scrollHeight;
}

async function sendApproval(approvalId, decision, card) {
    card.querySelectorAll('button').forEach(b => b.disabled = true);
    try {
        const resp = await fetch(`/api/approvals/${approvalId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ decision }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            setApprovalStatus(card, `Could not record decision: ${err.detail || resp.status}`);
            return;
        }
        setApprovalStatus(card, decision === 'approve' ? '✅ Approved' : '🚫 Denied');
    } catch (e) {
        setApprovalStatus(card, 'Could not reach the dashboard server.');
    }
}

function markApprovalResolved(meta) {
    const card = meta.approval_id && document.getElementById(`approval-${meta.approval_id}`);
    if (!card || card.dataset.resolved) return;
    if (meta.decision === 'timeout') setApprovalStatus(card, '⌛ Timed out. Nothing was sent.');
}

function setApprovalStatus(card, text) {
    card.dataset.resolved = '1';
    card.querySelectorAll('button').forEach(b => b.disabled = true);
    let status = card.querySelector('.approval-status');
    if (!status) {
        status = document.createElement('div');
        status.className = 'approval-status';
        card.appendChild(status);
    }
    status.textContent = text;
}
