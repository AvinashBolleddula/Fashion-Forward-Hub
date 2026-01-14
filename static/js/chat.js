// Check authentication on page load
// Reads the saved JWT/access token from the browser.
const token = localStorage.getItem('access_token');
// If no token exists, redirect user to the home/login page.
if (!token) {
    window.location.href = '/';
}

// Read stored user identity
const userEmail = localStorage.getItem('user_email');
if (userEmail) {
    // Display user email in header (we'll add this to HTML next)
    console.log('Logged in as:', userEmail);
}

// Keeps a running total of tokens used across messages.
let totalTokens = 0;

// Grab DOM elements (UI references)
// These pull elements from your HTML so you can read/write UI state.

// where messages appear
const chatContainer = document.getElementById('chatContainer');
// textarea/input
const messageInput = document.getElementById('messageInput');
// buttons
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
// checkbox/toggle
const simplifiedMode = document.getElementById('simplifiedMode');
// send button text + spinner
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');

// Retrieval controls:
// Use RAG
const useRagCheckbox = document.getElementById('useRag');
// dropdown (“dense”, “hybrid”, etc.)
const retrieverTypeSelect = document.getElementById('retrieverType');
//  top-k results
const topKInput = document.getElementById('topK');
// hybrid parameters
const alphaInput = document.getElementById('alpha');
const rffKInput = document.getElementById('rffK');
// toggle reranker
const useRerankerCheckbox = document.getElementById('useReranker');
// container to show/hide hybrid params
const hybridControls = document.getElementById('hybridControls');
// text display of current config
const configDisplay = document.getElementById('configText');
// container to show/hide rerank query input
const rerankControls = document.getElementById('rerankControls');
// rerank query field
const rerankQueryInput = document.getElementById('rerankQuery');


// 1. Your existing code (RAG toggle)
// RAG toggle: show/hide controls
// Runs whenever “Use RAG” checkbox changes.
// Defines which UI blocks should be shown/hidden when RAG changes.
useRagCheckbox.addEventListener('change', () => {
    const retrievalControls = [
        retrieverTypeSelect.parentElement,
        simplifiedMode.parentElement,
        topKInput.parentElement,
        useRerankerCheckbox.parentElement,
        hybridControls
        // ❌ Remove rerankControls from here!
    ];
    
    // If RAG is ON: show those controls.
    // If RAG is ON and reranker is ON, show the rerank query controls.
    if (useRagCheckbox.checked) {
        retrievalControls.forEach(control => {
            if (control) control.style.display = 'flex';
        });
        // Check if reranker is also checked
        if (useRerankerCheckbox.checked) {
            rerankControls.style.display = 'flex';
        }
    // If RAG is OFF: hide them and also hide rerank controls.
    } else {
        retrievalControls.forEach(control => {
            if (control) control.style.display = 'none';
        });
        rerankControls.style.display = 'none';
    }
    // Refreshes the “current configuration” text.
    updateConfigDisplay();
});

// 2. ADD THIS (Reranker toggle)
// Reranker toggle: show/hide rerank query input
// Runs when “Use Reranker” checkbox changes.
useRerankerCheckbox.addEventListener('change', () => {
    // Only show rerank controls when BOTH RAG and reranker are enabled.
    if (useRerankerCheckbox.checked && useRagCheckbox.checked) {
        rerankControls.style.display = 'flex';
    } else {
        rerankControls.style.display = 'none';
    }
    // Refreshes config text.
    updateConfigDisplay();
});


// Trigger on page load
// Forces the “change” logic to run once so the UI is correct initially.
useRagCheckbox.dispatchEvent(new Event('change'));


// Show/hide hybrid controls based on retriever type
// Runs when the dropdown changes.
retrieverTypeSelect.addEventListener('change', () => {
    // Show hybrid controls only when “hybrid” is selected.
    if (retrieverTypeSelect.value === 'hybrid') {
        hybridControls.style.display = 'flex';
    } else {
        hybridControls.style.display = 'none';
    }
    // Refresh config text.
    updateConfigDisplay();
});

// Update config display when any control changes
// Add rerankQueryInput to the update listeners
[useRagCheckbox, retrieverTypeSelect, simplifiedMode, topKInput, 
 alphaInput, rffKInput, useRerankerCheckbox, rerankQueryInput].forEach(element => {
    element.addEventListener('change', updateConfigDisplay);
    // For number/text inputs, also update while typing (input event), not just on blur
    if (element.tagName === 'INPUT' && element.type !== 'checkbox') {
        element.addEventListener('input', updateConfigDisplay);
    }
});

// Creates a human-readable config line in the UI.
function updateConfigDisplay() {
    // If RAG is off, show simple message and exit.
    if (!useRagCheckbox.checked) {
        configDisplay.textContent = 'RAG: Off | Direct LLM (No Retrieval)';
        return;
    }
    
    // Reads current UI control values
    const rag = useRagCheckbox.checked ? 'On' : 'Off';
    const retriever = retrieverTypeSelect.value.toUpperCase();
    const simplified = simplifiedMode.checked ? 'Yes' : 'No';
    const topK = topKInput.value;
    const reranker = useRerankerCheckbox.checked ? 'On' : 'Off';
    
    // Builds the display string.
    let configText = `RAG: ${rag} | Retriever: ${retriever} | Simplified: ${simplified} | Top K: ${topK} | Rerank: ${reranker}`;
    
    // Adds α and k only for hybrid mode.
    if (retrieverTypeSelect.value === 'hybrid') {
        configText += ` | α: ${alphaInput.value} | k: ${rffKInput.value}`;
    }
    
    // Writes it to the UI.
    configDisplay.textContent = configText;
}

// Initialize config display
// Ensures it shows something correct immediately.
updateConfigDisplay();

// Send message on button click
// Message sending UI wiring
// Click send → sendMessage()
sendBtn.addEventListener('click', sendMessage);

// Send message on Enter (Shift+Enter for new line)
// Enter sends message, Shift+Enter inserts newline
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Clear conversation
// Clear button clears chat.
clearBtn.addEventListener('click', clearConversation);


// sendMessage(): main chat flow
async function sendMessage() {
    // Get typed message and remove extra whitespace.
    const message = messageInput.value.trim();
    
    // Don’t send empty messages.
    if (!message) return;
    
    // Disable input
    // Prevent double send.
    messageInput.disabled = true;
    sendBtn.disabled = true;
    // Swap button text with spinner.
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    
    // Add user message to UI
    addMessage(message, 'user');
    messageInput.value = '';
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    try {
        // Build request payload with all new parameters
        // Build payload for backend
        const payload = {
            message: message,
            use_rag: useRagCheckbox.checked,
            retriever_type: retrieverTypeSelect.value,
            simplified: simplifiedMode.checked,
            top_k: parseInt(topKInput.value),
            use_reranker: useRerankerCheckbox.checked
        };

        // Add reranker parameters (in the payload section)
        if (useRerankerCheckbox.checked) {
            payload.use_reranker = true;
            const rerankQuery = rerankQueryInput.value.trim();
            if (rerankQuery) {
                payload.rerank_query = rerankQuery;
            }
        } else {
            payload.use_reranker = false;
        }
        
        // Add hybrid-specific parameters
        if (retrieverTypeSelect.value === 'hybrid') {
            payload.alpha = parseFloat(alphaInput.value);
            payload.k = parseInt(rffKInput.value);
        }
        
        // Call backend API
        // Sends chat request with Bearer token
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(payload)
        });
        
        // Handle auth/rate limit/errors
        // Token invalid/expired → clear localStorage and go to login.
        if (response.status === 401) {
            localStorage.removeItem('access_token');
            localStorage.removeItem('user_email');
            window.location.href = '/';
            return;
        }
        
        // Too many requests.
        if (response.status === 429) {
            throw new Error('Rate limit exceeded. Please wait a moment.');
        }
        
        // Generic error.
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        // Process response
        // Read JSON response
        const data = await response.json();
        
        // Remove typing indicator
        // Remove dots animation
        removeTypingIndicator(typingId);
        
        // Add bot response
        // Show assistant response.
        addMessage(data.response, 'bot');
        
        // Update token stats
        // Update totals
        updateTokenStats(data.tokens);
        
        // Log retrieval config (optional)
        // Debug: show retrieval config used
        console.log('Retrieval config:', data.retrieval_config);
        
    } catch (error) {
        // Remove typing indicator, display error message as bot message.
        removeTypingIndicator(typingId);
        addMessage(error.message || 'Sorry, something went wrong. Please try again.', 'bot');
        console.error('Error:', error);
    } finally {
        // Re-enable input
        // Restore UI state, show “Send” text again, focus input.
        messageInput.disabled = false;
        sendBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        messageInput.focus();
    }
}

// addMessage(): render a chat bubble
// 
function addMessage(content, type) {
    // Creates a <div> for the message bubble.
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Adds a label "You:" or "Assistant:".
    const label = type === 'user' ? '<strong>You:</strong>' : '<strong>Assistant:</strong>';
    
    // Format message content
    // Formats content via formatMessageContent().
    const formattedContent = formatMessageContent(content);
    
    // Appends to chat container and scrolls to bottom.
    messageContent.innerHTML = `${label}<p>${formattedContent}</p>`;
    messageDiv.appendChild(messageContent);
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}


// formatMessageContent(): simple formatting
function formatMessageContent(content) {
    // Replace newlines with <br>
    content = content.replace(/\n/g, '<br>');
    
    // Simple bold formatting **text**
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Product ID highlighting
    content = content.replace(/Product ID: (\d+)/g, '<span style="color: #667eea; font-weight: 600;">Product ID: $1</span>');
    
    return content;
}

// Adds a temporary “Assistant: …” bubble with animated dots and returns id 
function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typing-indicator';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = `
        <strong>Assistant:</strong>
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    typingDiv.appendChild(messageContent);
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return 'typing-indicator';
}


// Deletes that element if it exists
function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

// Adds this response’s token usage to running total.
// 
function updateTokenStats(tokens) {
    totalTokens += tokens.total;
    
    // Updates UI elements: total and last message tokens.
    document.getElementById('totalTokens').textContent = totalTokens;
    document.getElementById('lastTokens').textContent = tokens.total;
}

// clearConversation(): reset chat
async function clearConversation() {
    // Ask user first.
    if (!confirm('Are you sure you want to clear the conversation?')) {
        return;
    }
    
    try {
        // 	Calls POST /api/clear with Authorization header.
        const response = await fetch('/api/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            }
        });
        
        // If 401 → clear storage and redirect.
        if (response.status === 401) {
            localStorage.removeItem('access_token');
            localStorage.removeItem('user_email');
            window.location.href = '/';
            return;
        }
        
        // resets chat UI to default greeting
        // resets token counters in UI
        if (response.ok) {
            // Clear UI
            chatContainer.innerHTML = `
                <div class="message bot-message">
                    <div class="message-content">
                        <strong>Assistant:</strong>
                        <p>Hi! How can I help you today? 👋</p>
                    </div>
                </div>
            `;
            
            // Reset token count
            totalTokens = 0;
            document.getElementById('totalTokens').textContent = '0';
            document.getElementById('lastTokens').textContent = '0';
        }
    } catch (error) {
        console.error('Error clearing conversation:', error);
        alert('Failed to clear conversation');
    }
}

// Logout function
// Clears token + email from localStorage and redirects to /.
function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_email');
    window.location.href = '/';
}

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});


// Display user email in header
document.getElementById('userEmail').textContent = userEmail || '';