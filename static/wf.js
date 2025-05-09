// Default configurations
const DEFAULT_CONFIGS = {
    modelConfig: {
        "parameter_evaluation": "mistralai/mixtral-8x22b-instruct",
        "condition_finder": "mistralai/mixtral-8x22b-instruct"
    },
    interfaceConfig: {
        "model": "mistralai/mixtral-8x22b-instruct",
        "extras": {"max_tokens": 8000}
    },
    projectConfig: {
        "name": "custom_test",
        "num_retries_permodel": 3
    }
};

document.addEventListener('DOMContentLoaded', function() {
    // Initialize textareas with default values
    document.getElementById('modelConfig').value = JSON.stringify(DEFAULT_CONFIGS.modelConfig, null, 2);
    document.getElementById('interfaceConfig').value = JSON.stringify(DEFAULT_CONFIGS.interfaceConfig, null, 2);
    document.getElementById('projectConfig').value = JSON.stringify(DEFAULT_CONFIGS.projectConfig, null, 2);

    // Toggle advanced configuration
    document.getElementById('toggleAdvanced').addEventListener('click', function() {
        const advancedConfig = document.getElementById('advancedConfig');
        const isHidden = advancedConfig.classList.contains('hidden');
        advancedConfig.classList.toggle('hidden');
        this.textContent = isHidden ? 'Hide Advanced Configuration' : 'Show Advanced Configuration';
    });

    // Submit workflow
    document.getElementById('submitWorkflow').addEventListener('click', handleWorkflowSubmit);

    // Add close button handler for JSON panel
    document.getElementById('closeJsonPanel')?.addEventListener('click', () => {
        const jsonPanel = document.getElementById('jsonPanel');
        if (jsonPanel) {
            jsonPanel.classList.add('hidden');
        }
    });

    // Add copy button handler
    document.getElementById('copyJson')?.addEventListener('click', () => {
        const jsonContent = document.getElementById('jsonContent');
        if (jsonContent) {
            navigator.clipboard.writeText(jsonContent.textContent)
                .then(() => {
                    // Show a temporary success message
                    const copyBtn = document.getElementById('copyJson');
                    const originalText = copyBtn.textContent;
                    copyBtn.textContent = 'Copied!';
                    setTimeout(() => {
                        copyBtn.textContent = originalText;
                    }, 2000);
                })
                .catch(err => console.error('Failed to copy:', err));
        }
    });
});

// Log Dialog Management
class LogManager {
    constructor() {
        this.dialog = document.getElementById('logDialog');
        this.content = document.getElementById('logContent');
        this.messages = document.getElementById('logMessages');
        this.toggleBtn = document.getElementById('toggleLogs');
        this.clearBtn = document.getElementById('clearLogs');
        
        this.isMinimized = false;
        this.maxMessages = 100; // Maximum number of messages to keep
        
        this.setupEventListeners();
        this.setupSSE();
    }

    setupEventListeners() {
        this.toggleBtn.addEventListener('click', () => this.toggleDialog());
        this.clearBtn.addEventListener('click', () => this.clearLogs());
    }

    toggleDialog() {
        this.isMinimized = !this.isMinimized;
        this.dialog.classList.toggle('log-dialog-minimized', this.isMinimized);
        
        // Rotate arrow icon
        this.toggleBtn.querySelector('svg').style.transform = 
            this.isMinimized ? 'rotate(180deg)' : 'rotate(0deg)';
    }

    clearLogs() {
        this.messages.innerHTML = '';
        this.addMessage('Logs cleared', 'info');
    }

    addMessage(message, type = 'info') {
        // Create new message element
        const msgElement = document.createElement('div');
        msgElement.className = `log-message log-message-${type} p-2 rounded`;
        
        // Add timestamp
        const timestamp = new Date().toLocaleTimeString();
        msgElement.textContent = `[${timestamp}] ${message}`;
        
        // Add to container
        this.messages.appendChild(msgElement);
        
        // Scroll to bottom
        this.content.scrollTop = this.content.scrollHeight;
        
        // Limit number of messages
        while (this.messages.children.length > this.maxMessages) {
            this.messages.removeChild(this.messages.firstChild);
        }
    }

    setupSSE() {
        const eventSource = new EventSource('/logs');
        
        eventSource.onmessage = (event) => {
            const logEntry = JSON.parse(event.data);
            this.addMessage(logEntry.message, logEntry.level);
        };

        eventSource.onerror = (error) => {
            console.error('SSE Error:', error);
            eventSource.close();
            // Try to reconnect after 5 seconds
            setTimeout(() => this.setupSSE(), 5000);
        };
    }
}

// Initialize log manager
const logManager = new LogManager();

class WorkflowManager {
    constructor(logManager) {
        this.logManager = logManager;
        this.currentWorkflowId = null;
        this.statusCheckInterval = null;
        this.isMinimized = false;
        
        // Setup minimize functionality
        this.setupMinimizePanel();
    }

    setupMinimizePanel() {
        const minimizeBtn = document.getElementById('minimizePanel');
        const jsonPanel = document.getElementById('jsonPanel');
        
        if (minimizeBtn && jsonPanel) {
            minimizeBtn.addEventListener('click', () => {
                this.isMinimized = !this.isMinimized;
                if (this.isMinimized) {
                    jsonPanel.style.transform = 'translateX(calc(100% - 40px))';
                    minimizeBtn.innerHTML = `
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                        </svg>
                    `;
                } else {
                    jsonPanel.style.transform = 'translateX(0)';
                    minimizeBtn.innerHTML = `
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
                        </svg>
                    `;
                }
            });

            // Add hover effect
            jsonPanel.addEventListener('mouseenter', () => {
                if (this.isMinimized) {
                    jsonPanel.style.transform = 'translateX(0)';
                }
            });

            jsonPanel.addEventListener('mouseleave', () => {
                if (this.isMinimized) {
                    jsonPanel.style.transform = 'translateX(calc(100% - 40px))';
                }
            });
        }
    }

    async startWorkflow(payload) {
        try {
            const response = await fetch('/workflow/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail?.error || 'Failed to start workflow');
            
            this.currentWorkflowId = result.workflow_id;
            this.startStatusChecking();
            
            return result;
        } catch (error) {
            this.logManager.addMessage(`Error starting workflow: ${error.message}`, 'error');
            throw error;
        }
    }
    
    async stopWorkflow() {
        if (!this.currentWorkflowId) {
            this.logManager.addMessage('No active workflow to stop', 'warning');
            return;
        }
        
        try {
            this.logManager.addMessage('Requesting workflow stop...', 'info');
            
            const response = await fetch(`/stop-workflow/${this.currentWorkflowId}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Failed to stop workflow');
            
            this.logManager.addMessage('✅ Workflow stop requested successfully', 'success');
            // Don't clear workflow ID here - wait for status check to confirm stop
            
        } catch (error) {
            this.logManager.addMessage(`❌ Error stopping workflow: ${error.message}`, 'error');
        }
    }
    
    async updateResults(workflowId) {
        try {
            // Fetch results JSON
            const resultsResponse = await fetch(`/results/${workflowId}`);
            if (resultsResponse.ok) {
                const resultsData = await resultsResponse.json();
                this.displayJsonPanel(resultsData);
                
                // Also update the timeline image
                const timelinePlot = document.getElementById('timelinePlot');
                if (timelinePlot) {
                    timelinePlot.src = `/timeline/${workflowId}`;
                }
                
                // Show the panel
                const jsonPanel = document.getElementById('jsonPanel');
                if (jsonPanel) {
                    jsonPanel.classList.remove('hidden');
                }
            }

            // Add success message to logs
            this.logManager.addMessage('✅ Results and timeline are now available', 'success');
        } catch (error) {
            this.logManager.addMessage(`Error loading results: ${error.message}`, 'error');
        }
    }

    displayJsonPanel(jsonData) {
        // Format the JSON with indentation and syntax highlighting
        const formattedJson = JSON.stringify(jsonData, null, 2);
        
        // Update the content with syntax highlighting
        const jsonContent = document.getElementById('jsonContent');
        if (jsonContent) {
            jsonContent.textContent = formattedJson;
            jsonContent.style.color = '#1e293b'; // slate-800
            jsonContent.style.fontSize = '0.875rem'; // text-sm
        }
    }

    startStatusChecking() {
        if (this.statusCheckInterval) clearInterval(this.statusCheckInterval);
        
        this.statusCheckInterval = setInterval(async () => {
            if (!this.currentWorkflowId) return;
            
            try {
                const response = await fetch(`/workflow-status/${this.currentWorkflowId}`);
                const status = await response.json();
                
                if (status.status === 'stopped' || status.status === 'completed' || status.status === 'not_found') {
                    await this.updateResults(this.currentWorkflowId);
                    clearInterval(this.statusCheckInterval);
                    this.currentWorkflowId = null;
                    this.updateControlButtons(false);
                }
                
                // Log current status
                this.logManager.addMessage(`Current workflow status: ${status.status}`, 'info');
                
            } catch (error) {
                console.error('Error checking workflow status:', error);
                this.logManager.addMessage(`Error checking status: ${error.message}`, 'error');
            }
        }, 5000);
    }
    
    updateControlButtons(isRunning) {
        const startBtn = document.getElementById('submitWorkflow');
        const stopBtn = document.getElementById('stopWorkflow');
        
        if (startBtn && stopBtn) {  // Add null check
            startBtn.disabled = isRunning;
            stopBtn.disabled = !isRunning;
        }
    }
}

// Initialize managers
const workflowManager = new WorkflowManager(logManager);

// Update the handleWorkflowSubmit function
async function handleWorkflowSubmit() {
    const submitButton = document.getElementById('submitWorkflow');
    
    try {
        logManager.addMessage('Starting workflow submission...', 'info');
        
        const apiKey = document.getElementById('apiKey').value;
        if (!apiKey) {
            logManager.addMessage('API key is required', 'error');
            alert('Please enter an API key');
            return;
        }

        const payload = {
            calculation_prompt: document.getElementById('calculationPrompt').value,
            workflow_index: 0,
            gen_model_hierarchy: document.getElementById('modelHierarchy').value.split(','),
            model_config: JSON.parse(document.getElementById('modelConfig').value),
            interface_agent_kwargs: JSON.parse(document.getElementById('interfaceConfig').value),
            lm_api: apiKey,
            project_config: JSON.parse(document.getElementById('projectConfig').value),
        };

        submitButton.disabled = true;
        submitButton.textContent = 'Processing...';
        logManager.addMessage('Sending request to server...', 'info');

        workflowManager.updateControlButtons(true);
        await workflowManager.startWorkflow(payload);
        
        logManager.addMessage('Workflow started successfully', 'success');

    } catch (error) {
        logManager.addMessage(`Error: ${error.message}`, 'error');
        alert(`Error: ${error.message}`);
        console.error('Error:', error);
        workflowManager.updateControlButtons(false);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Start Workflow';
    }
}

// Add stop button handler with visual feedback
document.getElementById('stopWorkflow').addEventListener('click', async () => {
    const stopBtn = document.getElementById('stopWorkflow');
    try {
        stopBtn.textContent = 'Stopping...';
        stopBtn.disabled = true;
        await workflowManager.stopWorkflow();
    } catch (error) {
        console.error('Error stopping workflow:', error);
    } finally {
        stopBtn.textContent = 'Stop Workflow';
    }
});