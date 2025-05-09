// Wait for DOM to be fully loaded before attaching event listeners
document.addEventListener('DOMContentLoaded', function() {
    const workflowForm = document.getElementById('workflowForm');
    const statusDisplay = document.getElementById('statusDisplay');
    const resultsDisplay = document.getElementById('resultsDisplay');
    const jsonDisplay = document.getElementById('jsonDisplay');
    const timelinePlot = document.getElementById('timelinePlot');
    const logDisplay = document.getElementById('logDisplay');
    const downloadJson = document.getElementById('downloadJson');
    const downloadPlot = document.getElementById('downloadPlot');

    // Function to update status display
    function updateStatus(message, isError = false) {
        const statusHTML = `
            <div class="${isError ? 'text-red-600' : 'text-green-600'} font-medium">
                ${message}
            </div>
        `;
        statusDisplay.innerHTML = statusHTML;
    }

    // Function to update results
    function displayResults(results) {
        jsonDisplay.textContent = JSON.stringify(results, null, 2);
        resultsDisplay.classList.remove('hidden');
    }

    // Function to update timeline plot
    function displayTimeline(imageUrl) {
        timelinePlot.src = imageUrl;
    }

    // Function to update log
    function updateLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        logDisplay.innerHTML += `[${timestamp}] ${message}\n`;
        logDisplay.scrollTop = logDisplay.scrollHeight;
    }

    // Handle form submission
    workflowForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        try {
            updateStatus('Starting workflow...');
            
            const formData = new FormData(workflowForm);
            const response = await fetch('/workflow/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(Object.fromEntries(formData)),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            updateStatus(`Workflow started with ID: ${data.workflow_id}`);
            updateLog(`Workflow ${data.workflow_id} started successfully`);

            // Poll for results
            pollResults(data.workflow_id);

        } catch (error) {
            updateStatus(`Error: ${error.message}`, true);
            updateLog(`Error: ${error.message}`);
        }
    });

    // Function to poll for results
    async function pollResults(workflowId) {
        try {
            const response = await fetch(`/results/${workflowId}`);
            if (!response.ok) {
                if (response.status === 404) {
                    // Results not ready yet, poll again after delay
                    setTimeout(() => pollResults(workflowId), 5000);
                    return;
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            displayResults(results);
            
            // Get timeline plot
            const timelineUrl = `/timeline/${workflowId}`;
            displayTimeline(timelineUrl);
            
            updateStatus('Workflow completed successfully');
            updateLog('Results received and displayed');

        } catch (error) {
            updateStatus(`Error polling results: ${error.message}`, true);
            updateLog(`Error polling results: ${error.message}`);
        }
    }

    // Handle download buttons
    downloadJson.addEventListener('click', function() {
        const jsonString = jsonDisplay.textContent;
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'workflow_results.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    downloadPlot.addEventListener('click', function() {
        const a = document.createElement('a');
        a.href = timelinePlot.src;
        a.download = 'workflow_timeline.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
});