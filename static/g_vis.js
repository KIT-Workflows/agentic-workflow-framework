// Add error handling utility
function showError(message) {
    const errorContainer = document.getElementById('error-container');
    errorContainer.textContent = message;
    errorContainer.classList.remove('hidden');
    setTimeout(() => errorContainer.classList.add('hidden'), 5000);
}

// Add loading indicator management
function toggleLoading(show) {
    const loader = document.getElementById('loading-indicator');
    if (show) {
        loader.classList.remove('hidden');
    } else {
        loader.classList.add('hidden');
    }
}

// Add debug utilities
function updateDebug(message, level = 'info') {
    const debugContent = document.getElementById('debug-content');
    const debugCount = document.getElementById('debug-count');
    const timestamp = new Date().toLocaleTimeString();
    
    // Create new debug entry
    const entry = document.createElement('div');
    entry.className = `debug-entry mb-2 ${level}`;
    entry.innerHTML = `
        <span class="text-gray-400">[${timestamp}]</span>
        <span class="${level === 'error' ? 'text-red-500' : 
                     level === 'warning' ? 'text-yellow-500' : 
                     'text-green-500'}">${message}</span>
    `;
    
    // Add to debug panel
    debugContent.appendChild(entry);
    debugContent.scrollTop = debugContent.scrollHeight;
    
    // Update count
    const count = debugContent.getElementsByClassName('debug-entry').length;
    debugCount.textContent = count;
}

// Initialize debug panel
document.addEventListener('DOMContentLoaded', () => {
    updateDebug('Debug panel initialized');
    
    // Debug panel toggle
    const debugHeader = document.getElementById('debug-header');
    const debugContent = document.getElementById('debug-content');
    const debugToggleIcon = document.getElementById('debug-toggle-icon');
    
    debugHeader.addEventListener('click', () => {
        debugContent.classList.toggle('hidden');
        debugToggleIcon.style.transform = 
            debugContent.classList.contains('hidden') ? 'rotate(-90deg)' : 'rotate(0deg)';
    });
    
    // Clear debug button
    document.getElementById('clearDebug').addEventListener('click', (e) => {
        e.stopPropagation();
        debugContent.innerHTML = '';
        updateDebug('Debug panel cleared');
    });
    
    // Start network initialization
    initializeNetwork().catch(error => {
        updateDebug(`Network initialization failed: ${error.message}`, 'error');
    });

    // Add layout change handler
    const layoutSelect = document.getElementById('layoutSelect');
    if (layoutSelect) {
        layoutSelect.addEventListener('change', (e) => {
            updateLayout(e.target.value);
        });
    }

    // Register SVG extension when available
    if (cytoscape.use && window.cytoscapeSvg) {
        try {
            cytoscape.use(cytoscapeSvg);
            updateDebug('SVG extension registered successfully');
        } catch (error) {
            updateDebug(`Failed to register SVG extension: ${error.message}`, 'error');
        }
    } else {
        updateDebug('SVG extension not available', 'warning');
    }
});

// Add layout management
function updateLayout(layoutName) {
    if (!cy) {
        showError('Network not initialized');
        return;
    }

    updateDebug(`Changing layout to: ${layoutName}`);
    
    let layoutConfig;
    switch (layoutName) {
        case 'breadthfirst':
            layoutConfig = {
                name: 'breadthfirst',
                directed: true,
                padding: 50,
                spacingFactor: 1.5,
                animate: true,
                animationDuration: 500,
                fit: true,
                roots: function(ele) {
                    // Find nodes with no incoming edges (root nodes)
                    return cy.nodes().roots();
                }
            };
            break;
            
        case 'fcose':
        default:
            layoutConfig = {
                name: 'fcose',
                animate: true,
                animationDuration: 500,
                randomize: true,
                padding: 50
            };
            break;
    }

    // Run the layout
    try {
        const layout = cy.layout(layoutConfig);
        layout.run();
        updateDebug(`Layout updated to: ${layoutName}`);
    } catch (error) {
        updateDebug(`Layout update failed: ${error.message}`, 'error');
        showError(`Failed to update layout: ${error.message}`);
    }
}

// Add color mapping for namelists
const namelistColors = {
    '&CONTROL': '#2196F3',   // Blue
    '&SYSTEM': '#9C27B0',    // Purple
    '&ELECTRONS': '#FF9800', // Orange
    '&IONS': '#F44336',      // Red
    '&CELL': '#009688',      // Teal
    '&FCP': '#795548',       // Brown
    '&RISM': '#4CAF50',      // Green
    'Card: ATOMIC_SPECIES': '#E91E63',    // X | Mass_X | PseudoPot_X
    'Card: ATOMIC_POSITIONS': '#673AB7',  // X | x | y | z | if_pos(1) | if_pos(2) | if_pos(3)
    'Card: K_POINTS': '#3F51B5',         // nks | xk_x | xk_y | xk_z | wk | nk1 | nk2 | nk3 | sk1 | sk2 | sk3
    'Card: ADDITIONAL_K_POINTS': '#FF4081', // nks_add | k_x | k_y | k_z | wk_
    'Card: CELL_PARAMETERS': '#00BCD4',   // v1 | v2 | v3
    'Card: CONSTRAINTS': '#8BC34A',       // nconstr | constr_tol | constr_type | constr(1) | constr(2) | constr(3) | constr(4) | constr_target
    'Card: OCCUPATIONS': '#FFEB3B',       // f_inp1 | f_inp2
    'Card: ATOMIC_VELOCITIES': '#FF9800', // V | vx | vy | vz
    'Card: ATOMIC_FORCES': '#FF5722',     // X | fx | fy | fz
    'Card: SOLVENTS': '#795548',          // X | Density | Molecule | X | Density_Left | Density_Right | Molecule
    'Card: HUBBARD': '#607D8B',           // Hubbard parameters
    'default': '#9E9E9E'                  // Grey for unknown namelists
};

// Add function to get color for a namelist
function getNamelistColor(namelist) {
    return namelistColors[namelist] || namelistColors['default'];
}

// Add function to get contrasting text color
function getContrastColor(hexcolor) {
    // Convert hex to RGB
    const r = parseInt(hexcolor.slice(1,3), 16);
    const g = parseInt(hexcolor.slice(3,5), 16);
    const b = parseInt(hexcolor.slice(5,7), 16);
    
    // Calculate luminance
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    
    // Return black or white based on luminance
    return luminance > 0.5 ? '#000000' : '#ffffff';
}

// Update initializeNetwork with zoom settings
async function initializeNetwork() {
    updateDebug('Starting network initialization...');
    toggleLoading(true);
    
    try {
        updateDebug('Fetching data from server...');
        const response = await fetch('/api/data');
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server returned ${response.status}: ${errorText}`);
        }
        
        const data = await response.json();
        updateDebug(`Received data: ${data.nodes?.length || 0} nodes, ${data.edges?.length || 0} edges`);
        
        if (!data.nodes?.length) {
            throw new Error('No nodes received from server');
        }
        
        // Validate edges before initialization
        const validEdges = data.edges.filter(edge => {
            if (!edge.source || !edge.target) {
                updateDebug(`Skipping invalid edge: missing source or target`, 'warning');
                return false;
            }
            const sourceExists = data.nodes.some(node => node.id === edge.source);
            const targetExists = data.nodes.some(node => node.id === edge.target);
            if (!sourceExists || !targetExists) {
                updateDebug(`Skipping edge: ${edge.source} -> ${edge.target} (missing nodes)`, 'warning');
                return false;
            }
            return true;
        });
        
        updateDebug(`Validated edges: ${validEdges.length} valid out of ${data.edges.length} total`);
        
        // Get the currently selected layout
        const layoutSelect = document.getElementById('layoutSelect');
        const initialLayout = layoutSelect ? layoutSelect.value : 'fcose';

        // Initialize Cytoscape with the selected layout
        updateDebug('Initializing Cytoscape...');
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: {
                nodes: data.nodes.map(node => ({
                    data: { 
                        ...node,
                        backgroundColor: getNamelistColor(node.group)
                    }
                })),
                edges: validEdges.map(edge => ({
                    data: { ...edge }
                }))
            },
            style: [
                {
                    selector: 'node',
                    style: {
                        'label': 'data(label)',
                        'background-color': 'data(backgroundColor)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '8px',
                        'text-wrap': 'wrap',
                        'text-max-width': '80px',
                        'color': function(ele) {
                            return getContrastColor(ele.data('backgroundColor'));
                        },
                        'text-outline-width': 1,
                        'text-outline-color': function(ele) {
                            return ele.data('backgroundColor');
                        },
                        'shape': 'roundrectangle',
                        'width': 'label',
                        'height': 'label',
                        'padding': '5px',
                        'transition-property': 'background-color, opacity',
                        'transition-duration': '0.2s'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 1,
                        'line-color': '#999',
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'target-arrow-color': '#999',
                        'opacity': 0.6,
                        'transition-property': 'line-color, opacity',
                        'transition-duration': '0.2s'
                    }
                },
                {
                    selector: 'node:selected',
                    style: {
                        'border-width': 2,
                        'border-color': '#FFF',
                        'border-opacity': 0.8,
                        'background-opacity': 1
                    }
                },
                {
                    selector: 'edge:selected',
                    style: {
                        'width': 2,
                        'line-color': '#FFF',
                        'target-arrow-color': '#FFF',
                        'opacity': 1
                    }
                },
                {
                    selector: '.highlighted',
                    style: {
                        'opacity': 1,
                        'z-index': 999
                    }
                },
                {
                    selector: '.faded',
                    style: {
                        'opacity': 0.25
                    }
                },
                {
                    selector: 'edge.highlighted',
                    style: {
                        'line-color': '#FFF',
                        'target-arrow-color': '#FFF',
                        'width': 2
                    }
                }
            ],
            layout: {
                name: initialLayout,
                animate: false,
                padding: 50,
                spacingFactor: 1.5,
                directed: true,
                ...(initialLayout === 'breadthfirst' ? {
                    roots: function(ele) {
                        return cy.nodes().roots();
                    }
                } : {})
            },
            minZoom: 0.2,         // Minimum zoom level
            maxZoom: 2.2,         // Maximum zoom level
            zoomingEnabled: true,
            userZoomingEnabled: true,
            wheelSensitivity: 0.2, // Reduce mouse wheel sensitivity
        });
        
        // Initial zoom level
        cy.zoom({
            level: 0.8, // Set default zoom level
            renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 }
        });

        // Add zoom buttons if you want manual controls
        addZoomControls();
        
        // Add legend
        createLegend();
        
        // Add hover handlers
        cy.on('mouseover', 'node', function(e) {
            const node = e.target;
            const neighborhood = node.neighborhood().add(node);
            
            cy.elements().addClass('faded');
            neighborhood.removeClass('faded').addClass('highlighted');
        });

        cy.on('mouseout', 'node', function(e) {
            cy.elements().removeClass('faded').removeClass('highlighted');
        });
        
        updateDebug('Network initialized successfully');
        
        // Update node click handler
        cy.on('tap', 'node', async function(evt) {
            const node = evt.target;
            const nodeId = node.id();
            updateDebug(`Fetching details for node: ${nodeId}`);
            
            try {
                const response = await fetch(`/api/node/${encodeURIComponent(nodeId)}`);
                if (!response.ok) {
                    throw new Error(`Failed to fetch node details: ${response.statusText}`);
                }
                
                const details = await response.json();
                showNodeDetails(details);
            } catch (error) {
                updateDebug(`Error fetching node details: ${error.message}`, 'error');
                showError(`Failed to load details for ${nodeId}`);
            }
        });
        
    } catch (error) {
        updateDebug(`ERROR: ${error.message}`, 'error');
        showError(`Failed to initialize network: ${error.message}`);
        throw error;
    } finally {
        toggleLoading(false);
    }
}

// Add function to create legend
function createLegend() {
    const legend = document.createElement('div');
    legend.id = 'graph-legend';
    legend.className = 'fixed bottom-4 right-4 bg-white/90 p-4 rounded-lg shadow-lg max-w-xs';
    legend.style.maxHeight = '200px';
    legend.style.overflowY = 'auto';
    
    const title = document.createElement('div');
    title.className = 'font-bold mb-2 text-sm';
    title.textContent = 'Namelist Groups';
    legend.appendChild(title);
    
    Object.entries(namelistColors).forEach(([name, color]) => {
        if (name === 'default') return;
        
        const item = document.createElement('div');
        item.className = 'flex items-center gap-2 text-xs mb-1';
        
        const colorBox = document.createElement('div');
        colorBox.className = 'w-3 h-3 rounded';
        colorBox.style.backgroundColor = color;
        
        const label = document.createElement('span');
        label.textContent = name;
        
        item.appendChild(colorBox);
        item.appendChild(label);
        legend.appendChild(item);
    });
    
    document.body.appendChild(legend);
}

// Update zoom controls function with new position
function addZoomControls() {
    const controls = document.createElement('div');
    // Update position to top-right, accounting for the top control panel
    controls.className = 'fixed top-24 right-4 flex gap-2'; // Changed from bottom-4 left-4 to top-24 right-4
    controls.innerHTML = `
        <button id="zoom-in" class="bg-white/90 p-2 rounded-lg shadow-lg hover:bg-white">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
            </svg>
        </button>
        <button id="zoom-out" class="bg-white/90 p-2 rounded-lg shadow-lg hover:bg-white">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
            </svg>
        </button>
        <button id="zoom-fit" class="bg-white/90 p-2 rounded-lg shadow-lg hover:bg-white">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-5V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 4h-4m4 0l-5-5"/>
            </svg>
        </button>
    `;
    document.body.appendChild(controls);

    // Add event listeners
    document.getElementById('zoom-in').addEventListener('click', () => {
        cy.animate({
            zoom: cy.zoom() * 1.2,
            duration: 200
        });
    });

    document.getElementById('zoom-out').addEventListener('click', () => {
        cy.animate({
            zoom: cy.zoom() * 0.8,
            duration: 200
        });
    });

    document.getElementById('zoom-fit').addEventListener('click', () => {
        cy.fit(cy.elements(), 50);
    });
}

// Update performSearch to use gentler zoom
function performSearch(searchTerm) {
    if (!cy) {
        showError('Network not initialized');
        return;
    }
    
    if (!searchTerm?.trim()) {
        cy.elements().removeClass('hidden');
        return;
    }
    
    try {
        const term = searchTerm.trim().toLowerCase();
        cy.elements().addClass('hidden');
        
        const matchedNodes = cy.nodes().filter(node => 
            node.data('label')?.toLowerCase().includes(term)
        );
        
        if (matchedNodes.length === 0) {
            showStatus('No matches found');
            return;
        }
        
        matchedNodes.removeClass('hidden');
        matchedNodes.connectedEdges().removeClass('hidden');
        
        if (matchedNodes.length > 0) {
            cy.animate({
                fit: {
                    eles: matchedNodes,
                    padding: 50
                },
                duration: 500,
                easing: 'ease-in-out-cubic'
            });
        }
        
    } catch (error) {
        showError(`Search failed: ${error.message}`);
        updateDebug(`Search error: ${error.message}`, 'error');
    }
}

// Add search suggestions functionality
function updateSearchSuggestions(searchTerm) {
    if (!cy) return;
    
    const suggestionsContainer = document.getElementById('searchSuggestions');
    suggestionsContainer.innerHTML = '';
    
    if (!searchTerm?.trim()) {
        suggestionsContainer.classList.add('hidden');
        return;
    }
    
    const term = searchTerm.trim().toLowerCase();
    const matches = cy.nodes()
        .filter(node => node.data('label')?.toLowerCase().includes(term))
        .map(node => node.data('label'))
        .slice(0, 10); // Limit to 10 suggestions
    
    if (matches.length === 0) {
        suggestionsContainer.classList.add('hidden');
        return;
    }
    
    matches.forEach(match => {
        const div = document.createElement('div');
        div.className = 'px-4 py-2 hover:bg-blue-100 cursor-pointer';
        div.textContent = match;
        div.addEventListener('click', () => {
            document.getElementById('searchBox').value = match;
            performSearch(match);
            suggestionsContainer.classList.add('hidden');
        });
        suggestionsContainer.appendChild(div);
    });
    
    suggestionsContainer.classList.remove('hidden');
}

// Initialize search functionality
document.addEventListener('DOMContentLoaded', () => {
    // ... existing initialization code ...
    
    const searchBox = document.getElementById('searchBox');
    const searchButton = document.getElementById('searchButton');
    const suggestionsContainer = document.getElementById('searchSuggestions');
    
    if (searchBox) {
        // Add input handler for search suggestions
        searchBox.addEventListener('input', (e) => {
            updateSearchSuggestions(e.target.value);
        });
        
        // Add keyboard navigation for suggestions
        searchBox.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                performSearch(searchBox.value);
                suggestionsContainer.classList.add('hidden');
            } else if (e.key === 'Escape') {
                suggestionsContainer.classList.add('hidden');
            }
        });
        
        // Close suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchBox.contains(e.target) && !suggestionsContainer.contains(e.target)) {
                suggestionsContainer.classList.add('hidden');
            }
        });
    }
    
    if (searchButton) {
        searchButton.addEventListener('click', () => {
            performSearch(searchBox.value);
            suggestionsContainer.classList.add('hidden');
        });
    }
});

// Add status message function
function showStatus(message) {
    const status = document.getElementById('status-message');
    status.textContent = message;
    status.classList.remove('hidden');
    setTimeout(() => status.classList.add('hidden'), 3000);
}

// Add function to format JSON for display
function formatJSON(obj) {
    if (!obj) return '';
    
    const formatValue = (value) => {
        if (value === null) return '<span class="json-null">null</span>';
        if (typeof value === 'boolean') return `<span class="json-boolean">${value}</span>`;
        if (typeof value === 'number') return `<span class="json-number">${value}</span>`;
        if (typeof value === 'string') return `<span class="json-string">"${value}"</span>`;
        if (Array.isArray(value)) {
            if (value.length === 0) return '[]';
            return '[\n' + value.map(v => '  ' + formatValue(v)).join(',\n') + '\n]';
        }
        if (typeof value === 'object') {
            const entries = Object.entries(value);
            if (entries.length === 0) return '{}';
            return '{\n' + entries.map(([k, v]) => 
                `  <span class="json-key">"${k}"</span>: ${formatValue(v)}`
            ).join(',\n') + '\n}';
        }
        return String(value);
    };
    
    return formatValue(obj);
}

// Update show node details function with copy functionality
function showNodeDetails(details) {
    const panel = document.getElementById('details-panel');
    const content = document.getElementById('details-content');
    const title = document.getElementById('details-title');
    
    // Create copy button if it doesn't exist
    let copyButton = document.getElementById('copy-details');
    if (!copyButton) {
        copyButton = document.createElement('button');
        copyButton.id = 'copy-details';
        copyButton.className = 'text-gray-400 hover:text-white flex items-center gap-1';
        copyButton.innerHTML = `
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"/>
            </svg>
            <span>Copy</span>
        `;
        
        // Add copy functionality
        copyButton.addEventListener('click', () => {
            try {
                // Create a temporary textarea to copy the formatted JSON
                const tempTextArea = document.createElement('textarea');
                tempTextArea.value = JSON.stringify(details, null, 2);
                document.body.appendChild(tempTextArea);
                tempTextArea.select();
                document.execCommand('copy');
                document.body.removeChild(tempTextArea);
                
                // Show success feedback
                copyButton.innerHTML = `
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M5 13l4 4L19 7"/>
                    </svg>
                    <span>Copied!</span>
                `;
                copyButton.className = 'text-green-400 flex items-center gap-1';
                
                // Reset button after 2 seconds
                setTimeout(() => {
                    copyButton.innerHTML = `
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                  d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"/>
                        </svg>
                        <span>Copy</span>
                    `;
                    copyButton.className = 'text-gray-400 hover:text-white flex items-center gap-1';
                }, 2000);
                
                showStatus('Content copied to clipboard');
            } catch (error) {
                showError('Failed to copy content');
                updateDebug(`Copy error: ${error.message}`, 'error');
            }
        });
        
        // Add button to panel header
        const header = document.querySelector('#details-panel > div');
        header.insertBefore(copyButton, document.getElementById('close-details'));
    }
    
    // Update panel content
    title.textContent = details.Parameter_Name || details.Card_Name || 'Parameter Details';
    content.innerHTML = formatJSON(details);
    
    // Show panel
    panel.style.transform = 'translateX(0)';
}

// Update event listener for closing details panel
document.addEventListener('DOMContentLoaded', () => {
    // ... existing initialization code ...
    
    // Add close button handler for details panel
    const closeButton = document.getElementById('close-details');
    if (closeButton) {
        closeButton.addEventListener('click', () => {
            const panel = document.getElementById('details-panel');
            panel.style.transform = 'translateX(-100%)';  // Hide to the left
        });
    }
});

// Add export functionality
function addExportButton() {
    const exportButton = document.createElement('button');
    exportButton.className = 'px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 flex items-center gap-2';
    exportButton.innerHTML = `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L12 4m4 4h-8"/>
        </svg>
        Export PNG
    `;
    
    // Add to the control panel
    const controlPanel = document.querySelector('#controls');
    controlPanel.appendChild(exportButton);
    
    // Export handler
    exportButton.addEventListener('click', () => {
        exportGraph();
    });
}

// Simplified export function for PNG only
function exportGraph() {
    if (!cy) {
        showError('Network not initialized');
        return;
    }
    
    try {
        // Get current date for filename
        const date = new Date().toISOString().split('T')[0];
        const time = new Date().toTimeString().split(' ')[0].replace(/:/g, '-');
        const filename = `graph-${date}-${time}.png`;
        
        // Export as PNG - current view only
        const blob = dataURLtoBlob(cy.png({
            scale: 2, // Higher resolution
            full: false, // Only current view
            quality: 1, // Best quality
            bg: '#ffffff', // White background
            clip: true // Clip to viewport
        }));
        
        downloadBlob(blob, filename);
        showStatus('Current view exported as PNG');
        
    } catch (error) {
        showError(`Failed to export graph: ${error.message}`);
        updateDebug(`Export error: ${error.message}`, 'error');
    }
}

// Helper functions remain the same
function dataURLtoBlob(dataURL) {
    const arr = dataURL.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], {type: mime});
}

function downloadBlob(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => window.URL.revokeObjectURL(url), 100);
}

// Update initialization to add export button
document.addEventListener('DOMContentLoaded', () => {
    // ... existing initialization code ...
    
    // Add export button
    addExportButton();
}); 