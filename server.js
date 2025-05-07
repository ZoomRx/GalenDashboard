// server.js
const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const { spawn } = require('child_process');
// Load environment variables from .env file
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3001;

// Langfuse configuration
const LANGFUSE_HOST = (process.env.LANGFUSE_HOST || 'langfuse.zoomrx.ai').replace(/^https?:\/\//, '');

// Get project IDs from environment variables
const projectIds = {
    HCP_P: process.env.HCP_P_PROJECT_ID,
    SCOPING: process.env.SCOPING_PROJECT_ID,
    SYNAPSE: process.env.SYNAPSE_PROJECT_ID,
    HASHTAG: process.env.HASHTAG_PROJECT_ID,
    SURVEY_CODING: process.env.SURVEY_CODING_PROJECT_ID,
};

// Path to the refresh status file
const REFRESH_STATUS_FILE = path.join(__dirname, 'refresh_status.json');

// Initialize the global variable for script status
try {
    // Check if status file exists
    if (fsSync.existsSync(REFRESH_STATUS_FILE)) {
        const statusData = JSON.parse(fsSync.readFileSync(REFRESH_STATUS_FILE, 'utf8'));
        
        // Convert date strings back to date objects
        if (statusData.started) statusData.started = new Date(statusData.started);
        if (statusData.completed) statusData.completed = new Date(statusData.completed);
        
        global.pythonScriptStatus = statusData;
        
        // If the script was running but the server restarted, update status
        if (statusData.running) {
            console.log("Found running script in status file, checking if it's still active");
            // We'll check if the script is actually still running in a moment
        }
    } else {
        global.pythonScriptStatus = null;
    }
} catch (err) {
    console.error("Error reading refresh status file:", err);
    global.pythonScriptStatus = null;
}

// Function to save the current status to file
async function saveStatusToFile() {
    if (global.pythonScriptStatus) {
        try {
            await fs.writeFile(REFRESH_STATUS_FILE, JSON.stringify(global.pythonScriptStatus, null, 2));
        } catch (err) {
            console.error("Error saving refresh status to file:", err);
        }
    }
}

// Middleware to parse JSON
app.use(express.json());

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Serve dashboard_data directory
app.use('/dashboard_data', express.static('dashboard_data'));

// Serve dashboard HTML with injected environment variables
app.get('/', async (req, res) => {
    try {
        // Read the HTML file
        const htmlPath = path.join(__dirname, 'public', 'dashboard.html');
        let htmlContent = await fs.readFile(htmlPath, 'utf8');
        
        // Add environment variables to the HTML
        const envScript = `
            <script>
                window.LANGFUSE_HOST = "${LANGFUSE_HOST}";
                window.HCP_P_PROJECT_ID = "${projectIds.HCP_P || ''}";
                window.SCOPING_PROJECT_ID = "${projectIds.SCOPING || ''}";
                window.SYNAPSE_PROJECT_ID = "${projectIds.SYNAPSE || ''}";
                window.HASHTAG_PROJECT_ID = "${projectIds.HASHTAG || ''}";
                window.SURVEY_CODING_PROJECT_ID = "${projectIds.SURVEY_CODING || ''}";
                window.DEFAULT_PROJECT_ID = "${projectIds.DEFAULT || ''}";
            </script>
        `;
        
        // Insert the script before the closing </head> tag
        htmlContent = htmlContent.replace('</head>', `${envScript}</head>`);
        
        // Send the modified HTML
        res.send(htmlContent);
    } catch (error) {
        console.error('Error serving dashboard HTML:', error);
        res.status(500).send('Error loading dashboard');
    }
});

// API endpoint to get all dashboard data
app.get('/api/dashboard-data', async (req, res) => {
    try {
        const dataFiles = [
            'agent_comparison.json',
            'daily_tokens_by_agent.json',
            'daily_cost_by_agent.json',
            'user_metrics.json',
            'agent_distribution.json',
            'conversation_lengths.json',
            'model_token_usage.json',
            'all_traces.json',
            'all_sessions.json',
            'all_observations.json'
        ];

        const data = {};
        
        // Read all JSON files
        await Promise.all(dataFiles.map(async (file) => {
            try {
                const content = await fs.readFile(path.join(__dirname, 'dashboard_data', file), 'utf8');
                const key = file.replace('.json', '');
                data[key] = JSON.parse(content);
            } catch (err) {
                console.error(`Error reading ${file}:`, err);
                data[file.replace('.json', '')] = [];
            }
        }));

        res.json(data);
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
        res.status(500).json({ error: 'Failed to fetch dashboard data' });
    }
});

// API endpoint to refresh data
app.post('/api/refresh-data', async (req, res) => {
    try {
        console.log('Starting Python data refresh script in the background...');
        
        // Track script status in global variable
        global.pythonScriptStatus = {
            running: true,
            started: new Date(),
            completed: null,
            scriptStarted: false,
            scriptCompleted: false,
            lastOutput: 'Starting script...',
            exitCode: null,
            processId: null
        };
        
        // Save initial status to file
        await saveStatusToFile();
        
        const pythonProcess = spawn('/app/GalenDashboard/galendashboard/bin/python3.13', ['langfuse_data_fetcher.py']);
        
        // Store process ID for potential future use
        global.pythonScriptStatus.processId = pythonProcess.pid;
        await saveStatusToFile();
        
        pythonProcess.stdout.on('data', (data) => {
            const output = data.toString();
            console.log(`Python script output: ${output}`);
            global.pythonScriptStatus.lastOutput = output;
            
            // Look for our specific markers
            if (output.includes('SCRIPT_MARKER: LANGFUSE_DATA_FETCHER_STARTING')) {
                console.log('Python data fetcher script has STARTED');
                global.pythonScriptStatus.scriptStarted = true;
                saveStatusToFile();
            }
            
            if (output.includes('SCRIPT_MARKER: LANGFUSE_DATA_FETCHER_COMPLETED')) {
                console.log('Python data fetcher script has COMPLETED');
                global.pythonScriptStatus.scriptCompleted = true;
                global.pythonScriptStatus.completed = new Date();
                saveStatusToFile();
            }
        });
        
        pythonProcess.stderr.on('data', (data) => {
            const error = data.toString();
            console.error(`Python script error: ${error}`);
            global.pythonScriptStatus.lastOutput = `ERROR: ${error}`;
            saveStatusToFile();
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python script completed with code ${code}`);
            global.pythonScriptStatus.running = false;
            global.pythonScriptStatus.exitCode = code;
            global.pythonScriptStatus.completed = new Date();
            
            if (code === 0 && !global.pythonScriptStatus.scriptCompleted) {
                // Script completed but we didn't see the completion marker
                global.pythonScriptStatus.scriptCompleted = true;
            }
            
            saveStatusToFile();
        });
        
        // Immediately return success - the script is running in the background
        res.json({ 
            success: true, 
            message: 'Data refresh started in the background',
            status: {
                running: true,
                started: global.pythonScriptStatus.started
            }
        });
        
    } catch (error) {
        console.error('Error starting data refresh script:', error);
        res.status(500).json({ 
            success: false, 
            message: 'Failed to start data refresh',
            details: error.message
        });
    }
});

// API endpoint to check the status of the background refresh process
app.get('/api/refresh-status', (req, res) => {
    if (!global.pythonScriptStatus) {
        return res.json({
            running: false,
            message: 'No refresh task has been started yet'
        });
    }
    
    const status = global.pythonScriptStatus;
    
    // If server restarted while script was running, check if the process is still alive
    if (status.running && status.processId) {
        try {
            // Use process.kill with signal 0 to test if process exists without killing it
            process.kill(status.processId, 0);
            // If we get here, process is still running
        } catch (e) {
            // Error means process is no longer running
            if (e.code === 'ESRCH') {
                console.log(`Process ${status.processId} is no longer running, updating status`);
                status.running = false;
                status.completed = status.completed || new Date();
                status.exitCode = status.exitCode !== null ? status.exitCode : 0;
                saveStatusToFile();
            }
        }
    }
    
    const response = {
        running: status.running,
        started: status.started,
        scriptStarted: status.scriptStarted,
        scriptCompleted: status.scriptCompleted,
        lastOutput: status.lastOutput,
        message: status.running ? 'Script is still running' : 'Script has completed'
    };
    
    if (status.completed) {
        response.completed = status.completed;
        response.exitCode = status.exitCode;
        response.duration = Math.round((status.completed - status.started) / 1000); // duration in seconds
    }
    
    res.json(response);
});

// API endpoint to get observations for a specific trace
app.get('/api/traces/:traceId/observations', async (req, res) => {
    try {
        const traceId = req.params.traceId;
        // Partial match support - if the trace ID starts with the provided string
        const isPartialMatch = traceId.length < 36; // UUID is typically 36 chars
        
        // Search for observation data in the raw_api directory
        const rawDataDir = path.join(__dirname, 'dashboard_data', 'raw_api');
        
        // Get all files in the raw_api directory that contain observations data
        let observationFiles = [];
        try {
            const files = await fs.readdir(rawDataDir);
            // Look for files matching the pattern AGENT_observations_raw.json
            observationFiles = files.filter(file => file.includes('observations_raw.json'));
        } catch (err) {
            console.error("Error accessing raw_api directory:", err);
            return res.status(404).json({ error: 'No observation data files found' });
        }
        
        if (observationFiles.length === 0) {
            return res.status(404).json({ error: 'No observation data files found' });
        }
        
        // First, try to determine which agent the trace belongs to
        // This will help us prioritize the correct file first
        let traceAgentFile = null;
        let fullTraceId = null;
        
        // Try to find the trace in the traces files to determine its agent
        const traceFiles = await fs.readdir(rawDataDir)
            .then(files => files.filter(file => file.includes('traces_raw.json')))
            .catch(() => []);
            
        for (const traceFile of traceFiles) {
            try {
                const filePath = path.join(rawDataDir, traceFile);
                const data = await fs.readFile(filePath, 'utf8');
                const traces = JSON.parse(data);
                
                // Check different possible data structures
                const traceArray = Array.isArray(traces) ? traces : 
                                  traces.data ? traces.data : [];
                
                // Look for the trace - using startsWith for partial matches if needed
                const foundTrace = isPartialMatch 
                    ? traceArray.find(trace => trace.id && trace.id.startsWith(traceId))
                    : traceArray.find(trace => trace.id === traceId);
                    
                if (foundTrace) {
                    // Store the full trace ID if we're doing a partial match
                    fullTraceId = foundTrace.id;
                    
                    // Extract agent from filename (e.g., HCP_P_traces_raw.json -> HCP_P)
                    const agentName = traceFile.split('_traces_raw.json')[0];
                    traceAgentFile = `${agentName}_observations_raw.json`;
                    break;
                }
            } catch (err) {
                console.error(`Error processing trace file ${traceFile}:`, err);
            }
        }
        
        // Use the full trace ID if we found one, otherwise use the provided ID
        const searchTraceId = fullTraceId || traceId;
        
        // If we found which agent this trace belongs to, prioritize that file
        if (traceAgentFile && observationFiles.includes(traceAgentFile)) {
            // Move this file to the front of the array
            observationFiles = [
                traceAgentFile,
                ...observationFiles.filter(file => file !== traceAgentFile)
            ];
        }
        
        // Collect all observations for the trace from all files
        let allTraceObservations = [];
        
        // Process all observation files
        for (const file of observationFiles) {
            try {
                const filePath = path.join(rawDataDir, file);
                const data = await fs.readFile(filePath, 'utf8');
                const parsedData = JSON.parse(data);
                
                // Handle different data structures
                let observations = [];
                if (parsedData.data) {
                    // Format: { data: [...observations] }
                    observations = parsedData.data;
                } else if (Array.isArray(parsedData)) {
                    // Format: [...observations]
                    observations = parsedData;
                }
                
                // Filter for the current trace ID
                const traceObservations = observations.filter(obs => obs.traceId === searchTraceId);
                
                // Find observations with token data
                const obsWithTokens = traceObservations.filter(obs => {
                    const hasTokens = obs.totalTokens > 0 || 
                                    (obs.usage && obs.usage.total > 0) ||
                                    (obs.usageDetails && obs.usageDetails.total > 0);
                    return hasTokens;
                });
                
                allTraceObservations = [...allTraceObservations, ...traceObservations];
                
                // If we've found observations and this is the agent's file, we can stop here
                if (traceObservations.length > 0 && file === traceAgentFile) {
                    break;
                }
            } catch (err) {
                console.error(`Error reading or parsing ${file}:`, err);
            }
        }
        
        if (allTraceObservations.length === 0) {
            return res.status(404).json({ error: `No observations found for trace ${searchTraceId}` });
        }
        
        // Calculate total tokens across all observations
        let totalObservationTokens = 0;
        
        const tokenObservations = allTraceObservations.map((obs, index) => {
            // Extract token usage from the observation
            let tokenUsage = 0;
            
            // Try to get the token usage from various fields
            if (obs.totalTokens !== undefined && obs.totalTokens > 0) {
                tokenUsage = obs.totalTokens;
            } else if (obs.usage && obs.usage.total > 0) {
                tokenUsage = obs.usage.total;
            } else if (obs.usageDetails && obs.usageDetails.total > 0) {
                tokenUsage = obs.usageDetails.total;
            } else if (obs.promptTokens !== undefined && obs.completionTokens !== undefined) {
                tokenUsage = obs.promptTokens + obs.completionTokens;
            } else if (obs.usage && obs.usage.input !== undefined && obs.usage.output !== undefined) {
                tokenUsage = obs.usage.input + obs.usage.output;
            }
            
            // Accumulate total tokens for validation
            totalObservationTokens += tokenUsage;
            
            // Get the tool name if available
            let toolName = obs.name || "Observation";
            if (!toolName || toolName === '') {
                if (obs.type) {
                    toolName = obs.type;
                } else if (obs.metadata && obs.metadata.langgraph_node) {
                    toolName = obs.metadata.langgraph_node;
                } else if (obs.input && obs.input.messages && Array.isArray(obs.input.messages)) {
                    for (const msg of obs.input.messages) {
                        if (msg.tool_calls && msg.tool_calls.length > 0) {
                            toolName = msg.tool_calls[0].name || "Tool Call";
                            break;
                        }
                    }
                }
            }
            
            return {
                toolCallNumber: index + 1,
                tokenUsage: tokenUsage,
                name: toolName,
                timestamp: obs.startTime || obs.timestamp,
                observationType: obs.type || "unknown",
                id: obs.id || `tc-${index+1}`
            };
        });
        
        // Sort observations by timestamp if available
        tokenObservations.sort((a, b) => {
            if (a.timestamp && b.timestamp) {
                return new Date(a.timestamp) - new Date(b.timestamp);
            }
            return a.toolCallNumber - b.toolCallNumber;
        });
        
        // Get expected token count from trace data if available
        let expectedTokens = null;
        if (fullTraceId) {
            const foundTrace = traceFiles.find(trace => trace.id === fullTraceId);
            if (foundTrace && foundTrace.totalTokens) {
                expectedTokens = foundTrace.totalTokens;
            }
        }
        
        res.json({ 
            traceId: searchTraceId, 
            toolCallCount: tokenObservations.length,
            toolCalls: tokenObservations,
            totalTraceTokens: totalObservationTokens,
            observationCount: allTraceObservations.length,
            expectTokens: expectedTokens
        });
    } catch (error) {
        console.error('Error fetching observation data:', error);
        res.status(500).json({ error: 'Failed to fetch observation data' });
    }
});

// Add a new API endpoint to check server-side last refresh timestamp
app.get('/api/last-refresh-time', async (req, res) => {
    try {
        // Try to read the last_refresh.txt file that our cron job creates
        const refreshFilePath = path.join(__dirname, 'dashboard_data', 'last_refresh.txt');
        
        // Check if the file exists
        try {
            const stats = await fs.stat(refreshFilePath);
            if (stats.isFile()) {
                // Read the timestamp from the file
                const timestampStr = await fs.readFile(refreshFilePath, 'utf8');
                const timestamp = timestampStr.trim();
                
                // Return the timestamp
                res.json({
                    success: true,
                    lastRefresh: timestamp,
                    // Also return file modification time as fallback
                    fileModified: stats.mtime.toISOString()
                });
                return;
            }
        } catch (statErr) {
            // File doesn't exist, fall back to checking file modification times
        }
        
        // If we don't have a specific timestamp file, check the modification time
        // of the most important data file
        const dataFilePath = path.join(__dirname, 'dashboard_data', 'all_traces.json');
        try {
            const stats = await fs.stat(dataFilePath);
            res.json({
                success: true,
                fileModified: stats.mtime.toISOString()
            });
        } catch (err) {
            throw new Error('Could not determine last refresh time');
        }
    } catch (error) {
        console.error('Error getting last refresh time:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to get last refresh time',
            error: error.message
        });
    }
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at Port: ${port}`);
});