// server.js
const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const app = express();
const port = process.env.PORT || 3001;

// Middleware to parse JSON
app.use(express.json());

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Serve dashboard_data directory
app.use('/dashboard_data', express.static('dashboard_data'));

// Serve dashboard HTML
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
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
        const pythonProcess = spawn('python3', ['langfuse_data_fetcher.py']);
        
        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python script error: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                res.json({ success: true, message: 'Data refreshed successfully' });
            } else {
                res.status(500).json({ success: false, message: 'Failed to refresh data' });
            }
        });
    } catch (error) {
        console.error('Error refreshing data:', error);
        res.status(500).json({ success: false, message: 'Failed to refresh data' });
    }
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at Port: ${port}`);
});