const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

// --- API Endpoint: Predict Health Risk ---
app.post('/api/predict', (req, res) => {
    const userData = req.body;
    
    // Path to the python script
    const pythonScript = path.join(__dirname, 'ai_service', 'predict_runner.py');
    
    // Spawn Python process
    const pythonProcess = spawn('python', [pythonScript, JSON.stringify(userData)]);
    
    let resultData = '';
    let errorData = '';

    // Collect data from script
    pythonProcess.stdout.on('data', (data) => {
        resultData += data.toString();
    });

    // Collect errors
    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
    });

    // When script finishes
    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error(`Python Script Error: ${errorData}`);
            return res.status(500).json({ error: "Failed to process data", details: errorData });
        }

        try {
            const parsedResult = JSON.parse(resultData);
            res.json(parsedResult);
        } catch (e) {
            console.error("JSON Parse Error:", e);
            res.status(500).json({ error: "Invalid response from AI model" });
        }
    });
});

app.listen(PORT, () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
});