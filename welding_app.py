#!/usr/bin/env python3
"""
MIG Welding Optimizer - Complete Working Application
Single-file solution with embedded HTML/CSS/JS
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import io
import webbrowser
import threading
import time

# ============================================
# GLOBAL MODEL STORAGE
# ============================================
MODELS = {
    'tensile_model': None,
    'penetration_model': None,
    'scaler': None,
    'trained': False,
    'training_data': None
}

# ============================================
# HTML TEMPLATE (Complete UI)
# ============================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIG Welding Optimizer & Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e94560;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            padding: 30px;
            margin-bottom: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            border: 2px solid #0f3460;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            color: #e94560;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .subtitle {
            color: #eaeaea;
            font-size: 1.1rem;
        }
        
        .badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            margin: 5px;
        }
        
        .badge-primary { background: #e94560; color: white; }
        .badge-warning { background: #f39c12; color: black; }
        .badge-success { background: #27ae60; color: white; }
        
        .panel {
            background: rgba(15, 52, 96, 0.3);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #0f3460;
            backdrop-filter: blur(10px);
        }
        
        .panel h2 {
            color: #e94560;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e94560;
            font-size: 1.4rem;
        }
        
        .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #eaeaea;
            font-weight: 500;
            font-size: 0.95rem;
        }
        
        input[type="number"],
        input[type="text"],
        select {
            width: 100%;
            padding: 12px 15px;
            background: rgba(0,0,0,0.3);
            border: 2px solid #0f3460;
            border-radius: 8px;
            color: #fff;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #e94560;
            box-shadow: 0 0 10px rgba(233, 69, 96, 0.3);
        }
        
        .help-text {
            display: block;
            margin-top: 6px;
            font-size: 0.85rem;
            color: #a0a0a0;
        }
        
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #e94560, #c0392b);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(233, 69, 96, 0.4);
        }
        
        .btn-secondary {
            background: #0f3460;
            color: #eaeaea;
            border: 2px solid #e94560;
        }
        
        .btn-secondary:hover {
            background: #e94560;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #27ae60, #229954);
            color: white;
        }
        
        .action-bar {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin: 30px 0;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            border-bottom: 2px solid #0f3460;
        }
        
        .tab-btn {
            padding: 14px 24px;
            background: transparent;
            border: none;
            color: #a0a0a0;
            font-size: 1rem;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab-btn.active {
            color: #e94560;
            border-bottom-color: #e94560;
            font-weight: 600;
        }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.4s; }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #0f3460;
        }
        
        th {
            background: rgba(233, 69, 96, 0.2);
            color: #e94560;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
        }
        
        tr:hover {
            background: rgba(15, 52, 96, 0.5);
        }
        
        .result-card {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            border: 2px solid;
            position: relative;
            overflow: hidden;
        }
        
        .result-card.tensile { border-color: #27ae60; }
        .result-card.penetration { border-color: #3498db; }
        .result-card.quality { border-color: #f39c12; }
        .result-card.warning { border-color: #e74c3c; }
        
        .result-value {
            font-size: 3.5rem;
            font-weight: 800;
            margin: 15px 0;
        }
        
        .result-card.tensile .result-value { color: #27ae60; }
        .result-card.penetration .result-value { color: #3498db; }
        .result-card.quality .result-value { color: #f39c12; }
        .result-card.warning .result-value { color: #e74c3c; }
        
        .result-label {
            color: #a0a0a0;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9rem;
        }
        
        .result-unit {
            color: #eaeaea;
            font-size: 1.1rem;
        }
        
        .status-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 15px;
        }
        
        .status-good { background: rgba(39, 174, 96, 0.2); color: #27ae60; }
        .status-warning { background: rgba(243, 156, 18, 0.2); color: #f39c12; }
        .status-bad { background: rgba(231, 76, 60, 0.2); color: #e74c3c; }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(26, 26, 46, 0.95);
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            z-index: 1000;
        }
        
        .spinner {
            width: 70px;
            height: 70px;
            border: 5px solid #0f3460;
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .log-container {
            background: rgba(0,0,0,0.4);
            border-radius: 10px;
            padding: 15px;
            max-height: 250px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
        
        .log-entry {
            padding: 6px 0;
            border-bottom: 1px solid #0f3460;
        }
        
        .log-entry.info { color: #3498db; }
        .log-entry.success { color: #27ae60; }
        .log-entry.error { color: #e74c3c; }
        .log-entry.warning { color: #f39c12; }
        
        .optimal-highlight {
            background: linear-gradient(135deg, rgba(233, 69, 96, 0.2), rgba(192, 57, 43, 0.2));
            border: 2px solid #e94560;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .metric-box {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #e94560;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #a0a0a0;
            margin-top: 8px;
        }
        
        .comparison-row {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            margin: 15px 0;
        }
        
        .comparison-box {
            flex: 1;
            text-align: center;
        }
        
        .comparison-box.predicted { border-right: 2px solid #0f3460; }
        
        .comparison-value {
            font-size: 2.2rem;
            font-weight: 700;
        }
        
        .comparison-box.predicted .comparison-value { color: #3498db; }
        .comparison-box.actual .comparison-value { color: #27ae60; }
        
        .diff-box {
            padding: 15px 25px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            text-align: center;
        }
        
        .diff-value {
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        .diff-value.good { color: #27ae60; }
        .diff-value.bad { color: #e74c3c; }
        
        .hidden { display: none !important; }
        
        .chart-container {
            height: 300px;
            margin-top: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 15px;
        }
        
        .preset-btn {
            padding: 10px 20px;
            background: rgba(15, 52, 96, 0.5);
            border: 2px solid #0f3460;
            border-radius: 8px;
            color: #eaeaea;
            cursor: pointer;
            transition: all 0.3s;
            margin: 5px;
        }
        
        .preset-btn:hover, .preset-btn.active {
            border-color: #e94560;
            color: #e94560;
        }
        
        .heat-input-bar {
            height: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            overflow: hidden;
            margin-top: 15px;
        }
        
        .heat-input-fill {
            height: 100%;
            transition: all 0.5s;
            border-radius: 10px;
        }
        
        @media (max-width: 768px) {
            h1 { font-size: 1.8rem; }
            .result-value { font-size: 2.5rem; }
            .comparison-row { flex-direction: column; }
            .comparison-box.predicted { border-right: none; border-bottom: 2px solid #0f3460; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <h1>🔥 MIG Welding Optimizer</h1>
            <p class="subtitle">316L Austenitic ↔ 430 Ferritic | 3mm Plate | Real-Time Prediction</p>
            <div>
                <span class="badge badge-primary">MIG Process</span>
                <span class="badge badge-warning">430 Grain Protection</span>
                <span class="badge badge-success">ML Powered</span>
            </div>
        </header>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('train')">1️⃣ Train Model</button>
            <button class="tab-btn" onclick="switchTab('predict')">2️⃣ Predict Quality</button>
            <button class="tab-btn" onclick="switchTab('optimize')">3️⃣ Optimize Parameters</button>
        </div>

        <!-- TAB 1: TRAIN MODEL -->
        <div id="train-tab" class="tab-content active">
            <div class="panel">
                <h2>📊 Load Training Data</h2>
                <p style="margin-bottom: 20px; color: #a0a0a0;">Train the machine learning models with your welding experiments.</p>
                
                <div class="grid-3">
                    <div>
                        <button class="btn btn-primary" onclick="loadDemoData()" style="width: 100%;">
                            🎲 Generate Demo Data (80 samples)
                        </button>
                        <p class="help-text">Creates synthetic data for testing</p>
                    </div>
                    <div>
                        <button class="btn btn-secondary" onclick="showManualEntry()" style="width: 100%;">
                            ✏️ Enter Data Manually
                        </button>
                        <p class="help-text">Input your experimental results</p>
                    </div>
                    <div>
                        <button class="btn btn-success" onclick="document.getElementById('file-upload').click()" style="width: 100%;">
                            📁 Upload CSV
                        </button>
                        <input type="file" id="file-upload" accept=".csv" style="display: none;" onchange="handleFileUpload(this)">
                        <p class="help-text">Load from CSV file</p>
                    </div>
                </div>

                <!-- Manual Entry Form -->
                <div id="manual-entry" class="hidden" style="margin-top: 25px;">
                    <h3 style="color: #e94560; margin-bottom: 15px;">Add Experiment</h3>
                    <div class="grid-4">
                        <div class="form-group">
                            <label>Current (A)</label>
                            <input type="number" id="m-current" placeholder="120" step="1">
                        </div>
                        <div class="form-group">
                            <label>Voltage (V)</label>
                            <input type="number" id="m-voltage" placeholder="22" step="0.5">
                        </div>
                        <div class="form-group">
                            <label>Speed (mm/min)</label>
                            <input type="number" id="m-speed" placeholder="100" step="5">
                        </div>
                        <div class="form-group">
                            <label>Filler Type</label>
                            <select id="m-filler">
                                <option value="ER309L">ER309L</option>
                                <option value="ER316L">ER316L</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Interpass Temp (°C)</label>
                            <input type="number" id="m-temp" placeholder="25" value="25">
                        </div>
                        <div class="form-group">
                            <label>Actual Tensile (MPa)</label>
                            <input type="number" id="m-tensile" placeholder="550">
                        </div>
                        <div class="form-group">
                            <label>Actual Penetration (mm)</label>
                            <input type="number" id="m-pen" placeholder="3.5" step="0.1">
                        </div>
                        <div class="form-group" style="display: flex; align-items: flex-end;">
                            <button class="btn btn-primary" onclick="addManualEntry()" style="width: 100%;">
                                ➕ Add to Dataset
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Data Preview -->
                <div id="data-preview-section" class="hidden" style="margin-top: 25px;">
                    <h3 style="color: #e94560; margin-bottom: 15px;">Dataset Preview (<span id="data-count">0</span> samples)</h3>
                    <div style="overflow-x: auto;">
                        <table id="preview-table">
                            <thead>
                                <tr>
                                    <th>Current</th>
                                    <th>Voltage</th>
                                    <th>Speed</th>
                                    <th>Filler</th>
                                    <th>Heat Input</th>
                                    <th>Tensile</th>
                                    <th>Penetration</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                    
                    <div class="action-bar">
                        <button class="btn btn-primary" onclick="trainModel()">
                            🧠 TRAIN MODEL
                        </button>
                        <button class="btn btn-secondary" onclick="downloadTemplate()">
                            📥 Download Template
                        </button>
                    </div>
                </div>

                <!-- Training Results -->
                <div id="training-results" class="hidden" style="margin-top: 25px;">
                    <div class="optimal-highlight">
                        <h3 style="color: #27ae60; margin-bottom: 15px;">✅ Model Trained Successfully</h3>
                        <div class="metric-grid">
                            <div class="metric-box">
                                <div class="metric-value" id="cv-tensile">--</div>
                                <div class="metric-label">Tensile R² Score</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value" id="cv-pen">--</div>
                                <div class="metric-label">Penetration R² Score</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value" id="train-samples">--</div>
                                <div class="metric-label">Training Samples</div>
                            </div>
                        </div>
                        <p style="margin-top: 15px; color: #a0a0a0;">
                            Cross-validation confirms model generalization. Ready for prediction and optimization.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: PREDICT -->
        <div id="predict-tab" class="tab-content">
            <div class="panel">
                <h2>🔮 Predict Welding Quality</h2>
                <p style="margin-bottom: 20px; color: #a0a0a0;">
                    Enter process parameters to get real-time predictions of tensile strength and penetration.
                </p>

                <div class="grid-3">
                    <div class="form-group">
                        <label>Current (A)</label>
                        <input type="number" id="p-current" value="120" min="70" max="180">
                        <div style="margin-top: 8px;">
                            <button class="preset-btn" onclick="setPredict('p-current', 90)">90A</button>
                            <button class="preset-btn" onclick="setPredict('p-current', 120)">120A</button>
                            <button class="preset-btn" onclick="setPredict('p-current', 150)">150A</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Voltage (V)</label>
                        <input type="number" id="p-voltage" value="22" min="18" max="30" step="0.5">
                    </div>
                    <div class="form-group">
                        <label>Travel Speed (mm/min)</label>
                        <input type="number" id="p-speed" value="110" min="50" max="300">
                    </div>
                    <div class="form-group">
                        <label>Filler Type</label>
                        <select id="p-filler">
                            <option value="ER309L">ER309L (Recommended)</option>
                            <option value="ER316L">ER316L</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Interpass Temperature (°C)</label>
                        <input type="number" id="p-temp" value="25" min="20" max="200">
                    </div>
                    <div class="form-group">
                        <label>MIG Efficiency</label>
                        <input type="number" id="p-efficiency" value="0.6" readonly style="background: rgba(100,100,100,0.2);">
                    </div>
                </div>

                <!-- Real-time Heat Input -->
                <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                        <div>
                            <h4>Calculated Heat Input</h4>
                            <p style="color: #a0a0a0; font-size: 0.9rem;">HI = (V × I × 0.6) / Speed</p>
                        </div>
                        <div style="text-align: center;">
                            <div id="rt-hi-value" style="font-size: 2.5rem; font-weight: 700; color: #3498db;">--</div>
                            <div style="color: #a0a0a0;">kJ/mm</div>
                        </div>
                        <div id="rt-hi-status" class="status-badge" style="margin: 0;">Calculate</div>
                    </div>
                    <div class="heat-input-bar">
                        <div id="rt-hi-bar" class="heat-input-fill" style="width: 0%; background: #3498db;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.85rem; color: #a0a0a0;">
                        <span>0</span>
                        <span style="color: #27ae60;">Safe (&lt;1.2)</span>
                        <span style="color: #e74c3c;">Danger (&gt;1.2)</span>
                        <span>2.0</span>
                    </div>
                </div>

                <!-- Actual Values (Optional) -->
                <div style="background: rgba(39, 174, 96, 0.1); border: 1px dashed #27ae60; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4 style="color: #27ae60; margin-bottom: 15px;">🎯 Compare with Actual (Optional)</h4>
                    <div class="grid-2">
                        <div class="form-group">
                            <label>Actual Tensile Strength (MPa)</label>
                            <input type="number" id="actual-tensile" placeholder="Enter measured value">
                        </div>
                        <div class="form-group">
                            <label>Actual Penetration (mm)</label>
                            <input type="number" id="actual-pen" placeholder="Enter measured value" step="0.1">
                        </div>
                    </div>
                </div>

                <div class="action-bar">
                    <button class="btn btn-primary" onclick="runPrediction()" style="font-size: 1.2rem; padding: 18px 40px;">
                        🔮 RUN PREDICTION
                    </button>
                </div>
            </div>

            <!-- Prediction Results -->
            <div id="prediction-results" class="hidden">
                <div class="grid-3">
                    <div class="result-card tensile">
                        <div class="result-label">Predicted Tensile Strength</div>
                        <div class="result-value" id="res-tensile">--</div>
                        <div class="result-unit">MPa</div>
                        <div class="status-badge" id="res-tensile-status">--</div>
                    </div>
                    <div class="result-card penetration">
                        <div class="result-label">Predicted Penetration</div>
                        <div class="result-value" id="res-pen">--</div>
                        <div class="result-unit">mm</div>
                        <div class="status-badge" id="res-pen-status">--</div>
                    </div>
                    <div class="result-card quality">
                        <div class="result-label">Quality Score</div>
                        <div class="result-value" id="res-quality">--</div>
                        <div class="result-unit">/ 100</div>
                        <div class="status-badge" id="res-quality-status">--</div>
                    </div>
                </div>

                <!-- Comparison with Actual -->
                <div id="actual-comparison" class="hidden panel" style="margin-top: 25px;">
                    <h3>🔍 Prediction vs Actual Comparison</h3>
                    
                    <div class="comparison-row">
                        <div class="comparison-box predicted" style="flex: 1;">
                            <div style="color: #a0a0a0; margin-bottom: 8px;">Predicted Tensile</div>
                            <div class="comparison-value" id="comp-pred-tensile" style="color: #3498db;">--</div>
                            <div style="color: #a0a0a0;">MPa</div>
                        </div>
                        <div class="comparison-box actual" style="flex: 1;">
                            <div style="color: #a0a0a0; margin-bottom: 8px;">Actual Tensile</div>
                            <div class="comparison-value" id="comp-actual-tensile" style="color: #27ae60;">--</div>
                            <div style="color: #a0a0a0;">MPa</div>
                        </div>
                        <div class="diff-box">
                            <div style="color: #a0a0a0; margin-bottom: 8px;">Error</div>
                            <div class="diff-value" id="diff-tensile">--</div>
                            <div style="color: #a0a0a0;" id="pct-tensile">--</div>
                        </div>
                    </div>

                    <div class="comparison-row">
                        <div class="comparison-box predicted" style="flex: 1;">
                            <div style="color: #a0a0a0; margin-bottom: 8px;">Predicted Penetration</div>
                            <div class="comparison-value" id="comp-pred-pen" style="color: #3498db;">--</div>
                            <div style="color: #a0a0a0;">mm</div>
                        </div>
                        <div class="comparison-box actual" style="flex: 1;">
                            <div style="color: #a0a0a0; margin-bottom: 8px;">Actual Penetration</div>
                            <div class="comparison-value" id="comp-actual-pen" style="color: #27ae60;">--</div>
                            <div style="color: #a0a0a0;">mm</div>
                        </div>
                        <div class="diff-box">
                            <div style="color: #a0a0a0; margin-bottom: 8px;">Error</div>
                            <div class="diff-value" id="diff-pen">--</div>
                            <div style="color: #a0a0a0;" id="pct-pen">--</div>
                        </div>
                    </div>
                </div>

                <!-- Safety Assessment -->
                <div class="panel" style="margin-top: 25px; background: rgba(0,0,0,0.4);">
                    <h3>🛡️ Safety Assessment for 430 Ferritic Steel</h3>
                    <div id="safety-content"></div>
                </div>
            </div>
        </div>

        <!-- TAB 3: OPTIMIZE -->
        <div id="optimize-tab" class="tab-content">
            <div class="panel">
                <h2>⚙️ Optimize Welding Parameters</h2>
                <p style="margin-bottom: 20px; color: #a0a0a0;">
                    Find the best current and speed combination that maximizes tensile strength while protecting the 430 ferritic side.
                </p>

                <div class="grid-3">
                    <div class="form-group">
                        <label>Max Heat Input Limit (kJ/mm)</label>
                        <input type="number" id="opt-max-hi" value="1.2" min="0.5" max="2.0" step="0.1">
                        <span class="help-text">Critical: 430 grain growth threshold</span>
                    </div>
                    <div class="form-group">
                        <label>Plate Thickness (mm)</label>
                        <input type="number" id="opt-thickness" value="3" min="1" max="10">
                        <span class="help-text">Determines penetration target</span>
                    </div>
                    <div class="form-group">
                        <label>Filler Preference</label>
                        <select id="opt-filler">
                            <option value="both">Both (Find Best)</option>
                            <option value="ER309L">ER309L Only</option>
                            <option value="ER316L">ER316L Only</option>
                        </select>
                    </div>
                </div>

                <div class="grid-2">
                    <div>
                        <h4 style="color: #e94560; margin-bottom: 15px;">Current Range (A)</h4>
                        <div class="grid-2">
                            <div class="form-group">
                                <label>Min</label>
                                <input type="number" id="opt-curr-min" value="70" min="50" max="200">
                            </div>
                            <div class="form-group">
                                <label>Max</label>
                                <input type="number" id="opt-curr-max" value="150" min="50" max="200">
                            </div>
                        </div>
                    </div>
                    <div>
                        <h4 style="color: #e94560; margin-bottom: 15px;">Speed Range (mm/min)</h4>
                        <div class="grid-2">
                            <div class="form-group">
                                <label>Min</label>
                                <input type="number" id="opt-spd-min" value="80" min="50" max="300">
                            </div>
                            <div class="form-group">
                                <label>Max</label>
                                <input type="number" id="opt-spd-max" value="150" min="50" max="300">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="form-group" style="margin-top: 20px;">
                    <label>Search Step Size</label>
                    <input type="number" id="opt-step" value="2" min="1" max="10">
                    <span class="help-text">Smaller = more precise but slower (1-10)</span>
                </div>

                <div class="action-bar">
                    <button class="btn btn-primary" onclick="runOptimization()" style="font-size: 1.2rem; padding: 18px 40px;">
                        🚀 RUN OPTIMIZATION
                    </button>
                </div>
            </div>

            <!-- Optimization Results -->
            <div id="optimization-results" class="hidden">
                <div class="optimal-highlight">
                    <h2 style="color: #e94560; text-align: center; margin-bottom: 25px;">🏆 OPTIMAL PARAMETERS FOUND</h2>
                    
                    <div class="grid-3">
                        <div class="metric-box" style="border: 2px solid #e94560;">
                            <div class="metric-value" id="opt-res-current" style="font-size: 2.5rem;">--</div>
                            <div class="metric-label">Current (A)</div>
                        </div>
                        <div class="metric-box" style="border: 2px solid #e94560;">
                            <div class="metric-value" id="opt-res-speed" style="font-size: 2.5rem;">--</div>
                            <div class="metric-label">Speed (mm/min)</div>
                        </div>
                        <div class="metric-box" style="border: 2px solid #e94560;">
                            <div class="metric-value" id="opt-res-filler" style="font-size: 2rem;">--</div>
                            <div class="metric-label">Filler Type</div>
                        </div>
                    </div>

                    <div class="metric-grid" style="margin-top: 25px;">
                        <div class="metric-box">
                            <div class="metric-value" id="opt-res-hi" style="color: #3498db;">--</div>
                            <div class="metric-label">Heat Input (kJ/mm)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="opt-res-pen" style="color: #27ae60;">--</div>
                            <div class="metric-label">Penetration (mm)</div>
                        </div>
                        <div class="metric-box" style="background: rgba(233, 69, 96, 0.2);">
                            <div class="metric-value" id="opt-res-tensile" style="color: #e94560; font-size: 2.5rem;">--</div>
                            <div class="metric-label">Tensile Strength (MPa)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="opt-res-safety">--</div>
                            <div class="metric-label">Safety Margin</div>
                        </div>
                    </div>

                    <div id="opt-constraints" style="margin-top: 20px;"></div>
                </div>

                <!-- Statistics -->
                <div class="panel" style="margin-top: 25px;">
                    <h3>📊 Optimization Statistics</h3>
                    <div class="grid-4">
                        <div class="metric-box">
                            <div class="metric-value" id="stat-total">--</div>
                            <div class="metric-label">Total Scanned</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="stat-valid">--</div>
                            <div class="metric-label">Valid Solutions</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="stat-rejected">--</div>
                            <div class="metric-label">Rejected by Constraints</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="stat-rate">--</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                    </div>
                </div>

                <!-- Top 5 Alternatives -->
                <div class="panel" style="margin-top: 25px;">
                    <h3>📋 Top 5 Alternative Solutions</h3>
                    <table id="alternatives-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Current (A)</th>
                                <th>Speed (mm/min)</th>
                                <th>Filler</th>
                                <th>Heat Input</th>
                                <th>Penetration</th>
                                <th>Tensile (MPa)</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- System Log -->
        <div class="panel">
            <h2>📝 System Log</h2>
            <div class="log-container" id="system-log">
                <div class="log-entry info">System initialized. Ready to train model.</div>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loading-overlay">
        <div class="spinner"></div>
        <p style="margin-top: 20px; color: #eaeaea; font-size: 1.1rem;">Processing...</p>
    </div>

    <script>
        // Global state
        let trainingData = [];
        let lastPrediction = null;
        let lastOptimization = null;

        // ==========================================
        // UTILITY FUNCTIONS
        // ==========================================
        function log(message, type = 'info') {
            const container = document.getElementById('system-log');
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            container.insertBefore(entry, container.firstChild);
        }

        function showLoading(show = true) {
            document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
        }

        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            
            // Show selected
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }

        // ==========================================
        // TAB 1: TRAIN MODEL
        // ==========================================
        function loadDemoData() {
            showLoading(true);
            log('Generating demo data...', 'info');
            
            fetch('/api/generate-demo', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({voltage: 22, efficiency: 0.6})
            })
            .then(r => r.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    trainingData = data.data;
                    displayDataPreview();
                    log(`Generated ${data.count} demo samples`, 'success');
                }
            })
            .catch(err => {
                showLoading(false);
                log('Error: ' + err.message, 'error');
            });
        }

        function showManualEntry() {
            document.getElementById('manual-entry').classList.remove('hidden');
        }

        function addManualEntry() {
            const entry = {
                Current_A: parseFloat(document.getElementById('m-current').value),
                Voltage_V: parseFloat(document.getElementById('m-voltage').value),
                Travel_Speed_mm_min: parseFloat(document.getElementById('m-speed').value),
                Filler_Type: document.getElementById('m-filler').value,
                Interpass_Temp_C: parseFloat(document.getElementById('m-temp').value) || 25,
                Heat_Input_kJ_mm: 0, // Will be calculated
                Tensile_Strength_MPa: parseFloat(document.getElementById('m-tensile').value),
                Penetration_Depth_mm: parseFloat(document.getElementById('m-pen').value)
            };
            
            // Calculate heat input
            entry.Heat_Input_kJ_mm = (entry.Voltage_V * entry.Current_A * 0.6) / entry.Travel_Speed_mm_min;
            
            if (!entry.Current_A || !entry.Tensile_Strength_MPa) {
                alert('Please fill in all required fields');
                return;
            }
            
            trainingData.push(entry);
            displayDataPreview();
            log(`Added manual entry. Total: ${trainingData.length} samples`, 'success');
            
            // Clear form
            ['m-current', 'm-voltage', 'm-speed', 'm-tensile', 'm-pen'].forEach(id => {
                document.getElementById(id).value = '';
            });
        }

        function handleFileUpload(input) {
            const file = input.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const text = e.target.result;
                const rows = text.split('\\n').filter(r => r.trim());
                const headers = rows[0].split(',').map(h => h.trim());
                
                // Parse CSV
                trainingData = [];
                for (let i = 1; i < rows.length; i++) {
                    const cols = rows[i].split(',');
                    if (cols.length >= 7) {
                        trainingData.push({
                            Current_A: parseFloat(cols[0]),
                            Voltage_V: parseFloat(cols[1]),
                            Travel_Speed_mm_min: parseFloat(cols[2]),
                            Filler_Type: cols[3].trim(),
                            Interpass_Temp_C: parseFloat(cols[4]),
                            Heat_Input_kJ_mm: parseFloat(cols[5]),
                            Tensile_Strength_MPa: parseFloat(cols[6]),
                            Penetration_Depth_mm: parseFloat(cols[7])
                        });
                    }
                }
                
                displayDataPreview();
                log(`Loaded ${trainingData.length} samples from CSV`, 'success');
            };
            reader.readAsText(file);
        }

        function displayDataPreview() {
            document.getElementById('data-preview-section').classList.remove('hidden');
            document.getElementById('data-count').textContent = trainingData.length;
            
            const tbody = document.querySelector('#preview-table tbody');
            tbody.innerHTML = '';
            
            trainingData.slice(0, 10).forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.Current_A}</td>
                    <td>${row.Voltage_V}</td>
                    <td>${row.Travel_Speed_mm_min}</td>
                    <td>${row.Filler_Type}</td>
                    <td>${row.Heat_Input_kJ_mm.toFixed(3)}</td>
                    <td>${row.Tensile_Strength_MPa}</td>
                    <td>${row.Penetration_Depth_mm}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function trainModel() {
            if (trainingData.length < 5) {
                alert('Need at least 5 samples to train');
                return;
            }
            
            showLoading(true);
            log('Training Random Forest models...', 'info');
            
            fetch('/api/train-model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({data: trainingData})
            })
            .then(r => r.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    document.getElementById('training-results').classList.remove('hidden');
                    document.getElementById('cv-tensile').textContent = data.cv_scores.tensile_mean.toFixed(3);
                    document.getElementById('cv-pen').textContent = data.cv_scores.pen_mean.toFixed(3);
                    document.getElementById('train-samples').textContent = data.n_samples;
                    log(`Model trained! Tensile R²: ${data.cv_scores.tensile_mean.toFixed(3)}, Penetration R²: ${data.cv_scores.pen_mean.toFixed(3)}`, 'success');
                } else {
                    log('Training failed: ' + data.error, 'error');
                }
            })
            .catch(err => {
                showLoading(false);
                log('Error: ' + err.message, 'error');
            });
        }

        function downloadTemplate() {
            const csv = `Current_A,Voltage_V,Travel_Speed_mm_min,Filler_Type,Interpass_Temp_C,Heat_Input_kJ_mm,Tensile_Strength_MPa,Penetration_Depth_mm
90,22,100,ER309L,25,1.19,520,2.8
110,23,110,ER316L,30,1.38,480,3.5
120,22,100,ER309L,25,1.58,580,4.2
130,24,90,ER309L,40,2.08,510,5.1
100,22,120,ER316L,25,1.10,495,3.0`;
            
            const blob = new Blob([csv], {type: 'text/csv'});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'MIG_welding_template.csv';
            a.click();
        }

        // ==========================================
        // TAB 2: PREDICTION (with real-time HI)
        // ==========================================
        function updateRealtimeHI() {
            const current = parseFloat(document.getElementById('p-current').value) || 0;
            const voltage = parseFloat(document.getElementById('p-voltage').value) || 0;
            const speed = parseFloat(document.getElementById('p-speed').value) || 1;
            const efficiency = 0.6;
            
            const hi = (voltage * current * efficiency) / speed;
            
            document.getElementById('rt-hi-value').textContent = hi.toFixed(3);
            
            const status = document.getElementById('rt-hi-status');
            const bar = document.getElementById('rt-hi-bar');
            const pct = Math.min((hi / 2.0) * 100, 100);
            
            bar.style.width = pct + '%';
            
            if (hi < 0.7) {
                status.textContent = 'Low Heat - Risk of cold lap';
                status.className = 'status-badge status-warning';
                bar.style.background = '#f39c12';
            } else if (hi <= 1.2) {
                status.textContent = '✓ Optimal Zone';
                status.className = 'status-badge status-good';
                bar.style.background = '#27ae60';
            } else {
                status.textContent = '⚠ High Heat - 430 Damage Risk!';
                status.className = 'status-badge status-bad';
                bar.style.background = '#e74c3c';
            }
        }

        // Attach listeners
        ['p-current', 'p-voltage', 'p-speed'].forEach(id => {
            document.getElementById(id).addEventListener('input', updateRealtimeHI);
        });

        function setPredict(id, val) {
            document.getElementById(id).value = val;
            updateRealtimeHI();
        }

        function runPrediction() {
            const params = {
                current: parseFloat(document.getElementById('p-current').value),
                voltage: parseFloat(document.getElementById('p-voltage').value),
                speed: parseFloat(document.getElementById('p-speed').value),
                filler: document.getElementById('p-filler').value,
                interpass: parseFloat(document.getElementById('p-temp').value),
                efficiency: 0.6
            };
            
            if (!params.current || !params.voltage || !params.speed) {
                alert('Please fill in all parameters');
                return;
            }
            
            showLoading(true);
            log('Running prediction...', 'info');
            
            fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            })
            .then(r => r.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    displayPredictionResults(data);
                    lastPrediction = data;
                    log(`Prediction complete: ${data.prediction.tensile.toFixed(1)} MPa, ${data.prediction.penetration.toFixed(2)} mm penetration`, 'success');
                } else {
                    log('Prediction failed: ' + data.error, 'error');
                    if (data.error.includes('No trained model')) {
                        alert('Please train a model first on the "Train Model" tab');
                    }
                }
            })
            .catch(err => {
                showLoading(false);
                log('Error: ' + err.message, 'error');
            });
        }

        function displayPredictionResults(data) {
            document.getElementById('prediction-results').classList.remove('hidden');
            
            const tensile = data.prediction.tensile;
            const pen = data.prediction.penetration;
            const hi = data.heat_input;
            
            // Tensile
            document.getElementById('res-tensile').textContent = tensile.toFixed(1);
            const tStatus = document.getElementById('res-tensile-status');
            if (tensile >= 580) {
                tStatus.textContent = '✓ High Strength';
                tStatus.className = 'status-badge status-good';
            } else if (tensile >= 500) {
                tStatus.textContent = '✓ Adequate';
                tStatus.className = 'status-badge status-good';
            } else {
                tStatus.textContent = '⚠ Low - Check parameters';
                tStatus.className = 'status-badge status-warning';
            }
            
            // Penetration
            document.getElementById('res-pen').textContent = pen.toFixed(2);
            const pStatus = document.getElementById('res-pen-status');
            if (pen >= 3.0) {
                pStatus.textContent = '✓ Full Penetration (>3mm)';
                pStatus.className = 'status-badge status-good';
            } else if (pen >= 2.4) {
                pStatus.textContent = '⚠ Partial (80%+)';
                pStatus.className = 'status-badge status-warning';
            } else {
                pStatus.textContent = '✗ Insufficient Fusion';
                pStatus.className = 'status-badge status-bad';
            }
            
            // Quality Score
            let qScore = 0;
            qScore += Math.min((tensile / 650) * 50, 50);
            qScore += (pen >= 3.0 && pen <= 5.0) ? 30 : Math.max(0, 30 - Math.abs(pen - 3.5) * 10);
            qScore += (hi <= 1.2 && hi >= 0.7) ? 20 : 10;
            
            document.getElementById('res-quality').textContent = Math.round(qScore);
            const qStatus = document.getElementById('res-quality-status');
            if (qScore >= 80) {
                qStatus.textContent = 'Excellent';
                qStatus.className = 'status-badge status-good';
            } else if (qScore >= 60) {
                qStatus.textContent = 'Good';
                qStatus.className = 'status-badge status-good';
            } else if (qScore >= 40) {
                qStatus.textContent = 'Acceptable';
                qStatus.className = 'status-badge status-warning';
            } else {
                qStatus.textContent = 'Poor - Review';
                qStatus.className = 'status-badge status-bad';
            }
            
            // Comparison with actual
            const actualT = parseFloat(document.getElementById('actual-tensile').value);
            const actualP = parseFloat(document.getElementById('actual-pen').value);
            
            if (actualT || actualP) {
                document.getElementById('actual-comparison').classList.remove('hidden');
                
                if (actualT) {
                    document.getElementById('comp-pred-tensile').textContent = tensile.toFixed(1);
                    document.getElementById('comp-actual-tensile').textContent = actualT.toFixed(1);
                    const diff = tensile - actualT;
                    const pct = (Math.abs(diff) / actualT) * 100;
                    document.getElementById('diff-tensile').textContent = (diff >= 0 ? '+' : '') + diff.toFixed(1);
                    document.getElementById('diff-tensile').className = 'diff-value ' + (pct < 10 ? 'good' : 'bad');
                    document.getElementById('pct-tensile').textContent = pct.toFixed(1) + '% error';
                }
                
                if (actualP) {
                    document.getElementById('comp-pred-pen').textContent = pen.toFixed(2);
                    document.getElementById('comp-actual-pen').textContent = actualP.toFixed(2);
                    const diff = pen - actualP;
                    const pct = (Math.abs(diff) / actualP) * 100;
                    document.getElementById('diff-pen').textContent = (diff >= 0 ? '+' : '') + diff.toFixed(2);
                    document.getElementById('diff-pen').className = 'diff-value ' + (pct < 15 ? 'good' : 'bad');
                    document.getElementById('pct-pen').textContent = pct.toFixed(1) + '% error';
                }
            }
            
            // Safety assessment
            let safetyHTML = '';
            if (hi <= 1.2) {
                safetyHTML += `<div class="status-badge status-good" style="margin: 10px 0;">
                    ✓ Heat Input Safe: ${hi.toFixed(3)} kJ/mm ≤ 1.2 limit. 430 ferritic HAZ protected from grain coarsening.
                </div>`;
            } else {
                safetyHTML += `<div class="status-badge status-bad" style="margin: 10px 0;">
                    ⚠ DANGER: Heat Input ${hi.toFixed(3)} kJ/mm exceeds 1.2 limit. Risk of brittle 430 HAZ!
                </div>`;
            }
            
            if (pen >= 3.0) {
                safetyHTML += `<div class="status-badge status-good" style="margin: 10px 0;">
                    ✓ Penetration Adequate: ${pen.toFixed(2)}mm ≥ 3mm required. Full fusion expected.
                </div>`;
            } else {
                safetyHTML += `<div class="status-badge status-warning" style="margin: 10px 0;">
                    ⚠ Penetration Risk: ${pen.toFixed(2)}mm < 3mm required. Incomplete fusion possible.
                </div>`;
            }
            
            document.getElementById('safety-content').innerHTML = safetyHTML;
        }

        // ==========================================
        // TAB 3: OPTIMIZATION
        // ==========================================
        function runOptimization() {
            const config = {
                max_heat_input: parseFloat(document.getElementById('opt-max-hi').value),
                plate_thickness: parseFloat(document.getElementById('opt-thickness').value),
                filler_preference: document.getElementById('opt-filler').value,
                current_min: parseInt(document.getElementById('opt-curr-min').value),
                current_max: parseInt(document.getElementById('opt-curr-max').value),
                speed_min: parseInt(document.getElementById('opt-spd-min').value),
                speed_max: parseInt(document.getElementById('opt-spd-max').value),
                step: parseInt(document.getElementById('opt-step').value),
                voltage: 22,
                efficiency: 0.6,
                interpass_temp: 25
            };
            
            showLoading(true);
            log('Starting optimization search...', 'info');
            
            fetch('/api/optimize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            })
            .then(r => r.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    displayOptimizationResults(data);
                    lastOptimization = data;
                    log(`Optimization complete! Best: ${data.optimal.Pred_Tensile_MPa.toFixed(1)} MPa`, 'success');
                } else {
                    log('Optimization failed: ' + data.error, 'error');
                    if (data.error.includes('No trained model')) {
                        alert('Please train a model first');
                    }
                }
            })
            .catch(err => {
                showLoading(false);
                log('Error: ' + err.message, 'error');
            });
        }

        function displayOptimizationResults(data) {
            document.getElementById('optimization-results').classList.remove('hidden');
            
            const opt = data.optimal;
            
            // Main results
            document.getElementById('opt-res-current').textContent = opt.Current_A;
            document.getElementById('opt-res-speed').textContent = opt.Speed_mm_min;
            document.getElementById('opt-res-filler').textContent = opt.Filler_Type;
            document.getElementById('opt-res-hi').textContent = opt.Heat_Input_kJ_mm.toFixed(3);
            document.getElementById('opt-res-pen').textContent = opt.Pred_Penetration_mm.toFixed(2);
            document.getElementById('opt-res-tensile').textContent = opt.Pred_Tensile_MPa.toFixed(1);
            document.getElementById('opt-res-safety').textContent = opt.Safety_Margin_HI.toFixed(3);
            
            // Constraints
            const cDiv = document.getElementById('opt-constraints');
            let cHTML = '<div style="display: flex; gap: 15px; flex-wrap: wrap;">';
            cHTML += `<div class="status-badge status-good">✓ Heat Input: ${opt.Heat_Input_kJ_mm.toFixed(3)} < 1.2 limit</div>`;
            cHTML += `<div class="status-badge status-good">✓ Penetration: ${opt.Pred_Penetration_mm.toFixed(2)}mm > 3mm target</div>`;
            cHTML += '</div>';
            cDiv.innerHTML = cHTML;
            
            // Statistics
            const stats = data.statistics;
            document.getElementById('stat-total').textContent = stats.total_scanned.toLocaleString();
            document.getElementById('stat-valid').textContent = stats.valid_count.toLocaleString();
            document.getElementById('stat-rejected').textContent = (stats.rejected_hi + stats.rejected_pen).toLocaleString();
            document.getElementById('stat-rate').textContent = ((stats.valid_count / stats.total_scanned) * 100).toFixed(1) + '%';
            
            // Alternatives table
            const tbody = document.querySelector('#alternatives-table tbody');
            tbody.innerHTML = data.top5.map((sol, i) => `
                <tr style="${i === 0 ? 'background: rgba(233, 69, 96, 0.2);' : ''}">
                    <td>${i + 1} ${i === 0 ? '★' : ''}</td>
                    <td>${sol.Current_A}</td>
                    <td>${sol.Speed_mm_min}</td>
                    <td>${sol.Filler_Type}</td>
                    <td>${sol.Heat_Input_kJ_mm.toFixed(3)}</td>
                    <td>${sol.Pred_Penetration_mm.toFixed(2)}</td>
                    <td><strong>${sol.Pred_Tensile_MPa.toFixed(1)}</strong></td>
                </tr>
            `).join('');
            
            // Scroll to results
            document.getElementById('optimization-results').scrollIntoView({behavior: 'smooth'});
        }

        // Initialize
        updateRealtimeHI();
    </script>
</body>
</html>
'''

# ============================================
# HTTP REQUEST HANDLER
# ============================================
class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.send_html(HTML_TEMPLATE)
        elif self.path == '/api/check-model':
            self.send_json({
                'trained': MODELS['trained'],
                'n_samples': len(MODELS['training_data']) if MODELS['training_data'] else 0
            })
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
        except:
            data = {}
        
        if self.path == '/api/generate-demo':
            self.handle_generate_demo(data)
        elif self.path == '/api/train-model':
            self.handle_train_model(data)
        elif self.path == '/api/predict':
            self.handle_predict(data)
        elif self.path == '/api/optimize':
            self.handle_optimize(data)
        else:
            self.send_error(404)
    
    def send_html(self, content):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def handle_generate_demo(self, data):
        """Generate synthetic welding data."""
        n_samples = 80
        voltage = data.get('voltage', 22)
        efficiency = data.get('efficiency', 0.6)
        
        np.random.seed(42)
        
        samples = []
        for _ in range(n_samples):
            curr = np.random.uniform(80, 140)
            volt = np.random.uniform(20, 26)
            speed = np.random.uniform(80, 140)
            filler = 'ER309L' if np.random.random() < 0.6 else 'ER316L'
            temp = np.random.uniform(20, 80)
            
            hi = (volt * curr * efficiency) / speed
            
            # Physics-based targets
            optimal_hi = 0.9
            tensile = 580 - 80 * ((hi - optimal_hi) ** 2)
            tensile += 20 if filler == 'ER309L' else 0
            tensile -= 0.15 * temp
            tensile += np.random.normal(0, 15)
            tensile = np.clip(tensile, 400, 650)
            
            pen = 1.0 + 2.0 * hi + np.random.normal(0, 0.2)
            pen = np.clip(pen, 1.5, 5.0)
            
            samples.append({
                'Current_A': round(curr, 1),
                'Voltage_V': round(volt, 1),
                'Travel_Speed_mm_min': round(speed, 1),
                'Filler_Type': filler,
                'Interpass_Temp_C': round(temp, 1),
                'Heat_Input_kJ_mm': round(hi, 3),
                'Tensile_Strength_MPa': round(tensile, 1),
                'Penetration_Depth_mm': round(pen, 2)
            })
        
        self.send_json({'success': True, 'data': samples, 'count': len(samples)})
    
    def handle_train_model(self, data):
        """Train Random Forest models."""
        training_data = data.get('data', [])
        
        if len(training_data) < 5:
            self.send_json({'success': False, 'error': 'Need at least 5 samples'})
            return
        
        try:
            # Prepare data
            df = pd.DataFrame(training_data)
            df['Filler_Code'] = (df['Filler_Type'] == 'ER316L').astype(int)
            
            features = ['Current_A', 'Voltage_V', 'Travel_Speed_mm_min', 
                       'Filler_Code', 'Interpass_Temp_C', 'Heat_Input_kJ_mm']
            X = df[features]
            y_tensile = df['Tensile_Strength_MPa']
            y_pen = df['Penetration_Depth_mm']
            
            # Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Models
            tensile_model = RandomForestRegressor(n_estimators=100, random_state=42)
            pen_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            tensile_model.fit(X_scaled, y_tensile)
            pen_model.fit(X_scaled, y_pen)
            
            # Cross Validation
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_tensile = cross_val_score(tensile_model, X_scaled, y_tensile, cv=cv, scoring='r2')
            cv_pen = cross_val_score(pen_model, X_scaled, y_pen, cv=cv, scoring='r2')
            
            # Save Global State
            MODELS['tensile_model'] = tensile_model
            MODELS['penetration_model'] = pen_model
            MODELS['scaler'] = scaler
            MODELS['trained'] = True
            MODELS['training_data'] = training_data
            
            self.send_json({
                'success': True,
                'n_samples': len(df),
                'cv_scores': {
                    'tensile_mean': float(cv_tensile.mean()),
                    'pen_mean': float(cv_pen.mean())
                }
            })
        except Exception as e:
            self.send_json({'success': False, 'error': str(e)})

    def handle_predict(self, data):
        """Predict outcomes based on parameters."""
        if not MODELS['trained']:
            self.send_json({'success': False, 'error': 'No trained model found'})
            return

        try:
            # Extract inputs
            current = float(data.get('current', 120))
            voltage = float(data.get('voltage', 22))
            speed = float(data.get('speed', 110))
            filler = data.get('filler', 'ER309L')
            interpass = float(data.get('interpass', 25))
            efficiency = float(data.get('efficiency', 0.6))
            
            # Calculate Heat Input
            hi = (voltage * current * efficiency) / speed
            
            # Prepare feature vector
            filler_code = 1 if filler == 'ER316L' else 0
            features = np.array([[current, voltage, speed, filler_code, interpass, hi]])
            
            # Scale
            features_scaled = MODELS['scaler'].transform(features)
            
            # Predict
            tensile = float(MODELS['tensile_model'].predict(features_scaled)[0])
            pen = float(MODELS['penetration_model'].predict(features_scaled)[0])
            
            self.send_json({
                'success': True,
                'prediction': {
                    'tensile': tensile,
                    'penetration': pen
                },
                'heat_input': hi
            })
        except Exception as e:
            self.send_json({'success': False, 'error': str(e)})

    def handle_optimize(self, data):
        """Find optimal parameters."""
        if not MODELS['trained']:
            self.send_json({'success': False, 'error': 'No trained model found'})
            return

        try:
            # Constraints & Ranges
            max_hi = float(data.get('max_heat_input', 1.2))
            target_pen = float(data.get('plate_thickness', 3.0)) # Usually target >= thickness
            filler_pref = data.get('filler_preference', 'both')
            
            curr_range = range(data.get('current_min', 80), data.get('current_max', 150) + 1, data.get('step', 5))
            speed_range = range(data.get('speed_min', 80), data.get('speed_max', 200) + 1, data.get('step', 5))
            
            fillers = []
            if filler_pref in ['both', 'ER309L']: fillers.append('ER309L')
            if filler_pref in ['both', 'ER316L']: fillers.append('ER316L')
            
            voltage = data.get('voltage', 22)
            interpass = data.get('interpass_temp', 25)
            efficiency = data.get('efficiency', 0.6)
            
            candidates = []
            total_scanned = 0
            rejected_hi = 0
            rejected_pen = 0
            
            for curr in curr_range:
                for spd in speed_range:
                    for fil in fillers:
                        total_scanned += 1
                        
                        # Calcs
                        hi = (voltage * curr * efficiency) / spd
                        
                        # Constraint 1: Heat Input
                        if hi > max_hi:
                            rejected_hi += 1
                            continue
                            
                        # Predict
                        f_code = 1 if fil == 'ER316L' else 0
                        feats = np.array([[curr, voltage, spd, f_code, interpass, hi]])
                        feats_scaled = MODELS['scaler'].transform(feats)
                        
                        pred_t = float(MODELS['tensile_model'].predict(feats_scaled)[0])
                        pred_p = float(MODELS['penetration_model'].predict(feats_scaled)[0])
                        
                        # Constraint 2: Penetration (Targeting full penetration, e.g., >= thickness)
                        # Relaxing it slightly to >= thickness * 0.9 for practical results or strictly >= 3.0
                        if pred_p < 3.0: # Hard requirement from requirements "Full Penetration (>3mm)"
                             rejected_pen += 1
                             continue
                        
                        # Score (Maximize Tensile)
                        candidates.append({
                            'Current_A': curr,
                            'Speed_mm_min': spd,
                            'Filler_Type': fil,
                            'Heat_Input_kJ_mm': hi,
                            'Pred_Tensile_MPa': pred_t,
                            'Pred_Penetration_mm': pred_p,
                            'Safety_Margin_HI': max_hi - hi
                        })
            
            # Sort by Tensile Strength (Desc)
            candidates.sort(key=lambda x: x['Pred_Tensile_MPa'], reverse=True)
            
            if not candidates:
                 self.send_json({
                    'success': False, 
                    'error': 'No solution found satisfying all constraints. Try relaxing Heat Input limits or Speed ranges.',
                    'statistics': {
                        'total_scanned': total_scanned, 
                        'valid_count': 0,
                        'rejected_hi': rejected_hi,
                        'rejected_pen': rejected_pen
                    }
                })
                 return

            self.send_json({
                'success': True,
                'optimal': candidates[0],
                'top5': candidates[:5],
                'statistics': {
                    'total_scanned': total_scanned,
                    'valid_count': len(candidates),
                    'rejected_hi': rejected_hi,
                    'rejected_pen': rejected_pen
                }
            })
            
        except Exception as e:
            self.send_json({'success': False, 'error': str(e)})

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f'Starting MIG Optimizer Server on port {port}...')
    print(f'Open http://localhost:{port} in your browser')
    
    # Auto-open browser
    webbrowser.open(f'http://localhost:{port}')
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\\nServer stopped.')
        httpd.server_close()

if __name__ == '__main__':
    run_server()