const appState = {
    connected: false,
    sessionId: null,
    equipment: null,
    lastDetection: null,
    sensorData: [],
    backendHealth: null,
    isStreaming: false,
    videoStream: null,
    socket: null,
    recognition: null,
    listening: false
};

// Navigation SPA
function showSection(sectionId, evt) {
    document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');
    if(evt) evt.target.classList.add('active');
    addLog(`Switched to ${sectionId}`);
}

// Logging
function addLog(message, type='INFO') {
    const logDiv = document.getElementById('systemLog');
    if (!logDiv) return;
    const entry = document.createElement('div');
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${type}: ${message}`;
    logDiv.appendChild(entry);
    logDiv.scrollTop = logDiv.scrollHeight;
}
function sensorLog(message) {
    const sensorLogEl = document.getElementById('sensorLog');
    if (!sensorLogEl) return;
    const entry = document.createElement('div');
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    sensorLogEl.appendChild(entry);
    sensorLogEl.scrollTop = sensorLogEl.scrollHeight;
}

// WebSocket
function connectWebSocket() {
    if(appState.connected) { addLog('Already connected'); return; }
    appState.socket = io('http://127.0.0.1:5000');
    appState.socket.on('connect', () => {
        appState.connected = true;
        appState.sessionId = appState.socket.id;
        updateStatusUI();
        addLog('WebSocket connected');
    });
    appState.socket.on('disconnect', () => {
        appState.connected = false;
        updateStatusUI();
        addLog('WebSocket disconnected', 'WARN');
    });
    appState.socket.on('connection_status', data => {
        addLog(`Connection status: ${data.status}, Session: ${data.session_id}`);
    });
    appState.socket.on('equipment_identified', data => {
        addLog(`Equipment identified: ${data.equipment.serial_number}`);
        appState.equipment = data.equipment;
        updateEquipmentDisplay(data.equipment);
    });
    appState.socket.on('detection_result', data => {
        addLog(`Detection result received: ${data.detection.status}`);
        appState.lastDetection = data;
        displayDetectionResult(data);
    });
    appState.socket.on('sensor_data', data => {
        updateSensorData(data);
    });
    appState.socket.on('stream_status', data => { /* Optionnel */ });
    appState.socket.on('error', data => {
        addLog(`WebSocket error: ${data.message}`, 'ERROR');
        // Optionnel: affiche une alerte utilisateur
    });
}
function disconnectWebSocket() {
    if(appState.socket) {
        appState.socket.disconnect();
        appState.socket = null;
        appState.connected = false;
        updateStatusUI();
        addLog('WebSocket disconnected');
    }
}
function updateStatusUI() {
    const statusEl = document.getElementById('connectionStatus');
    const textEl = document.getElementById('connectionText');
    const sessionEl = document.getElementById('sessionId');
    if(statusEl && textEl && sessionEl) {
        if(appState.connected) {
            statusEl.className = 'status-indicator status-connected';
            textEl.textContent = 'Connected';
            sessionEl.textContent = appState.sessionId || '-';
        } else {
            statusEl.className = 'status-indicator status-disconnected';
            textEl.textContent = 'Disconnected';
            sessionEl.textContent = '-';
        }
    }
    const equipEl = document.getElementById('currentEquipment');
    if(equipEl) equipEl.textContent = appState.equipment?.serial_number || 'None';
}

// Camera / Streaming
async function startCamera() {
    try {
        appState.videoStream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480 } });
        const videoEl = document.getElementById('videoElement');
        if (videoEl) videoEl.srcObject = appState.videoStream;
        addLog('Camera started successfully');
        const btn = document.getElementById('captureBtn');
        if (btn) btn.disabled = false;
    } catch(e) {
        addLog(`Failed to start camera: ${e.message}`, 'ERROR');
    }
}
function stopCamera() {
    if(appState.videoStream) {
        appState.videoStream.getTracks().forEach(track=>track.stop());
        appState.videoStream = null;
        const videoEl = document.getElementById('videoElement');
        if (videoEl) videoEl.srcObject = null;
        addLog('Camera stopped');
        if(appState.isStreaming) toggleStreaming();
    }
}
function toggleStreaming() {
    if(!appState.videoStream) return addLog('Start camera first', 'WARN');
    if(!appState.socket || !appState.connected) return addLog('Connect to WebSocket first', 'WARN');
    const toggleBtn = document.getElementById('streamToggle');
    if(appState.isStreaming) {
        appState.isStreaming = false;
        if(toggleBtn) {
            toggleBtn.textContent = 'Start Streaming';
            toggleBtn.className = 'btn btn-success';
        }
        addLog('Video streaming stopped');
    } else {
        appState.isStreaming = true;
        if(toggleBtn) {
            toggleBtn.textContent = 'Stop Streaming';
            toggleBtn.className = 'btn btn-warning';
        }
        addLog('Video streaming started');
        streamVideo();
    }
}
function streamVideo() {
    if(!appState.isStreaming || !appState.videoStream || !appState.socket?.connected) return;
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('captureCanvas');
    if (!video || !canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(function(blob) {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = function() {
            const base64data = reader.result.split(',')[1];
            appState.socket.emit('video_stream', { frame: base64data });
        };
        reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.8);
    if(appState.isStreaming) setTimeout(streamVideo, 100); // 10 FPS
}
function captureFrame() {
    if(!appState.socket || !appState.connected) return addLog('Connect to WebSocket first', 'WARN');
    appState.socket.emit('capture_frame', {});
    addLog('Frame capture requested');
    const status = document.getElementById('captureStatus');
    if (status) status.textContent = 'Capture requested, waiting for analysis...';
}

// Upload image REST (QR + Defect)
function uploadImage(input) {
    const file = input.files[0]; if(!file) return;
    const formData = new FormData();
    formData.append('image', file);

    // QR detection
    fetch('http://127.0.0.1:5000/detect_qrcode', { method:'POST', body:formData })
        .then(r=>r.json()).then(data=>{
        addLog(`QR Detection result: ${JSON.stringify(data)}`);
        displayDetectionResult({ detection:data, type:'qr' });
    }).catch(e=>{ addLog(`QR Detection error: ${e.message}`, 'ERROR'); });

    // Defect detection
    fetch('http://127.0.0.1:5000/detect_defect', { method:'POST', body:formData })
        .then(r=>r.json()).then(data=>{
        addLog(`Defect Detection result: ${JSON.stringify(data)}`);
        displayDetectionResult({ detection:data, type:'defect' });
    }).catch(e=>{ addLog(`Defect Detection error: ${e.message}`, 'ERROR'); });

    addLog(`Image uploaded: ${file.name}`);
}
function sendTestQR() {
    document.getElementById('imageUpload').click();
}
function sendTestDefect() {
    document.getElementById('imageUpload').click();
}

// Equipment & Documentation
function updateEquipmentDisplay(equipment) {
    const equipEl = document.getElementById('currentEquipment');
    if (equipEl) equipEl.textContent = equipment.serial_number;
    document.getElementById('equipmentSerial').textContent = `Serial: ${equipment.serial_number}`;
    document.getElementById('equipmentModel').textContent = `Model: ${equipment.equipment_name || equipment.model || 'Unknown'}`;
    document.getElementById('equipmentStatus').textContent = `Status: ${equipment.status || 'Active'}`;
}
function fetchEquipmentInfo() {
    const serial = document.getElementById('serialInput').value.trim();
    if(!serial) return addLog('Enter serial number first', 'WARN');
    fetch(`http://127.0.0.1:5000/equipment/${serial}`)
        .then(r=>r.json())
        .then(data=>{
            if(data.error) addLog(`Equipment fetch error: ${data.error}`, 'ERROR');
            else {
                addLog(`Equipment info fetched: ${serial}`);
                appState.equipment = data;
                updateEquipmentDisplay(data);
            }
        }).catch(e=>addLog(`Fetch error: ${e.message}`, 'ERROR'));
}
function fetchDocumentation() {
    const serial = appState.equipment?.serial_number || document.getElementById('serialInput').value.trim();
    if(!serial) return addLog('No equipment identified or serial entered', 'WARN');
    fetch(`http://127.0.0.1:5000/equipment/${serial}/documentation`)
        .then(r=>r.json())
        .then(data=>{
            if(data.error) {
                addLog(`Documentation fetch error: ${data.error}`, 'ERROR');
                document.getElementById('documentationContent').innerHTML = `<p>Error: ${data.error}</p>`;
            } else {
                addLog(`Documentation loaded for: ${serial}`);
                document.getElementById('documentationContent').innerHTML =
                    `<h3>Documentation for ${serial}</h3><pre>${JSON.stringify(data.documentation, null, 2)}</pre>`;
            }
        }).catch(e=>addLog(`Documentation fetch error: ${e.message}`, 'ERROR'));
}

// Detection Results
function displayDetectionResult(data) {
    const resultsDiv = document.getElementById('detectionResults');
    const contentDiv = document.getElementById('detectionContent');
    let html = `<h4>Detection Results (${new Date().toLocaleTimeString()})</h4>`;
    if(data.equipment_id) html += `<p><strong>Equipment:</strong> ${data.equipment_id}</p>`;
    if(data.detection) {
        html += `<p><strong>Status:</strong> <span style="color:${data.detection.status=='GOOD'?'#4caf50':'#e53935'}">${data.detection.status || 'Unknown'}</span></p>`;
        if(data.detection.qr_code) html += `<p><strong>QR Code:</strong> ${data.detection.qr_code}</p>`;
        if(data.detection.anomaly_score !== undefined) html += `<p><strong>Anomaly Score:</strong> ${data.detection.anomaly_score}</p>`;
        if(data.detection.details) html += `<p><strong>Details:</strong> ${JSON.stringify(data.detection.details)}</p>`;
    }
    contentDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    document.getElementById('captureStatus').textContent = `Last capture: ${new Date().toLocaleTimeString()} - ${data.detection?.status || 'Completed'}`;
}

// Sensor Data
function updateSensorData(data) {
    const sensorData = data.data, serial = data.serial_number;
    if(sensorData.temperature!==undefined) {
        document.getElementById('tempValue').textContent = `${sensorData.temperature}°C`;
        document.getElementById('tempTime').textContent = new Date().toLocaleTimeString();
    }
    if(sensorData.vibration!==undefined) {
        document.getElementById('vibrationValue').textContent = `${sensorData.vibration} Hz`;
        document.getElementById('vibrationTime').textContent = new Date().toLocaleTimeString();
    }
    document.getElementById('sensorSerial').textContent = serial;
    document.getElementById('equipmentStatusValue').textContent = 'Active';
    sensorLog(`${serial}: T=${sensorData.temperature}°C, V=${sensorData.vibration}Hz`);
}

// Health Check
function checkHealth() {
    fetch('http://127.0.0.1:5000/health')
        .then(r=>r.json())
        .then(data=>{
            addLog(`Health check: ${data.status}`);
            document.getElementById('backendStatus').textContent = `Status: ${data.status}`;
            document.getElementById('backendTime').textContent = `Last check: ${new Date().toLocaleTimeString()}`;
            document.getElementById('dbStatus').textContent = `Status: ${data.database}`;
            document.getElementById('aiStatus').textContent = `Status: ${data.ai_models}`;
        }).catch(e=>addLog(`Health check error: ${e.message}`, 'ERROR'));
}

// Voice Control (Browser SpeechRecognition)
function startVoiceRecognition() {
    if(appState.listening) return;
    if(!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        document.getElementById('voiceStatusText').textContent = "Not supported";
        addLog('Voice recognition not supported in this browser', 'ERROR');
        return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    appState.recognition = new SpeechRecognition();
    appState.recognition.lang = "en-US";
    appState.recognition.interimResults = false;
    appState.recognition.maxAlternatives = 1;
    appState.recognition.onstart = () => {
        appState.listening = true;
        document.getElementById('voiceStatusText').textContent = "Listening...";
        addLog('Voice recognition started');
    };
    appState.recognition.onerror = (event) => {
        appState.listening = false;
        document.getElementById('voiceStatusText').textContent = "Error";
        addLog(`Voice error: ${event.error}`, 'ERROR');
    };
    appState.recognition.onend = () => {
        appState.listening = false;
        document.getElementById('voiceStatusText').textContent = "Ready";
    };
    appState.recognition.onresult = (event) => {
        const command = event.results[0][0].transcript.trim().toLowerCase();
        document.getElementById('lastCommand').textContent = command;
        addLog(`Voice heard: "${command}"`);
        // Actions mapping
        if(command.includes("start camera")) startCamera();
        if(command.includes("stop camera")) stopCamera();
        if(command.includes("capture")) captureFrame();
        if(command.includes("connect")) connectWebSocket();
        if(command.includes("disconnect")) disconnectWebSocket();
        if(command.includes("health check")) checkHealth();
        document.getElementById('voiceStatusText').textContent = "Ready";
    };
    appState.recognition.start();
}
function stopVoiceRecognition() {
    if(appState.recognition && appState.listening) {
        appState.recognition.stop();
        appState.listening = false;
        document.getElementById('voiceStatusText').textContent = "Stopped";
        addLog('Voice recognition stopped');
    }
}

// Ready
document.addEventListener('DOMContentLoaded', function() {
    addLog('Application initialized');
    // Optionnel: auto-connexion WS ou autres initialisations
});
