BACKEND_URL = 'http://127.0.0.1:5000';
const appState = {
    connected: false,
    sessionId: null,
    equipment: null,
    lastDetection: null,
    sensorData: [],
    backendHealth: null,
    isStreaming: false,
    videoStream: null,
    socket: io(BACKEND_URL),
    recognition: null,
    listening: false
};

// Navigation SPA
function showSection(sectionId, evt) {
    document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');
    if (evt) evt.target.classList.add('active');
    addLog(`Switched to ${sectionId}`);
}

function showAlert(message, timeout = 4000) {
    let alertBox = document.getElementById('alertBox');
    if (!alertBox) {
        alertBox = document.createElement('div');
        alertBox.id = 'alertBox';
        alertBox.style.position = 'fixed';
        alertBox.style.top = '20px';
        alertBox.style.right = '20px';
        alertBox.style.zIndex = 1000;
        alertBox.style.background = '#e74c3c';
        alertBox.style.color = '#fff';
        alertBox.style.padding = '15px 25px';
        alertBox.style.borderRadius = '10px';
        alertBox.style.boxShadow = '0 4px 16px rgba(0,0,0,0.2)';
        alertBox.style.fontWeight = 'bold';
        alertBox.style.fontSize = '16px';
        document.body.appendChild(alertBox);
    }
    alertBox.textContent = message;
    alertBox.style.display = 'block';
    setTimeout(() => {
        alertBox.style.display = 'none';
    }, timeout);
}

function subscribeToSensors(serial) {
    if (!appState.socket || !appState.connected) {
        addLog('Not connected, cannot subscribe to sensors.', 'WARN');
        return;
    }
    if (!serial) {
        addLog('No serial provided for subscription.', 'WARN');
        return;
    }
    appState.socket.emit('subscribe_sensor', { serial_number: serial });
    addLog(`Subscribed to sensor data for ${serial}`);
}

function unsubscribeFromSensors(serial) {
    if (!appState.socket || !appState.connected) return;
    if (!serial) return;
    appState.socket.emit('unsubscribe_sensor', { serial_number: serial });
    addLog(`Unsubscribed from sensor data for ${serial}`);
}



// Logging
function addLog(message, type = 'INFO') {
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
    if (appState.connected) {
        addLog('Already connected');
        return;
    }
    appState.socket = io(`${BACKEND_URL}`);
    appState.socket.on('connect', () => {
        appState.connected = true;
        appState.sessionId = appState.socket.id;
        updateStatusUI();
        addLog('WebSocket connected');
    });

    function autoReconnectWebSocket(delay = 5000) {
        if (!appState.connected) {
            setTimeout(() => {
                addLog('Tentative de reconnexion WebSocket...');
                connectWebSocket();
            }, delay);
        }
    }

// Appelle autoReconnectWebSocket() dans ton handler de déconnexion :
    appState.socket.on('disconnect', () => {
        appState.connected = false;
        updateStatusUI();
        addLog('WebSocket disconnected', 'WARN');
        showAlert('WebSocket déconnecté du serveur !');
        autoReconnectWebSocket(5000); // Reconnexion après 5 sec
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
    appState.socket.on('subscription_status', data => {
        if (data.status === 'subscribed') {
            addLog(`Subscription OK: ${data.serial_number}`);
        } else if (data.status === 'unsubscribed') {
            addLog(`Unsubscribed: ${data.serial_number}`);
        } else {
            addLog(`Sensor subscription error: ${data.message}`, 'ERROR');
            showAlert(`Erreur abonnement capteurs: ${data.message}`);
        }
    });

    appState.socket.on('stream_status', data => { /* Optionnel */
    });
    appState.socket.on('error', data => {
        addLog(`WebSocket error: ${data.message}`, 'ERROR');
        showAlert(`Erreur WebSocket : ${data.message}`);
        // Affiche une notification ou un message temporaire dans l’UI
    });

}

function disconnectWebSocket() {
    if (appState.socket) {
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
    if (statusEl && textEl && sessionEl) {
        if (appState.connected) {
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
    if (equipEl) equipEl.textContent = appState.equipment?.serial_number || 'None';
    const captureBtn = document.getElementById('captureBtn');
    if (captureBtn) captureBtn.disabled = !appState.connected;
    const streamBtn = document.getElementById('streamToggle');
    if (streamBtn) streamBtn.disabled = !appState.connected || !appState.videoStream;

}

// Camera / Streaming
async function startCamera() {
    try {
        appState.videoStream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
        const videoEl = document.getElementById('videoElement');
        if (videoEl) videoEl.srcObject = appState.videoStream;
        addLog('Camera started successfully');
        const btn = document.getElementById('captureBtn');
        if (btn) btn.disabled = false;
    } catch (e) {
        addLog(`Failed to start camera: ${e.message}`, 'ERROR');
        showAlert("Impossible d'accéder à la caméra. Vérifiez les droits ou branchez un périphérique.");
    }
}


function stopCamera() {
    if (appState.videoStream) {
        appState.videoStream.getTracks().forEach(track => track.stop());
        appState.videoStream = null;
        const videoEl = document.getElementById('videoElement');
        if (videoEl) videoEl.srcObject = null;
        addLog('Camera stopped');
        if (appState.isStreaming) toggleStreaming();
    }
}

function toggleStreaming() {
    if (!appState.videoStream) return addLog('Start camera first', 'WARN');
    if (!appState.socket || !appState.connected) return addLog('Connect to WebSocket first', 'WARN');
    const toggleBtn = document.getElementById('streamToggle');
    if (appState.isStreaming) {
        appState.isStreaming = false;
        if (toggleBtn) {
            toggleBtn.textContent = 'Start Streaming';
            toggleBtn.className = 'btn btn-success';
        }
        addLog('Video streaming stopped');
    } else {
        appState.isStreaming = true;
        if (toggleBtn) {
            toggleBtn.textContent = 'Stop Streaming';
            toggleBtn.className = 'btn btn-warning';
        }
        addLog('Video streaming started');
        streamVideo();
    }
}

function streamVideo() {
    if (!appState.isStreaming || !appState.videoStream || !appState.socket?.connected) return;
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('captureCanvas');
    if (!video || !canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(function (blob) {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = function () {
            const base64data = reader.result.split(',')[1];
            appState.socket.emit('video_stream', {frame: base64data});
        };
        reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.8);
    if (appState.isStreaming) setTimeout(streamVideo, 100); // 10 FPS
}


function captureFrame() {
    if (!appState.socket || !appState.connected) return addLog('Connect to WebSocket first', 'WARN');
    appState.socket.emit('capture_frame', {});
    addLog('Frame capture requested');
    const status = document.getElementById('captureStatus');
    if (status) status.textContent = 'Capture requested, waiting for analysis...';
}

// Upload image REST (QR + Defect)
function uploadImage(input) {
    const file = input.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('image', file);

    if (!file.type.startsWith('image/')) {
        addLog('Format invalide. Fichier non image.', 'ERROR');
        showAlert('Format de fichier non supporté (image attendue)');
        return;
    }

    // QR detection
    // Détection QR
    fetch(`${BACKEND_URL}/detect_qrcode`, {method: 'POST', body: formData})
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                addLog(`Erreur détection QR: ${data.error}`, 'ERROR');
                showAlert(`Erreur QR: ${data.error}`);
            } else {
                addLog(`QR Detection result: ${JSON.stringify(data)}`);
                displayDetectionResult({detection: data, type: 'qr'});
            }
        })
        .catch(e => {
            addLog(`QR Detection error: ${e.message}`, 'ERROR');
            showAlert(`Erreur réseau (QR): ${e.message}`);
        });

    // Détection Défaut
    fetch(`${BACKEND_URL}/detect_defect`, {method: 'POST', body: formData})
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                addLog(`Erreur détection défaut: ${data.error}`, 'ERROR');
                showAlert(`Erreur défaut: ${data.error}`);
            } else {
                addLog(`Defect Detection result: ${JSON.stringify(data)}`);
                displayDetectionResult({detection: data, type: 'defect'});
            }
        })
        .catch(e => {
            addLog(`Defect Detection error: ${e.message}`, 'ERROR');
            showAlert(`Erreur réseau (défaut): ${e.message}`);
        });
    addLog(`Image uploaded: ${file.name}`);
}

function sendTestQR() {
    document.getElementById('imageUpload').click();
}

function sendTestDefect() {
    document.getElementById('imageUpload').click();
}

// Equipment & Documentation
let currentSensorSerial = null;

function updateEquipmentDisplay(equipment) {
    // Unsubscribe/resubscribe capteurs
    if (currentSensorSerial && currentSensorSerial !== equipment.serial_number) {
        unsubscribeFromSensors(currentSensorSerial);
        resetSensorUI();
    }
    if (equipment.serial_number && currentSensorSerial !== equipment.serial_number) {
        subscribeToSensors(equipment.serial_number);
        currentSensorSerial = equipment.serial_number;
    }

    // Mise à jour UI info
    const equipEl = document.getElementById('currentEquipment');
    if (equipEl) equipEl.textContent = equipment.serial_number || '-';
    document.getElementById('equipmentSerial').textContent = `Serial: ${equipment.serial_number || '-'}`;
    document.getElementById('equipmentModel').textContent = `Model: ${equipment.equipment_name || equipment.model || '-'}`;
    document.getElementById('equipmentStatus').textContent = `Status: ${equipment.status || '-'}`;
}

function resetSensorUI() {
    document.getElementById('tempValue').textContent = '--°C';
    document.getElementById('tempTime').textContent = 'Never';
    document.getElementById('vibrationValue').textContent = '-- Hz';
    document.getElementById('vibrationTime').textContent = 'Never';
    document.getElementById('equipmentStatusValue').textContent = 'Unknown';
    document.getElementById('sensorSerial').textContent = '-';
    document.getElementById('sensorLog').innerHTML = '<div>[SENSORS] Waiting for sensor data...</div>';
}


function fetchEquipmentInfo() {
    const serial = document.getElementById('serialInput').value.trim();
    if (!serial) {
        addLog('Enter serial number first', 'WARN');
        showAlert('Entrez un numéro de série');
        return;
    }
    fetch(`${BACKEND_URL}/equipment/${serial}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                addLog(`Equipment fetch error: ${data.error}`, 'ERROR');
                showAlert(`Équipement non trouvé (${data.error})`);
                updateEquipmentDisplay({ serial_number: serial, equipment_name: '-', model: '-', status: 'Not found' });
            }
            else {
                addLog(`Equipment info fetched: ${serial}`);
                appState.equipment = data;
                updateEquipmentDisplay(data);
            }
        }).catch(e => {
        addLog(`Fetch error: ${e.message}`, 'ERROR');
        showAlert('Erreur réseau lors de la récupération équipement');
    });
}


function fetchDocumentation() {
    const serial = appState.equipment?.serial_number || document.getElementById('serialInput').value.trim();
    if (!serial) {
        addLog('No equipment identified or serial entered', 'WARN');
        showAlert('Identifiez un équipement ou entrez un numéro de série');
        return;
    }
    fetch(`${BACKEND_URL}/equipment/${serial}/documentation`)
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                addLog(`Documentation fetch error: ${data.error}`, 'ERROR');
                document.getElementById('documentationContent').innerHTML = `<p style="color:#e74c3c;">Error: ${data.error}</p>`;
            } else {
                addLog(`Documentation loaded for: ${serial}`);
                document.getElementById('documentationContent').innerHTML =
                    `<h3>Documentation for ${serial}</h3><pre style="white-space:pre-wrap;">${typeof data.documentation === 'object' ? JSON.stringify(data.documentation, null, 2) : data.documentation}</pre>`;
            }
        }).catch(e => {
        addLog(`Documentation fetch error: ${e.message}`, 'ERROR');
        document.getElementById('documentationContent').innerHTML =
            `<p style="color:#e74c3c;">Erreur réseau lors de la récupération documentation</p>`;
    });
}

// Detection Results
function displayDetectionResult(data) {
    const resultsDiv = document.getElementById('detectionResults');
    const contentDiv = document.getElementById('detectionContent');
    let html = `<h4>Detection Results (${new Date().toLocaleTimeString()})</h4>`;
    if (data.equipment_id) html += `<p><strong>Equipment:</strong> ${data.equipment_id}</p>`;
    if (data.detection) {
        html += `<p><strong>Status:</strong> <span style="color:${data.detection.status == 'GOOD' ? '#4caf50' : '#e53935'}">${data.detection.status || 'Unknown'}</span></p>`;
        if (data.detection.qr_code) html += `<p><strong>QR Code:</strong> ${data.detection.qr_code}</p>`;
        if (data.detection.anomaly_score !== undefined) html += `<p><strong>Anomaly Score:</strong> ${data.detection.anomaly_score}</p>`;
        if (data.detection.details) html += `<p><strong>Details:</strong> ${JSON.stringify(data.detection.details)}</p>`;
    }
    contentDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    document.getElementById('captureStatus').textContent = `Last capture: ${new Date().toLocaleTimeString()} - ${data.detection?.status || 'Completed'}`;
}

// Sensor Data
function updateSensorData(data) {
    const sensorData = data.data, serial = data.serial_number;
    if (!serial || !sensorData) return;
    document.getElementById('tempValue').textContent = (sensorData.temperature !== undefined) ? `${sensorData.temperature}°C` : '--°C';
    document.getElementById('tempTime').textContent = new Date().toLocaleTimeString();
    document.getElementById('vibrationValue').textContent = (sensorData.vibration !== undefined) ? `${sensorData.vibration} Hz` : '-- Hz';
    document.getElementById('vibrationTime').textContent = new Date().toLocaleTimeString();
    document.getElementById('sensorSerial').textContent = serial;
    document.getElementById('equipmentStatusValue').textContent = 'Active';
    sensorLog(`${serial}: T=${sensorData.temperature}°C, V=${sensorData.vibration}Hz`);
}


// Health Check
function checkHealth() {
    fetch(`${BACKEND_URL}/health`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            addLog(`Health check: ${data.status} (${data.version || ''})`);
            document.getElementById('backendStatus').textContent = `Status: ${data.status} (v${data.version || 'N/A'})`;
            document.getElementById('backendTime').textContent = `Last check: ${new Date().toLocaleTimeString()}`;
            document.getElementById('dbStatus').textContent = `Status: ${data.database}`;
            document.getElementById('aiStatus').textContent = `Status: ${data.ai_models}`;
        })
        .catch(e => {
            addLog(`Health check error: ${e.message}`, 'ERROR');
            showAlert('Backend non disponible ou erreur santé !');
            document.getElementById('backendStatus').textContent = 'Status: ERROR';
            document.getElementById('dbStatus').textContent = 'Status: -';
            document.getElementById('aiStatus').textContent = 'Status: -';
        });
}

// Voice Control (Browser SpeechRecognition)
function startVoiceRecognition() {
    if (appState.listening) return;
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        document.getElementById('voiceStatusText').textContent = "Not supported";
        addLog('Voice recognition not supported in this browser', 'ERROR');
        showAlert("Reconnaissance vocale non supportée sur ce navigateur.");
        return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    appState.recognition = new SpeechRecognition();
    appState.recognition.lang = "en-US"; // ou "fr-FR" si tu veux en français
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
        if (event.error === 'no-speech') {
            showAlert("Aucun son détecté. Essayez de parler plus rapidement après avoir cliqué sur Start.");
        } else if (event.error === 'audio-capture') {
            showAlert("Micro non détecté ou non autorisé.");
        } else if (event.error === 'not-allowed') {
            showAlert("Permission micro refusée. Vérifiez les réglages navigateur.");
        } else {
            showAlert(`Erreur reconnaissance vocale : ${event.error}`);
        }
    };

    appState.recognition.onend = () => {
        appState.listening = false;
        document.getElementById('voiceStatusText').textContent = "Ready";
    };
    appState.recognition.onresult = (event) => {
        const command = event.results[0][0].transcript.trim().toLowerCase();
        document.getElementById('lastCommand').textContent = command;
        addLog(`Voice heard: "${command}"`);
        handleVoiceCommand(command); // mapping ici
        document.getElementById('voiceStatusText').textContent = "Ready";
    };
    appState.recognition.start();
}

function stopVoiceRecognition() {
    if (appState.recognition && appState.listening) {
        appState.recognition.stop();
        appState.listening = false;
        document.getElementById('voiceStatusText').textContent = "Stopped";
        addLog('Voice recognition stopped');
    }
}
function handleVoiceCommand(command) {
    command = command.trim().toLowerCase();
    if (!appState.socket || !appState.connected) {
        showAlert("WebSocket non connectée. Action impossible.");
        return;
    }
    if (!appState.videoStream && besoinVideo) {
        showAlert("Aucune caméra détectée.");
        return;
    }

    // ===================== AFFICHER CAPTEURS =====================
    if (
        command.includes("show sensor") ||
        command.includes("show sensors") ||
        command.includes("afficher données capteur") ||
        command.includes("afficher capteur") ||
        command.includes("afficher les capteurs") ||
        command.includes("afficher sensors")
    ) {
        showSection('sensors');
        showAlert("Sensor data displayed.");
        return;
    }

    // ===================== AFFICHER DOCUMENTATION =====================
    if (
        command.includes("show documentation") ||
        command.includes("open documentation") ||
        command.includes("afficher documentation") ||
        command.includes("ouvrir documentation") ||
        command.includes("afficher la documentation")
    ) {
        showSection('doc');
        fetchDocumentation();
        showAlert("Documentation displayed.");
        return;
    }

    // ===================== FERMER DOCUMENTATION =====================
    if (
        command.includes("close documentation") ||
        command.includes("fermer documentation") ||
        command.includes("retour accueil") ||
        command.includes("aller accueil") ||
        command.includes("back to dashboard") ||
        command.includes("go home") ||
        command.includes("return dashboard") ||
        command.includes("dashboard") ||
        command.includes("accueil")
    ) {
        showSection('dashboard');
        showAlert("Returned to dashboard.");
        return;
    }

    // ===================== DÉTECTION ANOMALIE / DEFECT =====================
    if (
        command.includes("detect defect") ||
        command.includes("detecter anomalie") ||
        command.includes("détecter anomalie") ||
        command.includes("detect anomaly") ||
        command.includes("détecter défaut") ||
        command.includes("detect anomaly") ||
        command.includes("detect defect")
    ) {
        // On va sur Capture et lance captureFrame()
        showSection('capture');
        setTimeout(() => captureFrame(), 400); // Laisse le temps d'afficher la section
        showAlert("Anomaly detection triggered.");
        return;
    }

    // ===================== AUTRES COMMANDES STANDARD =====================
    if (command.includes("start camera") || command.includes("démarrer caméra")) { startCamera(); return; }
    if (command.includes("stop camera") || command.includes("arrêter caméra")) { stopCamera(); return; }
    if (command.includes("start streaming") || command.includes("démarrer streaming")) { toggleStreaming(); return; }
    if (command.includes("stop streaming") || command.includes("arrêter streaming")) { toggleStreaming(); return; }
    if (command.includes("capture") || command.includes("capturer")) { captureFrame(); return; }
    if (command.includes("connect")) { connectWebSocket(); return; }
    if (command.includes("disconnect")) { disconnectWebSocket(); return; }
    if (command.includes("health check") || command.includes("état système") || command.includes("check health") || command.includes("vérifier état")) { checkHealth(); return; }
    if (command.includes("fetch info") || command.includes("info équipement") || command.includes("show equipment")) { fetchEquipmentInfo(); showSection('info'); return; }
    if (command.includes("stream") || command.includes("video")) { showSection('stream'); return; }
    if (command.includes("voice") || command.includes("commande vocale")) { showSection('voice'); return; }

    // Option : Feedback inconnu
    showAlert(`Commande non reconnue: "${command}"`);
}




// Ready
document.addEventListener('DOMContentLoaded', function () {
    addLog('Application initialized');
    // Optionnel: auto-connexion WS ou autres initialisations
});
