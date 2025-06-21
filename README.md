# AR Assembly Detection

## Overview

This project is a system for augmented reality (AR) assisted defect detection in industrial components, specifically screws. It combines computer vision, machine learning, and web technologies to provide a real-time interface for identifying manufacturing defects. The system can also integrate with sensor data from equipment and offers voice control capabilities.

## Features

*   **Screw Defect Detection:** Utilizes machine learning models (e.g., Autoencoders) to identify anomalies and defects in screw images.
*   **QR Code Scanning:** Detects QR codes to identify equipment and retrieve relevant information.
*   **Real-time Video Streaming:** Supports live video streaming from a client device to the server for analysis.
*   **Sensor Data Integration:** Receives and displays sensor data (e.g., temperature, vibration) from connected equipment (simulated or real).
*   **Web-Based User Interface:** Provides a comprehensive web client for interacting with the system, viewing results, and managing equipment.
*   **WebSocket Communication:** Employs SocketIO for efficient real-time, bidirectional communication between the client and server.
*   **Database Integration:** Manages equipment details and (planned) persistence of sensor/detection data using PostgreSQL.
*   **Voice Control:** Offers voice commands for hands-free operation of the client interface.
*   **ESP32 Simulation:** Includes a script to simulate an ESP32 device sending sensor data for testing and development.

## Technology Stack

*   **Backend:** Python, Flask, Flask-SocketIO, OpenCV, PyTorch, SQLAlchemy, Psycopg2
*   **Frontend:** HTML, CSS, JavaScript, Socket.IO (client)
*   **Database:** PostgreSQL
*   **Machine Learning:** Scikit-learn, Ultralytics (YOLO), TensorFlow (potentially, based on conda environment)
*   **Libraries:** Pyzbar (QR code decoding), Pillow, Requests, SpeechRecognition, Albumentations (image augmentation)
*   **Environment Management:** Conda

## Project Structure

```
.
├── app/                      # Flask application (frontend)
│   ├── static/               # Static assets (CSS, JS, images)
│   ├── templates/            # HTML templates
│   └── ...
├── archived/                 # Older scripts, experimental features
├── dataset/                  # Datasets, including the MVTec Screw dataset
│   └── screw/
│       ├── ground_truth/
│       ├── test/
│       ├── train/
│       ├── dataset.yaml
│       └── readme.txt        # MVTec dataset information
├── models/                   # Trained machine learning models (e.g., autoencoder_screw.pt)
├── __pycache__/              # Python bytecode cache
├── cert.pem                  # SSL certificate (self-signed for development)
├── client.py                 # Main script for the AR Web Client (alternative to app/)
├── client_test.py            # Test script for the client
├── db.py                     # Database interaction logic
├── esp32_sensor_sim.py       # ESP32 sensor data simulator
├── esp32_sensors.ino         # Arduino code for ESP32 (actual hardware)
├── key.pem                   # SSL private key (self-signed for development)
├── lab_instructions.md       # Setup and usage instructions (primarily in French)
├── LICENSE                   # Project license
├── main_qrcode.py            # Script focused on QR code detection
├── qr_code_detector.py       # QR code detection module
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── reset_dataset_from_clean_copy.py # Script to reset the dataset
├── server.py                 # Main Flask server application
├── server_doc.yaml           # OpenAPI/Swagger documentation for the server (likely)
├── sqldb_manip.sql           # SQL scripts for database manipulation
├── sudo_detection.py         # Main detection management logic
├── test_media/               # Media files for testing (e.g., QR codes)
├── voice_control.py          # Voice control implementation
└── ws_server_doc.yaml        # WebSocket API documentation (likely)
```

## Setup and Installation

*(Based on `lab_instructions.md`)*

1.  **Prerequisites:**
    *   Conda (Miniconda recommended)
    *   Python 3.11
    *   Functional microphone (for voice control)

2.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n ai_env python=3.11
    conda activate ai_env
    ```

3.  **Install Dependencies:**
    The `lab_instructions.md` provides several ways to install dependencies. A consolidated approach is:
    ```bash
    # Install core packages
    conda install -y -c defaults numpy flask requests PyYAML
    conda install -y -c conda-forge pyzbar Pillow qrcode opencv SpeechRecognition psycopg2 scikit-learn albumentations matplotlib seaborn flask-socketio flask-sqlalchemy python-socketio websocket-client tensorflow ultralytics

    # Install PyTorch (CPU version recommended in instructions)
    conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch

    # Install any remaining packages from requirements.txt if necessary
    # (though conda commands above should cover most)
    pip install -r requirements.txt
    # Ensure qrcode is installed if not covered
    pip install qrcode
    ```
    *Note: The `lab_instructions.md` mentions potential DLL conflicts with OpenMP on Windows and suggests renaming a DLL in the conda environment. Refer to it for details if you encounter such issues.*

4.  **Database Setup:**
    *   The project uses PostgreSQL. Ensure you have a PostgreSQL server running.
    *   Database connection details are likely configured within `db.py` or environment variables (not explicitly detailed in provided files, may require inspection).
    *   The `sqldb_manip.sql` file may contain scripts for creating tables or initial data.

5.  **Dataset Preparation (Optional but Recommended for ML features):**
    *   To reset/prepare the dataset:
        ```bash
        python -m reset_dataset_from_clean_copy
        ```
    *   The `archived/` directory and `lab_instructions.md` contain scripts for data augmentation, YOLO dataset preparation, etc. (e.g., `convert_masks_to_yolo.py`, `prepare_yolo_dataset.py`). These are likely for model training or fine-tuning.

6.  **Running the Application:**
    *   **Start the Server:**
        ```bash
        python server.py
        ```
        The server will typically run on `https://0.0.0.0:5000`.
    *   **Start the Server:**
        ```bash
        python server.py
        ```
        The server will typically run on `https://0.0.0.0:5000` and serves the main web interface from the `app/` directory.

    *   **Start the Standalone Client (Optional):**
        The project also includes a separate Flask-based client in `client.py`. This might be used for specific testing or development scenarios. To run it:
        ```bash
        python client.py
        ```
        This client usually connects to the backend server and runs on `https://0.0.0.0:5050`. The primary web interface is typically accessed via the main server.

## Usage

1.  **Start the Server:**
    Follow the instructions in "Setup and Installation" to run `server.py`.

2.  **Access the Web Client:**
    Open your web browser and navigate to the address where the server is running (e.g., `https://localhost:5000` or `https://<server_ip>:5000`). The client interface is served from the `app/` directory.

3.  **Using the Web Interface (`app/templates/index.html` via `main.js`):**
    *   **Connect WebSocket:** Establish a connection to the backend server.
    *   **Camera & Streaming:**
        *   "Start Camera": Activates the local device camera.
        *   "Start Streaming": Sends the camera feed to the server for real-time QR code detection (to identify equipment) and potential continuous analysis.
        *   "Stop Camera"/"Stop Streaming": Deactivates the camera/streaming.
    *   **Capture & Analysis:**
        *   "Capture Frame": Captures a single frame from the video stream and sends it to the server for defect detection. Results are displayed on the interface.
        *   "Upload Image (Defect/QR)": Allows uploading static images for defect or QR code detection.
    *   **Equipment Information:**
        *   Once equipment is identified (usually via QR code scanning during streaming), its details (serial number, model, status) can be viewed.
        *   Documentation for the identified equipment can be fetched and displayed.
    *   **Sensor Data:**
        *   Displays live sensor data (temperature, vibration) for the identified equipment if an ESP32 (real or simulated) is sending data.
    *   **System Health:** Shows the status of the backend server, database, and AI models.
    *   **Voice Control:** Activate voice commands to control the UI (see `voice_control.py` and the UI for available commands).

4.  **Simulating ESP32 Sensor Data:**
    To test the sensor data integration without actual hardware:
    ```bash
    python esp32_sensor_sim.py -id=<SERIAL_NUMBER> -api=https://<server_ip>:5000 -port=<local_sim_port>
    ```
    *   `-id`: Serial number of the equipment to simulate (should match an entry in the database for full functionality).
    *   `-api`: URL of the main AR server (default: `http://127.0.0.1:5000`, **note the default is HTTP, ensure your server is accessible or change to HTTPS if needed**).
    *   `-port`: Local port for the simulator's own status page (default: 8081).

    Example:
    ```bash
    python esp32_sensor_sim.py -id=SN-IM2025007 -api=https://192.168.1.114:5000
    ```
    The simulator will periodically send random sensor data to the specified server endpoint (`/sensors/<SERIAL_NUMBER>`).

5.  **Voice Commands:**
    The web client has a voice control section. Activate "Start Listening" and use commands like:
    *   "show sensors" / "afficher données capteur"
    *   "detect defect" / "détecter anomalie"
    *   "start camera" / "démarrer caméra"
    *   Refer to the "Voice Control" tab in the web UI or `app/templates/index.html` for a more comprehensive list of commands in English and French.

## Dataset

This project utilizes the MVTec Screw dataset for training and testing defect detection models.
For detailed information, attribution, and license, please refer to `dataset/screw/readme.txt`.

**License for MVTec Dataset:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## Archived Components

The `archived/` directory contains scripts and code that may be from earlier development stages, experiments, or specific data processing pipelines (e.g., YOLO dataset preparation, older training scripts). These are generally not part of the main operational application but might be useful for understanding the project's evolution or for specific development tasks.

## License

This project is licensed under the terms specified in the `LICENSE` file. (Please ensure a `LICENSE` file exists at the root of the project and accurately reflects the desired licensing terms).

## Acknowledgments

*   The MVTec Screw dataset is used as a key resource for anomaly detection. See `dataset/screw/readme.txt` for citation details.

## Contact

(To be filled in if there's a designated contact person or email for inquiries about the project.)

---

*This README was generated based on an analysis of the project's codebase. Some details, especially regarding specific configurations (e.g., database connection strings) or operational nuances, might require further manual verification by the project maintainers.*
