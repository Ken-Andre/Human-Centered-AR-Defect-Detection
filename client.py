import socketio # Import the Socket.IO client library
import json
import base64
import cv2
import numpy as np
import time # Import time for a slight delay if needed

# Create a Socket.IO client instance
sio = socketio.Client()

# Event handler for successful connection
@sio.event
def connect():
    try:
        print('Connection established with Socket.IO server!')
        # Load a test image
        img = cv2.imread("test_media/qr_code_5.png")
        if img is None:
            print("Error: Could not load image from test_media/qr_code_5.png")
            sio.disconnect()
            return

        _, buffer = cv2.imencode('.png', img)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        # Emit the 'video_frame' event with the frame data
        # The data dictionary directly matches what handle_video_frame expects
        sio.emit('video_frame', {'frame': frame_data})
    except Exception as e:
        print(f"Error sending frame: {e}")
        sio.disconnect()

@sio.event
def disconnect():
    print('Disconnected from Socket.IO server.')

# Event handler for 'qr_result' from the server
@sio.event
def qr_result(data):
    print("Server response (QR result):", data)
    sio.disconnect() # Disconnect after receiving the response

# Connect to the Flask-SocketIO server
# The URL should be the base URL of your Flask app, Socket.IO handles the path
try:
    sio.connect('http://127.0.0.1:5000',wait_timeout=10) # Connect to the base URL
    # Keep the client running to listen for events
    sio.wait()
except Exception as e:
    print(f"Failed to connect to Socket.IO server: {e}")