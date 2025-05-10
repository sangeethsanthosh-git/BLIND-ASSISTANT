from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import logging
import os
from ultralytics import YOLO

app = Flask(__name__)

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Load YOLOv8 model
try:
    model = YOLO('runs/detect/train/weights/best.pt')
    logging.info("YOLOv8 model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load YOLOv8 model: {str(e)}")
    model = None

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Generate video frames
def gen_frames():
    camera = None
    # Try camera indices 0 to 4 with DirectShow backend
    for i in range(5):
        camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        if camera.isOpened():
            logging.info(f"Camera opened successfully on index {i}")
            break
        camera.release()
    
    if not camera or not camera.isOpened():
        logging.error("Error: Could not open any webcam. Check device, drivers, and permissions.")
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            logging.warning("Failed to read frame")
            break
        else:
            # Placeholder for monocular depth estimation (from Blind_Assistance_System.pdf)
            # Example: Use MiDaS model for depth (requires additional setup)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()
    logging.info("Camera released")

# Route to trigger welcome note
@app.route('/welcome')
def welcome():
    return jsonify({'message': 'Hi, welcome to Visual Assistant, let’s go'})

# Route for object detection using YOLOv8
@app.route('/detect_objects')
def detect_objects():
    if not model:
        return jsonify({'error': 'YOLOv8 model not loaded'}), 500
    
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        logging.error("Error: Could not open webcam for detection")
        return jsonify({'error': 'Could not open webcam'}), 500
    
    success, frame = camera.read()
    if not success:
        logging.warning("Failed to read frame for detection")
        camera.release()
        return jsonify({'error': 'Failed to read frame'}), 500
    
    # Perform detection with YOLOv8
    results = model(frame)
    objects = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            objects.append(label)
    
    camera.release()
    return jsonify({'objects': objects})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')